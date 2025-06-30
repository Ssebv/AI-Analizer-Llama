"""
core/system.py
Módulo central que encapsula todo el pipeline de análisis de noticias.

✔  Carga y limpieza de datos
✔  Creación / carga del vector‑store
✔  Creación de la cadena RAG
✔  Orquestación de agentes (base y especializados)

Tanto la CLI (`main.py`) como la aplicación Streamlit (`app.py`)
deben importar y reutilizar la clase `IntegratedNewsSystem` que
se expone aquí, evitando duplicación de lógica.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

# Dependencias internas ────────────────────────────────────────────────
from data.data_processor import NewsDataProcessor
from data.vectorstore_manager import NewsVectorStoreManager, NewsRAGChain

from agents.llama_agents import (
    create_text_agent,
    create_analysis_agent,
    create_conversational_agent,
    test_all_agents,
)
from agents.specialized_agents import initialize_agents

# ───────────────────────────── logger global ──────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class IntegratedNewsSystem:
    """
    Motor central reutilizable.
    No imprime directamente — devuelve estados/errores y deja el
    front‑end decidir qué mostrar.
    """

    def __init__(self, excel_path: str | Path, *, auto_init: bool = False) -> None:
        load_dotenv()  # permite configurar vía .env
        self.excel_path = Path(excel_path)

        # ─ componentes ─
        self.processor: NewsDataProcessor | None = None
        self.vectorstore_manager: NewsVectorStoreManager | None = None
        self.rag_chain: NewsRAGChain | None = None

        self.base_agents: Dict[str, Any] = {}
        self.specialized_manager = None

        self.initialized: bool = False
        if auto_init:
            self.initialize_system()

    # ---------------------------------------------------------------- #
    #                           PÚBLICOS                                #
    # ---------------------------------------------------------------- #
    def initialize_system(self) -> bool:
        """Inicializa datos, vector‑store, RAG y agentes."""
        try:
            self._initialize_data()
            self._initialize_vectorstore_and_rag()
            self._initialize_agents()
            self.initialized = True
            logger.info("✅ Sistema completamente inicializado")
            return True
        except Exception as exc:  # pragma: no cover
            logger.exception("❌ Falló la inicialización: %s", exc)
            self.initialized = False
            return False

    def query_with_agent(self, query: str, agent_type: str = "auto", **kwargs) -> Dict[str, Any]:
        """
        Devuelve respuesta y metadatos.
        El front‑end (CLI o Streamlit) decide qué mostrar.
        """
        if not self.initialized:
            return {"error": "Sistema no inicializado"}

        # 1) Selección automática de agente
        if agent_type == "auto":
            agent_type = self._determine_best_agent(query.lower())

        # 2) Delegar al componente adecuado
        if agent_type == "rag":
            return self.rag_chain.query(query)

        if agent_type in self.base_agents:
            return {
                "answer": self.base_agents[agent_type].process(query, **kwargs),
                "agent_used": agent_type,
            }

        if self.specialized_manager and agent_type in self.specialized_manager.list_agents():
            return {
                "answer": self.specialized_manager.process_query(query, agent_type, **kwargs),
                "agent_used": agent_type,
            }

        return {"error": f"Agente no reconocido: {agent_type}"}

    def get_system_status(self) -> Dict[str, Any]:
        """Snapshot de salud para la UI."""
        return {
            "initialized": self.initialized,
            "dataset_ok": self.processor is not None,
            "vectorstore_ok": self.vectorstore_manager is not None,
            "rag_ok": self.rag_chain is not None,
            "base_agents": list(self.base_agents.keys()),
            "specialized_agents": self.specialized_manager.list_agents() if self.specialized_manager else [],
        }

    # ---------------------------------------------------------------- #
    #                       MÉTODOS PRIVADOS                            #
    # ---------------------------------------------------------------- #
    def _initialize_data(self) -> None:
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel no encontrado → {self.excel_path}")

        self.processor = NewsDataProcessor(str(self.excel_path))
        self.processor.clean_and_preprocess()
        logger.info("Dataset cargado con %s filas", len(self.processor.df))

    def _initialize_vectorstore_and_rag(self) -> None:
        assert self.processor is not None, "Procesador no inicializado"
        documents = self.processor.create_documents_for_vectorstore()

        self.vectorstore_manager = NewsVectorStoreManager(
            model_name=os.getenv("LLAMA_MODEL", "llama3.1:8b"),
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
            collection_name=os.getenv("COLLECTION_NAME", "news_collection"),
        )

        store = self.vectorstore_manager.load_existing_vectorstore()
        if store is None or store._collection.count() == 0:
            logger.info("Creando vector‑store porque no existe o está vacío…")
            self.vectorstore_manager.create_vectorstore(documents)

        self.rag_chain = NewsRAGChain(self.vectorstore_manager)
        self.rag_chain.create_chain()

    def _initialize_agents(self) -> None:
        # Agentes base
        config_env = os.getenv("AGENT_CONFIG_ENV", "development")
        test_all_agents(config_env)  # logs de salud

        self.base_agents = {
            "text": create_text_agent("NewsTextAgent", config_env),
            "analysis": create_analysis_agent("NewsAnalysisAgent", config_env),
            "conversational": create_conversational_agent("NewsConversationalAgent", config_env),
        }

        # Agentes especializados
        self.specialized_manager = initialize_agents(config_env)

    # ------------------ heurística de selección ----------------------
    @staticmethod
    def _determine_best_agent(query_lower: str) -> str:
        if any(k in query_lower for k in ("buscar", "fuentes", "qué dice")):
            return "rag"
        if any(k in query_lower for k in ("analizar", "patrones", "insights")):
            return "analysis"
        if any(k in query_lower for k in ("explica", "ayuda", "cómo", "dime")):
            return "conversational"
        return "rag"