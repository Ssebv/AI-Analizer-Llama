#!/usr/bin/env python3
"""
main.py
Script principal para inicializar el sistema de análisis de noticias con Llama
Integrado con sistema de agentes especializados
"""

# --- CONFIGURACIÓN CHROMADB (DEBE IR PRIMERO) ----------------------------
import os
import logging
import warnings

# Deshabilitar telemetría de ChromaDB ANTES de cualquier import
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configurar logging para suprimir mensajes de telemetría
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# Suprimir warnings específicos
warnings.filterwarnings("ignore", category=UserWarning, module="chromadb")
warnings.filterwarnings("ignore", message=".*telemetry.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain.memory")
warnings.filterwarnings("ignore", message=".*migration guide.*")

# --- Path configuration --------------------------------------------------
from pathlib import Path
import sys

# Aseguramos que la carpeta src esté en el PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR  = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

# Ruta por defecto al Excel
DEFAULT_EXCEL_PATH = SRC_DIR / "data" / "noticias_test_ingeniero_IA.xlsx"

# --- Carga de dependencias ------------------------------------------------
from dotenv import load_dotenv

from data.data_processor import NewsDataProcessor
from data.vectorstore_manager import NewsVectorStoreManager, NewsRAGChain

# Importar sistema de agentes
from agents.llama_agents import (
    AgentFactory, 
    create_text_agent, 
    create_analysis_agent, 
    create_conversational_agent,
    test_all_agents
)
from agents.specialized_agents import (
    AgentManager as SpecializedAgentManager,
    initialize_agents
)


class IntegratedNewsSystem:
    """Sistema integrado de análisis de noticias con agentes especializados"""
    
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.processor = None
        self.vectorstore_manager = None
        self.rag_chain = None
        self.base_agents = {}
        self.specialized_manager = None
        self.is_initialized = False
        
    def initialize_system(self) -> bool:
        """Inicializa todo el sistema: datos, vectorstore, RAG y agentes"""
        try:
            print("🚀 Inicializando sistema integrado de análisis de noticias...")
            
            # 1. Verificar Ollama
            if not self._check_ollama_connection():
                return False
                
            # 2. Procesar datos
            if not self._initialize_data_processing():
                return False
                
            # 3. Configurar vectorstore y RAG
            if not self._initialize_vectorstore_and_rag():
                return False
                
            # 4. Inicializar agentes
            if not self._initialize_agents():
                return False
                
            self.is_initialized = True
            print("✅ Sistema completamente inicializado!")
            return True
            
        except Exception as e:
            print(f"❌ Error durante inicialización: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_ollama_connection(self) -> bool:
        """Verifica conexión con Ollama"""
        print("🔍 Verificando conexión con Ollama…")
        try:
            import requests
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama está ejecutándose")
                return True
            else:
                print("❌ Ollama no responde correctamente")
                return False
        except requests.exceptions.RequestException as exc:
            print(f"❌ Error conectando con Ollama: {exc}")
            print("💡 Asegúrate de que Ollama esté ejecutándose: ollama serve")
            return False
        except Exception as exc:
            print(f"❌ Error inesperado con Ollama: {exc}")
            return False
    
    def _initialize_data_processing(self) -> bool:
        """Inicializa el procesamiento de datos"""
        try:
            # Verificar archivo Excel
            if not os.path.exists(self.excel_path):
                print(f"❌ Archivo Excel no encontrado: {self.excel_path}")
                return False
            print(f"✅ Archivo Excel encontrado: {self.excel_path}")

            # Procesar datos
            print("\n📊 Inicializando procesador de noticias…")
            self.processor = NewsDataProcessor(self.excel_path)
            print("🔄 Procesando datos…")
            df = self.processor.clean_and_preprocess()

            # Mostrar resumen
            summary = self.processor.get_statistical_summary()
            print("\n📈 Resumen del dataset:")
            print(f"   • Total noticias: {summary['total_noticias']}")
            print(f"   • Rango fechas:  {summary['rango_fechas']['inicio']} → {summary['rango_fechas']['fin']}")
            print(f"   • Medios únicos: {summary['medios_unicos']}")
            print(f"   • Secciones únicas: {summary['secciones_unicas']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en procesamiento de datos: {e}")
            return False
    
    def _initialize_vectorstore_and_rag(self) -> bool:
        """Inicializa vectorstore y sistema RAG"""
        try:
            # Crear documentos
            print("\n🔄 Creando documentos para vector store…")
            documents = self.processor.create_documents_for_vectorstore(
                chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
            )

            # Inicializar vectorstore
            print("\n🗄️ Inicializando vector store…")
            self.vectorstore_manager = NewsVectorStoreManager(
                model_name=os.getenv("LLAMA_MODEL", "llama3:8b"),
                persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
                collection_name=os.getenv("COLLECTION_NAME", "news_collection"),
            )

            print("🔄 Configurando base de datos vectorial…")
            store = self.vectorstore_manager.load_existing_vectorstore()

            # Crear o verificar vectorstore
            if store is None:
                print("📝 No se encontró colección. Creando vector store…")
                self.vectorstore_manager.create_vectorstore(documents)
            else:
                try:
                    doc_count = store._collection.count()
                    if doc_count == 0:
                        print("⚠️ Colección vacía. Reconstruyendo vector store…")
                        self.vectorstore_manager.create_vectorstore(documents)
                    else:
                        print(f"✅ Vector store existente cargado con {doc_count} documentos")
                except Exception as e:
                    print(f"⚠️ Error verificando colección: {e}")
                    print("📝 Reconstruyendo vector store…")
                    self.vectorstore_manager.create_vectorstore(documents)

            # Mostrar estadísticas
            stats = self.vectorstore_manager.get_collection_stats()
            print("\n📊 Estadísticas del vector store:")
            print(f"   • Total documentos: {stats.get('total_documents', 'N/A')}")
            print(f"   • Estado: {stats.get('status', 'N/A')}")

            # Inicializar RAG
            print("\n🤖 Inicializando sistema RAG…")
            self.rag_chain = NewsRAGChain(self.vectorstore_manager)
            self.rag_chain.create_chain()
            
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando vectorstore/RAG: {e}")
            return False
    
    def _initialize_agents(self) -> bool:
        """Inicializa sistema de agentes"""
        try:
            print("\n🤖 Inicializando sistema de agentes…")
            
            # 1. Probar conexión de agentes base
            print("🔧 Probando agentes base…")
            config_env = os.getenv("AGENT_CONFIG_ENV", "development")
            agent_test_results = test_all_agents(config_env)
            
            if not all(agent_test_results.values()):
                print(f"⚠️ Algunos agentes base no funcionan: {agent_test_results}")
            else:
                print("✅ Todos los agentes base funcionando")
            
            # 2. Crear agentes base
            self.base_agents = {
                'text': create_text_agent("NewsTextAgent", config_env),
                'analysis': create_analysis_agent("NewsAnalysisAgent", config_env),
                'conversational': create_conversational_agent("NewsConversationalAgent", config_env)
            }
            
            # 3. Inicializar agentes especializados
            print("🔧 Inicializando agentes especializados…")
            self.specialized_manager = initialize_agents(config_env)
            
            # 4. Verificar estado de agentes especializados
            health_status = self.specialized_manager.health_check()
            if not all(health_status.values()):
                print(f"⚠️ Algunos agentes especializados no funcionan: {health_status}")
            else:
                print("✅ Todos los agentes especializados funcionando")
            
            print("\n🤖 Sistema de agentes inicializado:")
            print(f"   • Agentes base: {list(self.base_agents.keys())}")
            print(f"   • Agentes especializados: {self.specialized_manager.list_agents()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error inicializando agentes: {e}")
            return False
    
    def query_with_agent(self, query: str, agent_type: str = "auto", **kwargs) -> dict:
        """Procesa consulta usando el agente apropiado"""
        if not self.is_initialized:
            return {"error": "Sistema no inicializado"}
        
        try:
            # Determinar agente automáticamente si es necesario
            if agent_type == "auto":
                agent_type = self._determine_best_agent(query)
            
            print(f"🔍 Procesando con agente: {agent_type}")
            
            # Procesar según tipo de agente
            if agent_type in self.base_agents:
                return self._process_with_base_agent(query, agent_type, **kwargs)
            elif agent_type in self.specialized_manager.list_agents():
                return self._process_with_specialized_agent(query, agent_type, **kwargs)
            elif agent_type == "rag":
                return self._process_with_rag(query)
            else:
                return {"error": f"Tipo de agente no reconocido: {agent_type}"}
                
        except Exception as e:
            return {"error": f"Error procesando consulta: {e}"}
    
    def _determine_best_agent(self, query: str) -> str:
        """Determina el mejor agente para la consulta"""
        query_lower = query.lower()
        
        # Palabras clave para diferentes tipos
        rag_keywords = ['buscar', 'encontrar', 'fuentes', 'noticias sobre', 'qué dice']
        analysis_keywords = ['analizar', 'análisis', 'patrones', 'tendencias', 'insights']
        conversational_keywords = ['explica', 'ayuda', 'cómo', 'por qué', 'dime']
        
        if any(keyword in query_lower for keyword in rag_keywords):
            return "rag"
        elif any(keyword in query_lower for keyword in analysis_keywords):
            return "analysis"
        elif any(keyword in query_lower for keyword in conversational_keywords):
            return "conversational"
        else:
            return "rag"  # Por defecto usar RAG para búsquedas
    
    def _process_with_base_agent(self, query: str, agent_type: str, **kwargs) -> dict:
        """Procesa con agentes base"""
        agent = self.base_agents[agent_type]
        
        # Preparar contexto si es necesario
        context = kwargs.get('context', '')
        if not context and hasattr(self, 'processor'):
            # Obtener contexto básico del dataset
            summary = self.processor.get_statistical_summary()
            context = f"Dataset: {summary['total_noticias']} noticias, {summary['medios_unicos']} medios"
        
        # Procesar según tipo
        if agent_type == 'conversational':
            result = agent.process(query, **kwargs)
        else:
            # Para text y analysis, incluir contexto
            input_data = {
                "prompt": query,
                "context": context
            }
            result = agent.process(input_data, **kwargs)
        
        return {
            "answer": result,
            "agent_used": agent_type,
            "agent_name": agent.agent_name
        }
    
    def _process_with_specialized_agent(self, query: str, agent_type: str, **kwargs) -> dict:
        """Procesa con agentes especializados"""
        context = kwargs.get('context', '')
        result = self.specialized_manager.process_query(
            query, 
            agent_type, 
            context, 
            **kwargs
        )
        
        return {
            "answer": result,
            "agent_used": agent_type,
            "agent_name": f"Specialized{agent_type.capitalize()}Agent"
        }
    
    def _process_with_rag(self, query: str) -> dict:
        """Procesa con sistema RAG tradicional"""
        result = self.rag_chain.query(query)
        result["agent_used"] = "rag"
        result["agent_name"] = "NewsRAGChain"
        return result
    
    def get_system_status(self) -> dict:
        """Obtiene estado completo del sistema"""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        status = {
            "status": "initialized",
            "components": {
                "data_processor": bool(self.processor),
                "vectorstore": bool(self.vectorstore_manager),
                "rag_chain": bool(self.rag_chain),
                "base_agents": len(self.base_agents),
                "specialized_agents": len(self.specialized_manager.list_agents()) if self.specialized_manager else 0
            }
        }
        
        # Estadísticas del dataset
        if self.processor:
            summary = self.processor.get_statistical_summary()
            status["dataset"] = summary
        
        # Estado de agentes
        if self.base_agents:
            status["base_agents_status"] = {
                name: agent.get_status() for name, agent in self.base_agents.items()
            }
        
        if self.specialized_manager:
            status["specialized_agents_status"] = self.specialized_manager.health_check()
        
        return status
    
    def list_available_agents(self) -> dict:
        """Lista todos los agentes disponibles"""
        agents = {
            "base_agents": list(self.base_agents.keys()) if self.base_agents else [],
            "specialized_agents": self.specialized_manager.list_agents() if self.specialized_manager else [],
            "rag": ["rag"]
        }
        
        agents["all"] = agents["base_agents"] + agents["specialized_agents"] + agents["rag"]
        return agents


def main() -> bool:
    """Función principal"""
    load_dotenv()
    
    excel_path = os.getenv("EXCEL_PATH", str(DEFAULT_EXCEL_PATH))
    system = IntegratedNewsSystem(excel_path)
    
    if system.initialize_system():
        # Ejemplo de consulta
        test_query = "¿Cuáles son las principales noticias sobre deportes?"
        print("\n🔍 Ejemplo de consulta:")
        print(f"Pregunta: {test_query}")
        
        result = system.query_with_agent(test_query)
        
        if "error" not in result:
            print(f"\nAgente usado: {result.get('agent_name', 'N/A')}")
            print(f"Respuesta: {result['answer'][:300]}...")
            if 'sources' in result and result['sources']:
                print(f"Fuentes: {len(result['sources'])} encontradas")
        else:
            print(f"Error: {result['error']}")
        
        return True
    
    return False


def interactive_mode():
    """Modo interactivo mejorado con selección de agentes"""
    load_dotenv()
    
    try:
        excel_path = os.getenv("EXCEL_PATH", str(DEFAULT_EXCEL_PATH))
        system = IntegratedNewsSystem(excel_path)
        
        print("🔄 Inicializando sistema…")
        if not system.initialize_system():
            print("❌ Error inicializando sistema")
            return
        
        print("\n🤖 Sistema integrado listo!")
        print("💬 Comandos disponibles:")
        print("   • 'agentes' - Ver agentes disponibles")
        print("   • 'estado' - Ver estado del sistema")
        print("   • 'usar [agente]' - Cambiar agente (ej: 'usar analysis')")
        print("   • 'auto' - Modo automático (selección automática de agente)")
        print("   • 'limpiar' - Limpiar historial")
        print("   • 'salir' - Terminar")
        print("=" * 60)
        
        current_agent = "auto"
        
        while True:
            try:
                query = input(f"\n💬 [{current_agent}] Tu pregunta: ").strip()
                
                # Comandos especiales
                if query.lower() in {"salir", "exit", "quit"}:
                    print("👋 ¡Hasta luego!")
                    break
                elif query.lower() == "agentes":
                    agents = system.list_available_agents()
                    print("\n🤖 Agentes disponibles:")
                    for category, agent_list in agents.items():
                        if category != "all":
                            print(f"   {category}: {', '.join(agent_list)}")
                    continue
                elif query.lower() == "estado":
                    status = system.get_system_status()
                    print(f"\n📊 Estado del sistema:")
                    print(f"   • Estado: {status['status']}")
                    if 'dataset' in status:
                        print(f"   • Noticias: {status['dataset']['total_noticias']}")
                    print(f"   • Agentes base: {status['components']['base_agents']}")
                    print(f"   • Agentes especializados: {status['components']['specialized_agents']}")
                    continue
                elif query.lower().startswith("usar "):
                    new_agent = query[5:].strip()
                    available = system.list_available_agents()["all"]
                    if new_agent in available:
                        current_agent = new_agent
                        print(f"✅ Cambiado a agente: {current_agent}")
                    else:
                        print(f"❌ Agente no disponible. Disponibles: {', '.join(available)}")
                    continue
                elif query.lower() == "auto":
                    current_agent = "auto"
                    print("✅ Modo automático activado")
                    continue
                elif query.lower() == "limpiar":
                    # Limpiar historial de agentes conversacionales
                    for agent in system.base_agents.values():
                        if hasattr(agent, 'clear_history'):
                            agent.clear_history()
                    if system.rag_chain:
                        system.rag_chain.clear_history()
                    print("✅ Historial limpiado")
                    continue
                
                if not query:
                    continue
                
                print("🔍 Procesando…")
                result = system.query_with_agent(query, current_agent)
                
                if "error" in result:
                    print(f"❌ {result['error']}")
                else:
                    print(f"\n🤖 [{result.get('agent_name', 'N/A')}]:")
                    print(result['answer'])
                    
                    if 'sources' in result and result['sources']:
                        print(f"\n📚 Fuentes ({len(result['sources'])}):")
                        for i, src in enumerate(result['sources'][:3], 1):
                            print(f"   {i}. {src['titulo']} — {src['medio']} ({src['fecha']})")
                            
            except KeyboardInterrupt:
                print("\n👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    except Exception as e:
        print(f"❌ Error en modo interactivo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        if main():
            print("\n💡 Para usar el modo interactivo ejecuta:\n   python main.py interactive")