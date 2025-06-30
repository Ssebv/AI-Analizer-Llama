"""
specialized_agents.py
Agentes especializados para análisis de noticias usando Llama
"""
import logging
from typing import Dict, List, Any, Optional
from models.llama_setup import LlamaNewsAnalyzer, ChileanNewsPrompts, setup_llama_for_news_analysis

logger = logging.getLogger(__name__)

class NewsAnalysisAgent:
    """Agente especializado en análisis de noticias"""
    
    def __init__(self, config: str = "development"):
        self.analyzer = setup_llama_for_news_analysis()
        self.prompts = ChileanNewsPrompts()
        self.config = config
        
    def analyze_news(self, question: str, context: str = "") -> str:
        """Analiza noticias basado en una pregunta específica"""
        prompt = self.prompts.ANALYSIS_PROMPT.format(
            system_prompt=self.prompts.SYSTEM_PROMPT,
            context=context,
            question=question
        )
        
        try:
            response = self.analyzer.get_llm().invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error en análisis de noticias: {e}")
            return f"Error procesando consulta: {e}"
    
    def summarize_news(self, news_content: str) -> str:
        """Crea resumen ejecutivo de noticias"""
        prompt = self.prompts.SUMMARY_PROMPT.format(
            system_prompt=self.prompts.SYSTEM_PROMPT,
            news_content=news_content
        )
        
        try:
            response = self.analyzer.get_llm().invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error en resumen: {e}")
            return f"Error generando resumen: {e}"

class DataProcessingAgent:
    """Agente especializado en procesamiento de datos"""
    
    def __init__(self, config: str = "fast"):
        # Configuración más rápida para procesamiento de datos
        from models.llama_setup import CONFIGS
        custom_config = CONFIGS[config]
        self.analyzer = setup_llama_for_news_analysis(
            custom_config=custom_config
        )
        self.prompts = ChileanNewsPrompts()
    
    def process_temporal_data(self, temporal_data: str) -> str:
        """Procesa datos temporales para identificar patrones"""
        prompt = self.prompts.TEMPORAL_ANALYSIS_PROMPT.format(
            system_prompt=self.prompts.SYSTEM_PROMPT,
            temporal_data=temporal_data
        )
        
        try:
            response = self.analyzer.get_llm().invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error en análisis temporal: {e}")
            return f"Error procesando datos temporales: {e}"
    
    def extract_insights(self, data: str, focus_area: str = "") -> str:
        """Extrae insights específicos de los datos"""
        custom_prompt = f"""
        {self.prompts.SYSTEM_PROMPT}
        
        Datos a analizar: {data}
        Área de enfoque: {focus_area}
        
        Extrae insights clave, patrones importantes y hallazgos relevantes.
        Proporciona recomendaciones accionables basadas en los datos.
        
        Insights:
        """
        
        try:
            response = self.analyzer.get_llm().invoke(custom_prompt)
            return response
        except Exception as e:
            logger.error(f"Error extrayendo insights: {e}")
            return f"Error en extracción de insights: {e}"

class SearchAgent:
    """Agente especializado en búsquedas y filtrado"""
    
    def __init__(self, config: str = "fast"):
        from models.llama_setup import CONFIGS
        custom_config = CONFIGS[config]
        self.analyzer = setup_llama_for_news_analysis(
            custom_config=custom_config
        )
        self.prompts = ChileanNewsPrompts()
    
    def search_and_filter(self, query: str, data: str, filters: Dict[str, Any] = None) -> str:
        """Busca y filtra información según criterios específicos"""
        filter_text = ""
        if filters:
            filter_text = f"Filtros aplicar: {filters}"
        
        search_prompt = f"""
        {self.prompts.SYSTEM_PROMPT}
        
        Consulta de búsqueda: {query}
        {filter_text}
        
        Datos disponibles: {data}
        
        Busca información relevante que coincida con la consulta.
        Aplica los filtros especificados y presenta resultados organizados.
        
        Resultados:
        """
        
        try:
            response = self.analyzer.get_llm().invoke(search_prompt)
            return response
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return f"Error en búsqueda: {e}"

class AgentManager:
    """Gestor central de agentes"""
    
    def __init__(self):
        self.agents = {
            'news': NewsAnalysisAgent(),
            'data': DataProcessingAgent(),
            'search': SearchAgent()
        }
        self.default_agent = 'news'
    
    def get_agent(self, agent_type: str):
        """Obtiene un agente específico"""
        return self.agents.get(agent_type, self.agents[self.default_agent])
    
    def process_query(self, query: str, agent_type: str = None, context: str = "", **kwargs) -> str:
        """Procesa una consulta usando el agente apropiado"""
        if not agent_type:
            agent_type = self._determine_agent_type(query)
        
        agent = self.get_agent(agent_type)
        
        # Procesar según el tipo de agente
        if agent_type == 'news':
            return agent.analyze_news(query, context)
        elif agent_type == 'data':
            if 'temporal_data' in kwargs:
                return agent.process_temporal_data(kwargs['temporal_data'])
            else:
                return agent.extract_insights(context, query)
        elif agent_type == 'search':
            filters = kwargs.get('filters', {})
            return agent.search_and_filter(query, context, filters)
        
        return "Tipo de agente no reconocido"
    
    def _determine_agent_type(self, query: str) -> str:
        """Determina automáticamente el tipo de agente basado en la consulta"""
        query_lower = query.lower()
        
        # Palabras clave para diferentes tipos de análisis
        temporal_keywords = ['temporal', 'tiempo', 'patrones', 'tendencias', 'histórico']
        search_keywords = ['buscar', 'encontrar', 'filtrar', 'específico']
        
        if any(keyword in query_lower for keyword in temporal_keywords):
            return 'data'
        elif any(keyword in query_lower for keyword in search_keywords):
            return 'search'
        else:
            return 'news'  # Por defecto
    
    def list_agents(self) -> List[str]:
        """Lista todos los agentes disponibles"""
        return list(self.agents.keys())
    
    def health_check(self) -> Dict[str, bool]:
        """Verifica el estado de todos los agentes"""
        status = {}
        for agent_name, agent in self.agents.items():
            try:
                # Test básico de conexión
                test_response = agent.analyzer.test_connection()
                status[agent_name] = test_response
            except Exception as e:
                logger.error(f"Error en health check de {agent_name}: {e}")
                status[agent_name] = False
        
        return status

# Función helper para inicializar el sistema de agentes
def initialize_agents(config_env: str = "development") -> AgentManager:
    """Inicializa el sistema de agentes con configuración específica"""
    try:
        manager = AgentManager()
        
        # Verificar que todos los agentes estén funcionando
        health = manager.health_check()
        
        if not all(health.values()):
            logger.warning(f"Algunos agentes no están funcionando correctamente: {health}")
        else:
            logger.info("Todos los agentes inicializados correctamente")
        
        return manager
        
    except Exception as e:
        logger.error(f"Error inicializando agentes: {e}")
        raise