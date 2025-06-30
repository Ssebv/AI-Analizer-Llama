"""
llama_agents.py
Agentes base para el sistema Llama
Estos son los agentes fundamentales que pueden ser extendidos por agentes especializados
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from models.llama_setup import LlamaNewsAnalyzer, ChileanNewsPrompts, setup_llama_for_news_analysis, CONFIGS

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Clase base abstracta para todos los agentes"""
    
    def __init__(
        self, 
        agent_name: str,
        config_env: str = "development",
        custom_config: Optional[Dict[str, Any]] = None
    ):
        self.agent_name = agent_name
        self.config_env = config_env
        self.custom_config = custom_config or {}
        self.analyzer = None
        self.prompts = ChileanNewsPrompts()
        self._setup_analyzer()
    
    def _setup_analyzer(self):
        """Configura el analizador Llama para este agente"""
        try:
            # Usar configuración personalizada si existe, sino usar predefinida
            if self.custom_config:
                config = self.custom_config
            else:
                config = CONFIGS.get(self.config_env, CONFIGS["development"])

            # Evitar pasar 'model_name' dos veces
            model_name = config.get("model_name", "llama3.1:8b")
            config_no_model = {k: v for k, v in config.items() if k != "model_name"}

            self.analyzer = setup_llama_for_news_analysis(
                model_name=model_name,
                custom_config=config_no_model
            )

            logger.info(f"Agente {self.agent_name} configurado con {self.config_env}")

        except Exception as e:
            logger.error(f"Error configurando agente {self.agent_name}: {e}")
            raise
    
    @abstractmethod
    def process(self, input_data: Union[str, Dict[str, Any]], **kwargs) -> str:
        """Método abstracto que debe implementar cada agente"""
        pass
    
    def test_connection(self) -> bool:
        """Prueba la conexión del agente"""
        try:
            return self.analyzer.test_connection()
        except Exception as e:
            logger.error(f"Error en test de conexión para {self.agent_name}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Obtiene el estado del agente"""
        return {
            "agent_name": self.agent_name,
            "config_env": self.config_env,
            "connected": self.test_connection(),
            "model": self.analyzer.model_name if self.analyzer else "No configurado"
        }

class LlamaTextAgent(BaseAgent):
    """Agente base para procesamiento de texto con Llama"""
    
    def __init__(self, agent_name: str = "TextAgent", **kwargs):
        super().__init__(agent_name, **kwargs)
        self.conversation_history = []
    
    def process(self, input_data: Union[str, Dict[str, Any]], **kwargs) -> str:
        """Procesa texto usando el modelo Llama"""
        try:
            # Preparar el prompt
            if isinstance(input_data, str):
                prompt = input_data
            else:
                prompt = input_data.get("prompt", "")
            
            # Agregar contexto del sistema si no está presente
            if not prompt.startswith(self.prompts.SYSTEM_PROMPT):
                full_prompt = f"{self.prompts.SYSTEM_PROMPT}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Procesar con Llama
            response = self.analyzer.get_llm().invoke(full_prompt)
            
            # Guardar en historial si se requiere
            if kwargs.get("keep_history", False):
                self.conversation_history.append({
                    "input": prompt,
                    "output": response,
                    "timestamp": kwargs.get("timestamp")
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Error procesando en {self.agent_name}: {e}")
            return f"Error: {e}"
    
    def clear_history(self):
        """Limpia el historial de conversación"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Obtiene el historial de conversación"""
        return self.conversation_history.copy()

class LlamaAnalysisAgent(BaseAgent):
    """Agente base para análisis con capacidades avanzadas"""
    
    def __init__(self, agent_name: str = "AnalysisAgent", **kwargs):
        super().__init__(agent_name, **kwargs)
        self.analysis_cache = {}
    
    def process(self, input_data: Union[str, Dict[str, Any]], **kwargs) -> str:
        """Procesa análisis usando técnicas estructuradas"""
        try:
            analysis_type = kwargs.get("analysis_type", "general")
            
            # Verificar cache si está habilitado
            if kwargs.get("use_cache", True):
                cache_key = self._generate_cache_key(input_data, analysis_type)
                if cache_key in self.analysis_cache:
                    logger.info(f"Resultado obtenido desde cache para {cache_key}")
                    return self.analysis_cache[cache_key]
            
            # Realizar análisis
            result = self._perform_analysis(input_data, analysis_type, **kwargs)
            
            # Guardar en cache
            if kwargs.get("use_cache", True):
                self.analysis_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error en análisis de {self.agent_name}: {e}")
            return f"Error en análisis: {e}"
    
    def _perform_analysis(self, input_data: Union[str, Dict[str, Any]], analysis_type: str, **kwargs) -> str:
        """Realiza el análisis específico"""
        
        # Preparar datos de entrada
        if isinstance(input_data, dict):
            data = input_data.get("data", "")
            context = input_data.get("context", "")
        else:
            data = str(input_data)
            context = kwargs.get("context", "")
        
        # Seleccionar prompt según tipo de análisis
        if analysis_type == "summary":
            prompt = self.prompts.SUMMARY_PROMPT.format(
                system_prompt=self.prompts.SYSTEM_PROMPT,
                news_content=data
            )
        elif analysis_type == "temporal":
            prompt = self.prompts.TEMPORAL_ANALYSIS_PROMPT.format(
                system_prompt=self.prompts.SYSTEM_PROMPT,
                temporal_data=data
            )
        else:  # general analysis
            prompt = self.prompts.ANALYSIS_PROMPT.format(
                system_prompt=self.prompts.SYSTEM_PROMPT,
                context=context,
                question=data
            )
        
        # Ejecutar análisis
        response = self.analyzer.get_llm().invoke(prompt)
        return response
    
    def _generate_cache_key(self, input_data: Union[str, Dict[str, Any]], analysis_type: str) -> str:
        """Genera clave para cache"""
        import hashlib
        
        # Convertir input a string consistente
        if isinstance(input_data, dict):
            data_str = str(sorted(input_data.items()))
        else:
            data_str = str(input_data)
        
        # Crear hash
        content = f"{data_str}_{analysis_type}_{self.agent_name}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def clear_cache(self):
        """Limpia el cache de análisis"""
        self.analysis_cache = {}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache"""
        return {
            "entries": len(self.analysis_cache),
            "keys": list(self.analysis_cache.keys())
        }

class LlamaConversationalAgent(LlamaTextAgent):
    """Agente conversacional con memoria y contexto"""
    
    def __init__(self, agent_name: str = "ConversationalAgent", **kwargs):
        super().__init__(agent_name, **kwargs)
        self.context_window = kwargs.get("context_window", 5)  # Últimas 5 interacciones
        self.personality = kwargs.get("personality", "profesional")
    
    def process(self, input_data: Union[str, Dict[str, Any]], **kwargs) -> str:
        """Procesa con contexto conversacional"""
        try:
            # Preparar mensaje actual
            if isinstance(input_data, str):
                current_message = input_data
            else:
                current_message = input_data.get("message", "")
            
            # Construir contexto conversacional
            context = self._build_conversational_context()
            
            # Crear prompt con personalidad y contexto
            prompt = self._create_conversational_prompt(current_message, context)
            
            # Procesar
            response = self.analyzer.get_llm().invoke(prompt)
            
            # Guardar en historial
            self.conversation_history.append({
                "user": current_message,
                "assistant": response,
                "timestamp": kwargs.get("timestamp")
            })
            
            # Mantener solo las últimas interacciones
            if len(self.conversation_history) > self.context_window:
                self.conversation_history = self.conversation_history[-self.context_window:]
            
            return response
            
        except Exception as e:
            logger.error(f"Error en conversación de {self.agent_name}: {e}")
            return f"Disculpa, ocurrió un error: {e}"
    
    def _build_conversational_context(self) -> str:
        """Construye el contexto conversacional"""
        if not self.conversation_history:
            return ""
        
        context_parts = ["Historial de conversación reciente:"]
        for entry in self.conversation_history[-3:]:  # Últimas 3 interacciones
            context_parts.append(f"Usuario: {entry['user']}")
            context_parts.append(f"Asistente: {entry['assistant']}")
        
        return "\n".join(context_parts)
    
    def _create_conversational_prompt(self, message: str, context: str) -> str:
        """Crea prompt conversacional personalizado"""
        personality_prompts = {
            "profesional": "Mantén un tono profesional y objetivo.",
            "amigable": "Sé amigable y cercano en tus respuestas.",
            "técnico": "Proporciona respuestas técnicas y detalladas.",
            "casual": "Usa un lenguaje casual y relajado."
        }
        
        personality_instruction = personality_prompts.get(self.personality, "")
        
        prompt = f"""
        {self.prompts.SYSTEM_PROMPT}
        
        Estilo de comunicación: {personality_instruction}
        
        {context}
        
        Usuario: {message}
        
        Responde de manera coherente con el contexto de la conversación:
        """
        
        return prompt

class AgentFactory:
    """Factory para crear diferentes tipos de agentes"""
    
    AGENT_TYPES = {
        "text": LlamaTextAgent,
        "analysis": LlamaAnalysisAgent,
        "conversational": LlamaConversationalAgent
    }
    
    @classmethod
    def create_agent(
        self, 
        agent_type: str, 
        agent_name: str = None,
        config_env: str = "development",
        **kwargs
    ) -> BaseAgent:
        """Crea un agente del tipo especificado"""
        
        if agent_type not in self.AGENT_TYPES:
            raise ValueError(f"Tipo de agente no válido: {agent_type}. Disponibles: {list(self.AGENT_TYPES.keys())}")
        
        agent_class = self.AGENT_TYPES[agent_type]
        
        if not agent_name:
            agent_name = f"{agent_type.capitalize()}Agent"
        
        return agent_class(
            agent_name=agent_name,
            config_env=config_env,
            **kwargs
        )
    
    @classmethod
    def list_agent_types(cls) -> List[str]:
        """Lista los tipos de agentes disponibles"""
        return list(cls.AGENT_TYPES.keys())

# Funciones helper para uso rápido
def create_text_agent(name: str = "TextAgent", config: str = "development") -> LlamaTextAgent:
    """Crea un agente de texto rápidamente"""
    return AgentFactory.create_agent("text", name, config)

def create_analysis_agent(name: str = "AnalysisAgent", config: str = "development") -> LlamaAnalysisAgent:
    """Crea un agente de análisis rápidamente"""
    return AgentFactory.create_agent("analysis", name, config)

def create_conversational_agent(name: str = "ConversationalAgent", config: str = "development") -> LlamaConversationalAgent:
    """Crea un agente conversacional rápidamente"""
    return AgentFactory.create_agent("conversational", name, config)

# Función para testing
def test_all_agents(config_env: str = "development") -> Dict[str, bool]:
    """Prueba todos los tipos de agentes"""
    results = {}
    
    for agent_type in AgentFactory.list_agent_types():
        try:
            agent = AgentFactory.create_agent(agent_type, config_env=config_env)
            results[agent_type] = agent.test_connection()
            logger.info(f"Agente {agent_type}: {'OK' if results[agent_type] else 'Error'}")
        except Exception as e:
            results[agent_type] = False
            logger.error(f"Error creando agente {agent_type}: {e}")
    
    return results