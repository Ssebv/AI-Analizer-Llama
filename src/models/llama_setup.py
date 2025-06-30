"""
Configuración y setup de Llama para el análisis de noticias
"""
import os
import logging
import requests
from typing import Optional, Dict, Any
from langchain_ollama import OllamaLLM
# Alias para compatibilidad con código existente
Ollama = OllamaLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult, Generation

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caché global para no instanciar repetidamente el mismo modelo
_ANALYZER_CACHE: dict[str, "LlamaNewsAnalyzer"] = {}

class LlamaNewsAnalyzer:
    """Clase principal para configurar Llama para análisis de noticias chilenas"""
    
    def __init__(
        self, 
        model_name: str = "llama3.1:8b",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.llm = None
        self._setup_llm()
        self._connection_tested = False
    
    def _setup_llm(self):
        """Configura el modelo Llama con parámetros optimizados"""
        try:
            self.llm = OllamaLLM(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                num_predict=self.max_tokens,
                top_p=0.9,
                repeat_penalty=1.1,
                verbose=False
            )
            logger.info(f"Llama modelo {self.model_name} configurado correctamente")
        except Exception as e:
            logger.error(f"Error configurando Llama: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Ping rápido al servidor Ollama (sin generar tokens)"""
        if self._connection_tested:
            return True
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            ok = resp.status_code == 200
            if ok:
                logger.info("Conexión con Ollama exitosa")
                self._connection_tested = True
            else:
                logger.error(f"Ollama respondió con código {resp.status_code}")
            return ok
        except requests.RequestException as e:
            logger.error(f"Error en conexión con Ollama: {e}")
            return False
    
    def get_llm(self):
        """Retorna la instancia del LLM configurado"""
        return self.llm

class ChileanNewsPrompts:
    """Prompts especializados para análisis de noticias chilenas"""
    
    SYSTEM_PROMPT = """
    Eres un experto analista de medios de comunicación chilenos, especializado en noticias locales 
    de las comunas de Los Andes, Calle Larga, Til Til y Colina en Chile.
    
    Características importantes:
    - Los Andes: Comuna de la Región de Valparaíso
    - Calle Larga: Comuna de la Región de Valparaíso  
    - Til Til: Comuna de la Región Metropolitana
    - Colina: Comuna de la Región Metropolitana
    
    Tu rol es proporcionar análisis profesionales, objetivos y contextualmente relevantes
    sobre las noticias de estas comunas.
    """
    
    ANALYSIS_PROMPT = """
    {system_prompt}
    
    Contexto del dataset: {context}
    
    Pregunta del usuario: {question}
    
    Instrucciones:
    1. Analiza la información disponible
    2. Proporciona insights relevantes y accionables
    3. Usa terminología periodística apropiada
    4. Considera el contexto geográfico y social chileno
    5. Sé específico en tus hallazgos
    
    Respuesta:
    """
    
    SUMMARY_PROMPT = """
    {system_prompt}
    
    Noticias a resumir: {news_content}
    
    Crea un resumen ejecutivo que incluya:
    1. Temas principales identificados
    2. Eventos más relevantes
    3. Tendencias observadas
    4. Insights clave para stakeholders
    
    Resumen:
    """
    
    TEMPORAL_ANALYSIS_PROMPT = """
    {system_prompt}
    
    Datos temporales: {temporal_data}
    
    Realiza un análisis temporal que incluya:
    1. Patrones estacionales
    2. Picos de actividad noticiosa
    3. Tendencias a lo largo del tiempo
    4. Correlaciones con eventos específicos
    
    Análisis temporal:
    """

def setup_llama_for_news_analysis(
    model_name: str | None = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> LlamaNewsAnalyzer:
    """
    Función helper para configurar Llama específicamente para análisis de noticias
    
    Args:
        model_name: Nombre del modelo Llama a utilizar
        custom_config: Configuración personalizada
    
    Returns:
        LlamaNewsAnalyzer: Instancia configurada
    """
    global _ANALYZER_CACHE
    # Si ya existe un analizador para este modelo y no hay config distinta, devuélvelo
    if model_name in _ANALYZER_CACHE and not custom_config:
        return _ANALYZER_CACHE[model_name]

    config = {
        "temperature": 0.1,
        "max_tokens": 2048,
        "base_url": "http://localhost:11434"
    }
    
    if custom_config:
        config.update(custom_config)

    # Evitar pasar 'model_name' dos veces
    if custom_config and "model_name" in custom_config:
        if model_name is None:
            model_name = config.pop("model_name")  # toma el del diccionario y lo quita
        else:
            config.pop("model_name")  # ya viene explícito, se quita del dict

    if model_name is None:
        model_name = "llama3.1:8b"
    
    analyzer = LlamaNewsAnalyzer(
        model_name=model_name,
        **config
    )
    
    # Verificar conexión
    if not analyzer.test_connection():
        raise ConnectionError("No se pudo conectar con Ollama. Asegúrate de que esté ejecutándose.")

    _ANALYZER_CACHE[model_name] = analyzer
    
    return analyzer

# Configuraciones predefinidas para diferentes casos de uso
CONFIGS = {
    "development": {
        "model_name": "llama3.1:8b",
        "temperature": 0.2,
        "max_tokens": 512
    },
    "production": {
        "model_name": "llama3.1:70b",
        "temperature": 0.1,
        "max_tokens": 2048
    },
    "fast": {
        "model_name": "llama3.1:8b",
        "temperature": 0.0,
        "max_tokens": 512
    }
}

def get_config(env: str = "development") -> Dict[str, Any]:
    """Obtiene configuración predefinida por ambiente"""
    return CONFIGS.get(env, CONFIGS["development"])