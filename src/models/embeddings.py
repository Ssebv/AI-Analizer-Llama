"""
Configuración de embeddings para el sistema de análisis de noticias
"""
import os
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class NewsEmbeddingsManager:
    """Maneja la creación y gestión de embeddings para noticias"""
    
    def __init__(
        self, 
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "auto"
    ):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo de embeddings"""
        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            logger.info(f"Modelo de embeddings {self.model_name} cargado correctamente")
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            raise
    
    def create_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Crea embeddings para una lista de textos
        
        Args:
            texts: Lista de textos a procesar
            batch_size: Tamaño del batch para procesamiento
            show_progress: Mostrar barra de progreso
        
        Returns:
            np.ndarray: Array de embeddings
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            logger.info(f"Creados {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error creando embeddings: {e}")
            raise
    
    def create_news_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Crea embeddings específicos para el dataset de noticias
        
        Args:
            df: DataFrame con las noticias
        
        Returns:
            np.ndarray: Embeddings de las noticias
        """
        # Combinar campos relevantes para crear texto completo
        combined_texts = self._combine_news_fields(df)
        return self.create_embeddings(combined_texts)
    
    def _combine_news_fields(self, df: pd.DataFrame) -> List[str]:
        """
        Combina los campos de las noticias para crear texto completo
        
        Args:
            df: DataFrame con las noticias
        
        Returns:
            List[str]: Lista de textos combinados
        """
        combined_texts = []
        
        for _, row in df.iterrows():
            # Combinar título, bajada y cuerpo
            text_parts = []
            
            if pd.notna(row.get('titulo')):
                text_parts.append(f"Título: {row['titulo']}")
            
            if pd.notna(row.get('bajada')):
                text_parts.append(f"Bajada: {row['bajada']}")
            
            if pd.notna(row.get('cuerpo')):
                # Limitar el cuerpo para evitar textos muy largos
                cuerpo = str(row['cuerpo'])[:2000]  # Primeros 2000 caracteres
                text_parts.append(f"Contenido: {cuerpo}")
            
            # Agregar metadatos relevantes
            if pd.notna(row.get('nombre_medio')):
                text_parts.append(f"Medio: {row['nombre_medio']}")
            
            if pd.notna(row.get('seccion')):
                text_parts.append(f"Sección: {row['seccion']}")
            
            combined_text = " | ".join(text_parts)
            combined_texts.append(combined_text)
        
        return combined_texts
    
    def similarity_search(
        self, 
        query: str, 
        embeddings: np.ndarray, 
        top_k: int = 5
    ) -> List[int]:
        """
        Busca los documentos más similares a una consulta
        
        Args:
            query: Consulta de búsqueda
            embeddings: Embeddings de los documentos
            top_k: Número de resultados a retornar
        
        Returns:
            List[int]: Índices de los documentos más similares
        """
        query_embedding = self.model.encode([query])
        
        # Calcular similitudes coseno
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        
        # Obtener top_k índices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return top_indices.tolist()

class ChileanNewsEmbeddings(NewsEmbeddingsManager):
    """Clase especializada para embeddings de noticias chilenas"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chile_specific_terms = [
            "comuna", "región", "municipalidad", "alcalde", "concejo municipal",
            "Los Andes", "Calle Larga", "Til Til", "Colina",
            "Región Metropolitana", "Región de Valparaíso"
        ]
    
    def preprocess_chilean_text(self, text: str) -> str:
        """
        Preprocesa texto específico para contexto chileno
        
        Args:
            text: Texto a preprocesar
        
        Returns:
            str: Texto preprocesado
        """
        # Normalizar términos específicos chilenos
        normalized_text = text
        
        # Aquí se pueden agregar más normalizaciones específicas
        # Por ejemplo, normalizar nombres de comunas, etc.
        
        return normalized_text
    
    def create_contextual_embeddings(
        self, 
        df: pd.DataFrame,
        include_geographic_context: bool = True
    ) -> np.ndarray:
        """
        Crea embeddings con contexto geográfico chileno
        
        Args:
            df: DataFrame con noticias
            include_geographic_context: Incluir contexto geográfico
        
        Returns:
            np.ndarray: Embeddings contextualizados
        """
        texts = self._combine_news_fields(df)
        
        if include_geographic_context:
            # Agregar contexto geográfico a cada texto
            contextualized_texts = []
            for i, text in enumerate(texts):
                if i < len(df):
                    row = df.iloc[i]
                    geographic_context = self._get_geographic_context(row)
                    contextualized_text = f"{geographic_context} {text}"
                    contextualized_texts.append(contextualized_text)
                else:
                    contextualized_texts.append(text)
            texts = contextualized_texts
        
        return self.create_embeddings(texts)
    
    def _get_geographic_context(self, row: pd.Series) -> str:
        """
        Obtiene contexto geográfico para una noticia
        
        Args:
            row: Fila del DataFrame con datos de la noticia
        
        Returns:
            str: Contexto geográfico
        """
        context_parts = []
        
        # Determinar región basada en heurísticas o datos disponibles
        # Esto se puede mejorar con más información del dataset
        
        context_parts.append("Contexto: Noticias locales de comunas chilenas.")
        
        return " ".join(context_parts)

def setup_embeddings_for_news(
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    use_chilean_context: bool = True
) -> NewsEmbeddingsManager:
    """
    Configura el sistema de embeddings para análisis de noticias
    
    Args:
        model_name: Nombre del modelo de embeddings
        use_chilean_context: Usar contexto específico chileno
    
    Returns:
        NewsEmbeddingsManager: Instancia configurada
    """
    if use_chilean_context:
        return ChileanNewsEmbeddings(model_name=model_name)
    else:
        return NewsEmbeddingsManager(model_name=model_name)

# Modelos recomendados para diferentes casos de uso
EMBEDDING_MODELS = {
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
    "spanish": "sentence-transformers/paraphrase-spanish-distilroberta",
    "fast": "all-MiniLM-L6-v2",
    "quality": "all-mpnet-base-v2"
}

def get_embedding_model(use_case: str = "multilingual") -> str:
    """Obtiene modelo recomendado para caso de uso específico"""
    return EMBEDDING_MODELS.get(use_case, EMBEDDING_MODELS["multilingual"])