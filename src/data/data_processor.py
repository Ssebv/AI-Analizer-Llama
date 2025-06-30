import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class NewsDataProcessor:
    """Procesador especializado para datos de noticias desde Excel"""
    
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.df = None
        self.processed_docs = []
        
    def load_data(self) -> pd.DataFrame:
        """Carga datos desde Excel con manejo de errores"""
        try:
            self.df = pd.read_excel(self.excel_path)
            print(f"✅ Datos cargados: {len(self.df)} noticias")
            return self.df
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None
    
    def clean_and_preprocess(self) -> pd.DataFrame:
        """Limpia y preprocesa los datos de noticias"""
        if self.df is None:
            self.load_data()
        
        # Limpieza básica
        self.df = self.df.dropna(subset=['titulo', 'cuerpo'])
        
        # Convertir fecha a datetime
        self.df['fecha'] = pd.to_datetime(self.df['fecha'], errors='coerce')
        
        # Combinar texto completo para análisis
        self.df['texto_completo'] = (
            self.df['titulo'].astype(str) + ' ' + 
            self.df['bajada'].fillna('').astype(str) + ' ' + 
            self.df['cuerpo'].astype(str)
        )
        
        # Extraer información temporal
        self.df['año'] = self.df['fecha'].dt.year
        self.df['mes'] = self.df['fecha'].dt.month
        self.df['dia_semana'] = self.df['fecha'].dt.day_name()
        
        # Limpiar texto
        self.df['texto_limpio'] = self.df['texto_completo'].apply(self._clean_text)
        
        print(f"✅ Datos procesados: {len(self.df)} noticias válidas")
        return self.df
    
    def _clean_text(self, text: str) -> str:
        """Limpia el texto de caracteres especiales y espacios extra"""
        if pd.isna(text):
            return ""
        
        # Remover caracteres especiales y espacios extra
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-]', '', text)
        return text.strip()
    
    def create_documents_for_vectorstore(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Crea documentos para el vector store con metadatos enriquecidos"""
        
        if self.df is None:
            self.clean_and_preprocess()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        documents = []
        
        for idx, row in self.df.iterrows():
            # Texto principal
            main_text = row['texto_limpio']
            
            # Metadatos enriquecidos
            metadata = {
                'id': idx,
                'titulo': row['titulo'],
                'medio': row['nombre_medio'],
                'tipo_medio': row['tipo_medio'],
                'fecha': row['fecha'].strftime('%Y-%m-%d') if pd.notna(row['fecha']) else None,
                'seccion': row['seccion'],
                'año': row['año'],
                'mes': row['mes'],
                'dia_semana': row['dia_semana'],
                'longitud_texto': len(main_text),
                'source': 'noticias_dataset'
            }
            
            # Dividir texto en chunks si es muy largo
            if len(main_text) > chunk_size:
                chunks = text_splitter.split_text(main_text)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_id'] = i
                    chunk_metadata['total_chunks'] = len(chunks)
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    ))
            else:
                documents.append(Document(
                    page_content=main_text,
                    metadata=metadata
                ))
        
        self.processed_docs = documents
        print(f"✅ Creados {len(documents)} documentos para vector store")
        return documents
    
    def get_statistical_summary(self) -> Dict:
        """Genera resumen estadístico de los datos"""
        if self.df is None:
            self.clean_and_preprocess()
        
        summary = {
            'total_noticias': len(self.df),
            'rango_fechas': {
                'inicio': self.df['fecha'].min().strftime('%Y-%m-%d'),
                'fin': self.df['fecha'].max().strftime('%Y-%m-%d')
            },
            'medios_unicos': self.df['nombre_medio'].nunique(),
            'secciones_unicas': self.df['seccion'].nunique(),
            'distribucion_por_medio': self.df['nombre_medio'].value_counts().to_dict(),
            'distribucion_por_seccion': self.df['seccion'].value_counts().to_dict(),
            'distribucion_temporal': {
                'por_año': self.df['año'].value_counts().sort_index().to_dict(),
                'por_mes': self.df['mes'].value_counts().sort_index().to_dict()
            },
            'estadisticas_texto': {
                'longitud_promedio': self.df['texto_completo'].str.len().mean(),
                'longitud_mediana': self.df['texto_completo'].str.len().median(),
                'longitud_max': self.df['texto_completo'].str.len().max(),
                'longitud_min': self.df['texto_completo'].str.len().min()
            }
        }
        
        return summary
    
    def filter_data(self, 
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   medios: Optional[List[str]] = None,
                   secciones: Optional[List[str]] = None) -> pd.DataFrame:
        """Filtra los datos según criterios específicos"""
        
        filtered_df = self.df.copy()
        
        if start_date:
            filtered_df = filtered_df[filtered_df['fecha'] >= pd.to_datetime(start_date)]
        
        if end_date:
            filtered_df = filtered_df[filtered_df['fecha'] <= pd.to_datetime(end_date)]
        
        if medios:
            filtered_df = filtered_df[filtered_df['nombre_medio'].isin(medios)]
        
        if secciones:
            filtered_df = filtered_df[filtered_df['seccion'].isin(secciones)]
        
        return filtered_df
    
    def export_processed_data(self, output_path: str):
        """Exporta los datos procesados"""
        if self.df is not None:
            self.df.to_excel(output_path, index=False)
            print(f"✅ Datos exportados a: {output_path}")


# Ejemplo de uso:
if __name__ == "__main__":
    # Inicializar procesador
    processor = NewsDataProcessor("data/noticias_test_ingeniero_IA.xlsx")
    
    # Procesar datos
    df = processor.clean_and_preprocess()
    
    # Crear documentos para vector store
    documents = processor.create_documents_for_vectorstore()
    
    # Obtener resumen estadístico
    summary = processor.get_statistical_summary()
    print("\n📊 Resumen del Dataset:")
    print(f"Total noticias: {summary['total_noticias']}")
    print(f"Rango de fechas: {summary['rango_fechas']['inicio']} - {summary['rango_fechas']['fin']}")
    print(f"Medios únicos: {summary['medios_unicos']}")
    print(f"Secciones únicas: {summary['secciones_unicas']}")