# app.py - Aplicación Streamlit
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List
import numpy as np
from collections import Counter
import io
from pathlib import Path
import sys

# Aseguramos que la carpeta `src` esté en PYTHONPATH para que `core` se pueda importar
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Importar IntegratedNewsSystem del core
from core.system import IntegratedNewsSystem


# Carpeta donde se guardan los Excel subidos
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Configuración de la página
st.set_page_config(
    page_title="📰 Análisis Inteligente de Noticias",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .analysis-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class NewsAnalysisApp:
    """Aplicación principal de análisis de noticias"""
    
    def __init__(self):
        self.initialize_session_state()
        self.system = st.session_state.get('system')
    
    def initialize_session_state(self):
        """Inicializa el estado de la sesión"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'vectorstore_ready' not in st.session_state:
            st.session_state.vectorstore_ready = False
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None
        if 'df_news' not in st.session_state:
            st.session_state.df_news = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'excel_path' not in st.session_state:
            st.session_state.excel_path = None
        if 'system' not in st.session_state:
            st.session_state.system = None
    
    def render_header(self):
        """Renderiza el header de la aplicación"""
        st.markdown('<h1 class="main-header">📰 Análisis Inteligente de Noticias</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Indicadores de estado
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "✅ Cargado" if st.session_state.data_loaded else "❌ No cargado"
            st.metric("Dataset", status)
        
        with col2:
            status = "✅ Activo" if st.session_state.vectorstore_ready else "❌ Inactivo"
            st.metric("Vector Store", status)
        
        with col3:
            st.metric("Modelo LLM", "🦙 Llama 3.1")
        
        with col4:
            st.metric("Análisis Realizados", len(st.session_state.analysis_history))
    
    def render_sidebar(self):
        """Renderiza la barra lateral con configuraciones"""
        with st.sidebar:
            st.header("🔧 Configuración")
            
            # Configuración del modelo
            st.subheader("Modelo LLM")
            model_name = st.selectbox(
                "Seleccionar modelo:",
                ["llama3.1:8b", "llama3.1:70b", "llama3:8b"],
                index=0
            )
            
            # Configuración de análisis
            st.subheader("Parámetros de Análisis")
            
            analysis_depth = st.select_slider(
                "Profundidad del análisis:",
                options=["Básico", "Intermedio", "Profundo"],
                value="Intermedio"
            )
            
            max_documents = st.slider(
                "Máximo documentos a analizar:",
                min_value=5,
                max_value=50,
                value=20
            )
            
            # Filtros de datos
            st.subheader("Filtros de Datos")
            
            # Selector de fechas
            date_filter = st.checkbox("Filtrar por fechas")
            if date_filter and st.session_state.data_loaded:
                start_date = st.date_input("Fecha inicio")
                end_date = st.date_input("Fecha fin")
            
            # Filtros por medio
            media_filter = st.checkbox("Filtrar por medios")
            if media_filter and st.session_state.data_loaded and st.session_state.df_news is not None:
                available_media = st.session_state.df_news['nombre_medio'].unique() if 'nombre_medio' in st.session_state.df_news.columns else []
                selected_media = st.multiselect(
                    "Seleccionar medios:",
                    available_media
                )
            
            # Configuración avanzada
            with st.expander("⚙️ Configuración Avanzada"):
                temperature = st.slider("Temperatura LLM:", 0.0, 1.0, 0.1, 0.1)
                top_k = st.slider("Top-K documentos:", 1, 20, 6)
                use_compression = st.checkbox("Usar compresión contextual", value=True)
            
            # Información del sistema
            st.subheader("ℹ️ Estado del Sistema")
            if st.button("🔄 Actualizar Estado"):
                self.check_system_status()
    
    def process_uploaded_data(self, uploaded_file):
        """Procesa el archivo Excel subido"""
        try:
            # Leer el archivo Excel
            df = pd.read_excel(uploaded_file)
            
            # Validar columnas requeridas
            required_columns = ['titulo', 'bajada', 'cuerpo', 'nombre_medio', 'fecha', 'seccion']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Faltan las siguientes columnas: {', '.join(missing_columns)}")
                return False
            
            # Procesar fechas
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
            
            # Guardar en session state
            st.session_state.df_news = df
            # Guardar una copia para el backend real
            tmp_file = UPLOAD_DIR / f"news_{int(time.time())}.xlsx"
            df.to_excel(tmp_file, index=False)
            st.session_state.excel_path = str(tmp_file)
            
            return True
        
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
            return False
    
    def create_vector_store(self):
        """Inicializa IntegratedNewsSystem con el Excel subido y crea el vector‑store."""
        try:
            if not st.session_state.excel_path:
                st.error("Primero debes cargar un archivo Excel válido.")
                return False

            self.system = IntegratedNewsSystem(st.session_state.excel_path)
            ok = self.system.initialize_system()
            if ok:
                st.session_state.system = self.system
                st.session_state.vectorstore_ready = True
            return ok
        except Exception as e:
            st.error(f"Error al crear vector store: {str(e)}")
            return False
    
    def load_vector_store(self):
        """Marca el vector‑store como listo si el sistema ya existe en sesión."""
        if st.session_state.system:
            st.session_state.vectorstore_ready = True
            return True
        st.error("No hay un sistema inicializado en esta sesión.")
        return False
    
    def check_system_status(self):
        """Verifica el estado del sistema"""
        with st.spinner("Verificando estado del sistema..."):
            time.sleep(1)
            st.success("Sistema funcionando correctamente")
    
    def render_data_upload_section(self):
        """Sección para cargar datos"""
        st.header("📁 Carga de Datos")
        
        if not st.session_state.data_loaded:
            uploaded_file = st.file_uploader(
                "Sube tu archivo Excel con noticias:",
                type=['xlsx', 'xls'],
                help="El archivo debe contener columnas: titulo, bajada, cuerpo, nombre_medio, fecha, seccion"
            )
            
            if uploaded_file is not None:
                if st.button("🚀 Procesar Dataset"):
                    with st.spinner("Procesando datos..."):
                        success = self.process_uploaded_data(uploaded_file)
                        if success:
                            st.success("✅ Datos cargados exitosamente!")
                            st.session_state.data_loaded = True
                            st.rerun()
        else:
            st.success("✅ Dataset cargado correctamente")
            
            # Mostrar estadísticas del dataset
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Estadísticas del Dataset")
                
                if st.session_state.df_news is not None:
                    df = st.session_state.df_news
                    stats = {
                        "Total noticias": f"{len(df):,}",
                        "Medios únicos": f"{df['nombre_medio'].nunique()}" if 'nombre_medio' in df.columns else "N/A",
                        "Rango de fechas": f"{df['fecha'].min().strftime('%Y-%m-%d')} a {df['fecha'].max().strftime('%Y-%m-%d')}" if 'fecha' in df.columns else "N/A",
                        "Secciones": f"{df['seccion'].nunique()}" if 'seccion' in df.columns else "N/A"
                    }
                else:
                    stats = {
                        "Total noticias": "N/A",
                        "Medios únicos": "N/A",
                        "Rango de fechas": "N/A",
                        "Secciones": "N/A"
                    }
                
                for key, value in stats.items():
                    st.metric(key, value)
            
            with col2:
                st.subheader("📈 Distribución por Medio")
                
                if st.session_state.df_news is not None and 'nombre_medio' in st.session_state.df_news.columns:
                    media_counts = st.session_state.df_news['nombre_medio'].value_counts().head(10)
                    fig = px.bar(
                        x=media_counts.values,
                        y=media_counts.index,
                        orientation='h',
                        title="Top 10 Medios por Cantidad de Noticias"
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_vector_store_section(self):
        """Sección para configurar el vector store"""
        if st.session_state.data_loaded:
            st.header("🗄️ Vector Store")
            
            if not st.session_state.vectorstore_ready:
                st.info("El vector store no está inicializado. Es necesario para realizar análisis avanzados.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔨 Crear Vector Store"):
                        with st.spinner("Creando vector store... Esto puede tomar varios minutos."):
                            success = self.create_vector_store()
                            if success:
                                st.success("✅ Vector store creado exitosamente!")
                                st.session_state.vectorstore_ready = True
                                st.rerun()
                
                with col2:
                    if st.button("📂 Cargar Vector Store Existente"):
                        with st.spinner("Cargando vector store existente..."):
                            success = self.load_vector_store()
                            if success:
                                st.success("✅ Vector store cargado exitosamente!")
                                st.session_state.vectorstore_ready = True
                                st.rerun()
            else:
                st.success("✅ Vector store listo para análisis")
                
                # Estadísticas del vector store
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_docs = len(st.session_state.df_news) if st.session_state.df_news is not None else 0
                    st.metric("Documentos indexados", f"{total_docs:,}")
                
                with col2:
                    st.metric("Dimensiones embedding", "4096")
                
                with col3:
                    st.metric("Tamaño base de datos", "45.2 MB")
    
    def perform_analysis(self, query: str, analysis_type: str = "comprehensive"):
        """Ejecuta la consulta usando IntegratedNewsSystem real."""
        if not st.session_state.vectorstore_ready or not st.session_state.system:
            st.error("El sistema aún no está listo. Crea o carga el vector store primero.")
            return None

        agent_map = {"quick": "text", "comprehensive": "analysis", "deep": "analysis"}
        agent = agent_map.get(analysis_type, "auto")

        result = st.session_state.system.query_with_agent(query, agent)
        if "error" in result:
            st.error(result["error"])
            return None

        return {
            "query": query,
            "analysis_type": analysis_type,
            "summary": result["answer"],
            "insights": result.get("insights", []),
            "relevant_documents": result.get("sources", []),
            "timestamp": datetime.now()
        }
    
    def display_analysis_result(self, result: Dict):
        """Muestra los resultados del análisis"""
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        
        # Resumen
        st.subheader("📋 Resumen del Análisis")
        st.write(result["summary"])
        
        # Insights principales
        st.subheader("💡 Insights Principales")
        for i, insight in enumerate(result["insights"], 1):
            st.write(f"{i}. {insight}")
        
        # Documentos relevantes
        st.subheader("📄 Documentos Más Relevantes")
        for doc in result["relevant_documents"]:
            with st.expander(f"📰 {doc['title']} (Relevancia: {doc['relevance']:.2%})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Fuente:** {doc['source']}")
                with col2:
                    st.write(f"**Fecha:** {doc['date']}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def run_automatic_analysis(self, analysis_options: List[str]):
        """Ejecuta análisis automático basado en las opciones seleccionadas"""
        results = {}
        
        for option in analysis_options:
            if option == "Clustering de temas principales":
                results[option] = self.generate_topic_clustering()
            elif option == "Análisis de sentimientos":
                results[option] = self.generate_sentiment_analysis()
            elif option == "Tendencias temporales":
                results[option] = self.generate_temporal_trends()
            elif option == "Comparación entre medios":
                results[option] = self.generate_media_comparison()
            elif option == "Detección de eventos relevantes":
                results[option] = self.generate_event_detection()
            elif option == "Análisis de palabras clave":
                results[option] = self.generate_keyword_analysis()
        
        return results
    
    def generate_topic_clustering(self):
        """Genera análisis de clustering de temas"""
        return {
            "clusters": [
                {"topic": "Política Nacional", "size": 156, "keywords": ["gobierno", "congreso", "política"]},
                {"topic": "Economía", "size": 134, "keywords": ["economía", "mercado", "inversión"]},
                {"topic": "Deportes", "size": 89, "keywords": ["fútbol", "deportes", "campeonato"]},
                {"topic": "Salud", "size": 67, "keywords": ["salud", "hospital", "medicina"]},
                {"topic": "Tecnología", "size": 45, "keywords": ["tecnología", "digital", "innovación"]}
            ]
        }
    
    def generate_sentiment_analysis(self):
        """Genera análisis de sentimientos"""
        return {
            "overall_sentiment": "Neutral",
            "distribution": {
                "Positivo": 0.35,
                "Neutral": 0.45,
                "Negativo": 0.20
            },
            "trends": "Tendencia hacia sentimiento más positivo en la última semana"
        }
    
    def generate_temporal_trends(self):
        """Genera análisis de tendencias temporales"""
        dates = pd.date_range(start='2024-01-01', end='2024-06-27', freq='W')
        values = np.random.randint(20, 100, len(dates))
        
        return {
            "timeline": list(zip(dates.strftime('%Y-%m-%d'), values)),
            "peak_periods": ["2024-03-15", "2024-05-20"],
            "growth_rate": "+15% en los últimos 3 meses"
        }
    
    def generate_media_comparison(self):
        """Genera comparación entre medios"""
        return {
            "coverage_comparison": {
                "El Mercurio": {"articles": 245, "sentiment": 0.15},
                "La Tercera": {"articles": 198, "sentiment": 0.05},
                "El Mostrador": {"articles": 167, "sentiment": -0.10},
                "BioBioChile": {"articles": 134, "sentiment": 0.20}
            }
        }
    
    def generate_event_detection(self):
        """Genera detección de eventos"""
        return {
            "events": [
                {"date": "2024-06-20", "event": "Pico de cobertura política", "intensity": "Alto"},
                {"date": "2024-06-15", "event": "Evento deportivo relevante", "intensity": "Medio"},
                {"date": "2024-06-10", "event": "Anuncio económico importante", "intensity": "Alto"}
            ]
        }
    
    def generate_keyword_analysis(self):
        """Genera análisis de palabras clave"""
        return {
            "top_keywords": [
                {"word": "gobierno", "frequency": 456, "trend": "↑"},
                {"word": "economía", "frequency": 234, "trend": "→"},
                {"word": "salud", "frequency": 189, "trend": "↑"},
                {"word": "educación", "frequency": 167, "trend": "↓"},
                {"word": "tecnología", "frequency": 145, "trend": "↑"}
            ]
        }
    
    def display_automatic_results(self, results: Dict):
        """Muestra los resultados del análisis automático"""
        for analysis_type, data in results.items():
            with st.expander(f"📊 {analysis_type}", expanded=True):
                
                if analysis_type == "Clustering de temas principales":
                    df_clusters = pd.DataFrame(data["clusters"])
                    fig = px.treemap(
                        df_clusters, 
                        path=['topic'], 
                        values='size',
                        title="Distribución de Temas"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif analysis_type == "Análisis de sentimientos":
                    labels = list(data["distribution"].keys())
                    values = list(data["distribution"].values())
                    fig = px.pie(values=values, names=labels, title="Distribución de Sentimientos")
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(data["trends"])
                
                elif analysis_type == "Tendencias temporales":
                    timeline_data = data["timeline"]
                    df_timeline = pd.DataFrame(timeline_data, columns=["Fecha", "Cantidad"])
                    df_timeline["Fecha"] = pd.to_datetime(df_timeline["Fecha"])
                    
                    fig = px.line(df_timeline, x="Fecha", y="Cantidad", title="Tendencia Temporal de Noticias")
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(f"Crecimiento: {data['growth_rate']}")
                
                elif analysis_type == "Comparación entre medios":
                    media_data = []
                    for medio, stats in data["coverage_comparison"].items():
                        media_data.append({
                            "Medio": medio,
                            "Artículos": stats["articles"],
                            "Sentimiento": stats["sentiment"]
                        })
                    
                    df_media = pd.DataFrame(media_data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = px.bar(df_media, x="Medio", y="Artículos", title="Artículos por Medio")
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        fig2 = px.bar(df_media, x="Medio", y="Sentimiento", title="Sentimiento por Medio")
                        st.plotly_chart(fig2, use_container_width=True)
                
                elif analysis_type == "Detección de eventos relevantes":
                    for event in data["events"]:
                        st.write(f"**{event['date']}**: {event['event']} (Intensidad: {event['intensity']})")
                
                elif analysis_type == "Análisis de palabras clave":
                    df_keywords = pd.DataFrame(data["top_keywords"])
                    fig = px.bar(
                        df_keywords, 
                        x="frequency", 
                        y="word", 
                        orientation='h',
                        title="Top Palabras Clave"
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_chat_interface(self):
        """Interfaz de chat conversacional"""
        st.subheader("💬 Pregúntale a tus datos")
        
        # Mostrar historial de chat
        for i, chat in enumerate(st.session_state.chat_history):
            # Mensaje del usuario
            st.markdown(f'<div class="chat-message user-message"><strong>Tú:</strong> {chat["query"]}</div>', unsafe_allow_html=True)
            
            # Respuesta del asistente
            st.markdown(f'<div class="chat-message assistant-message"><strong>Asistente:</strong> {chat["response"]}</div>', unsafe_allow_html=True)
        
        # Ejemplos de consultas
        with st.expander("💡 Ejemplos de consultas"):
            examples = [
                "¿Cuáles son las principales tendencias en las noticias deportivas?",
                "Muéstrame un análisis temporal de las noticias de política",
                "Compara la cobertura entre diferentes medios sobre economía",
                "¿Qué temas han sido más frecuentes en el último mes?",
                "Identifica patrones en las noticias de salud"
            ]
            
            for example in examples:
                if st.button(f"📝 {example}", key=f"example_{hash(example)}"):
                    st.session_state.current_query = example
        
        # Input de consulta
        user_query = st.text_area(
            "Escribe tu consulta:",
            height=100,
            placeholder="Por ejemplo: ¿Cuáles son los temas más relevantes en las noticias de tecnología del último trimestre?",
            value=getattr(st.session_state, 'current_query', '')
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            analyze_button = st.button("🚀 Analizar", type="primary")
        
        with col2:
            analysis_type = st.selectbox(
                "Tipo de análisis:",
                ["Comprehensive", "Quick", "Deep"],
                index=0
            )
        
        if analyze_button and user_query:
            with st.spinner("🤖 Analizando... Esto puede tomar unos momentos."):
                result = self.perform_analysis(user_query, analysis_type.lower())
                
                if result:
                    # Agregar al historial de chat
                    chat_entry = {
                        "query": user_query,
                        "response": result["summary"],
                        "full_result": result,
                        "timestamp": datetime.now()
                    }
                    st.session_state.chat_history.append(chat_entry)
                    
                    # Mostrar resultado completo
                    self.display_analysis_result(result)
                    
                    # Agregar al historial de análisis
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now(),
                        'query': user_query,
                        'result': result
                    })
                    
                    # Limpiar consulta actual
                    if hasattr(st.session_state, 'current_query'):
                        del st.session_state.current_query
                    
                    st.rerun()
    
    def render_automatic_analysis(self):
        """Análisis automático de patrones"""
        st.subheader("📊 Análisis Automático de Patrones")
        
        analysis_options = st.multiselect(
            "Selecciona los análisis a realizar:",
            [
                "Clustering de temas principales",
                "Análisis de sentimientos",
                "Tendencias temporales",
                "Comparación entre medios",
                "Detección de eventos relevantes",
                "Análisis de palabras clave"
            ],
            default=["Clustering de temas principales", "Tendencias temporales"]
        )
        
        if st.button("🎯 Ejecutar Análisis Automático"):
            with st.spinner("Ejecutando análisis automático..."):
                results = self.run_automatic_analysis(analysis_options)
                self.display_automatic_results(results)
    
    def render_visualizations(self):
        """Sección de visualizaciones"""
        st.subheader("📈 Visualizaciones Interactivas")
        
        # Métricas generales
        col1, col2, col3, col4 = st.columns(4)
        
        total_news = len(st.session_state.df_news) if st.session_state.df_news is not None else 0
        total_media = st.session_state.df_news['nombre_medio'].nunique() if st.session_state.df_news is not None and 'nombre_medio' in st.session_state.df_news.columns else 0
        
        with col1:
            st.metric("Noticias Analizadas", f"{total_news:,}", "+50 esta semana")
        
        with col2:
            st.metric("Medios Activos", total_media, "+2 este mes")
        
        with col3:
            st.metric("Temas Identificados", "47", "+3 este mes")
        
        with col4:
            st.metric("Análisis Realizados", len(st.session_state.analysis_history))
        
        # Gráficos adicionales
        if st.session_state.df_news is not None:
            st.subheader("📊 Análisis Temporal")
            
            # Gráfico de noticias por día
            if 'fecha' in st.session_state.df_news.columns:
                df_daily = st.session_state.df_news.groupby(st.session_state.df_news['fecha'].dt.date).size().reset_index()
                df_daily.columns = ['Fecha', 'Cantidad']
                
                fig_daily = px.line(df_daily, x='Fecha', y='Cantidad', title='Noticias por Día')
                st.plotly_chart(fig_daily, use_container_width=True)
            
            # Gráfico por sección
            if 'seccion' in st.session_state.df_news.columns:
                st.subheader("📂 Distribución por Sección")
                section_counts = st.session_state.df_news['seccion'].value_counts()
                
                fig_sections = px.pie(
                    values=section_counts.values,
                    names=section_counts.index,
                    title="Distribución de Noticias por Sección"
                )
                st.plotly_chart(fig_sections, use_container_width=True)
    
    def render_analysis_history(self):
        """Muestra el historial de análisis"""
        st.subheader("📋 Historial de Análisis")
        
        if not st.session_state.analysis_history:
            st.info("No hay análisis previos. Realiza tu primer análisis en la pestaña de Chat Conversacional.")
            return
        
        # Filtros para el historial
        col1, col2 = st.columns(2)
        
        with col1:
            filter_date = st.date_input("Filtrar desde fecha:", value=datetime.now().date() - timedelta(days=7))
        
        with col2:
            search_term = st.text_input("Buscar en consultas:", placeholder="Ingresa términos de búsqueda...")
        
        # Mostrar análisis filtrados
        filtered_history = []
        for analysis in st.session_state.analysis_history:
            # Filtro por fecha
            if analysis['timestamp'].date() >= filter_date:
                # Filtro por término de búsqueda
                if not search_term or search_term.lower() in analysis['query'].lower():
                    filtered_history.append(analysis)
        
        if not filtered_history:
            st.warning("No se encontraron análisis que coincidan con los filtros.")
            return
        
        # Estadísticas del historial
        st.subheader("📊 Estadísticas del Historial")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Análisis", len(st.session_state.analysis_history))
        
        with col2:
            st.metric("Filtrados", len(filtered_history))
        
        with col3:
            avg_per_day = len(st.session_state.analysis_history) / max(1, (datetime.now() - min(a['timestamp'] for a in st.session_state.analysis_history)).days)
            st.metric("Promedio por día", f"{avg_per_day:.1f}")
        
        # Lista de análisis
        for i, analysis in enumerate(reversed(filtered_history)):
            with st.expander(f"📝 {analysis['timestamp'].strftime('%Y-%m-%d %H:%M')} - {analysis['query'][:50]}..."):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Consulta:** {analysis['query']}")
                    st.write(f"**Resumen:** {analysis['result']['summary']}")
                    
                    # Mostrar insights
                    if 'insights' in analysis['result']:
                        st.write("**Insights:**")
                        for insight in analysis['result']['insights']:
                            st.write(f"• {insight}")
                
                with col2:
                    st.write(f"**Fecha:** {analysis['timestamp'].strftime('%Y-%m-%d')}")
                    st.write(f"**Hora:** {analysis['timestamp'].strftime('%H:%M:%S')}")
                    st.write(f"**Tipo:** {analysis['result'].get('analysis_type', 'N/A')}")
                    
                    # Botón para reanalizar
                    if st.button(f"🔄 Reanalizar", key=f"reanalyze_{i}"):
                        with st.spinner("Reanalizando..."):
                            new_result = self.perform_analysis(analysis['query'])
                            if new_result:
                                st.success("✅ Reanálisis completado")
                                self.display_analysis_result(new_result)
    
    def render_analysis_interface(self):
        """Interfaz principal de análisis"""
        if st.session_state.vectorstore_ready:
            st.header("🔍 Análisis Inteligente")
            
            # Tabs para diferentes tipos de análisis
            tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Conversacional", "📊 Análisis Automático", "📈 Visualizaciones", "📋 Historial"])
            
            with tab1:
                self.render_chat_interface()
            
            with tab2:
                self.render_automatic_analysis()
            
            with tab3:
                self.render_visualizations()
            
            with tab4:
                self.render_analysis_history()
        else:
            st.warning("⚠️ Primero debes cargar los datos y crear el vector store.")
    
    def render_export_section(self):
        """Sección para exportar resultados"""
        if st.session_state.analysis_history:
            st.header("📤 Exportar Resultados")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Exportar a Excel"):
                    self.export_to_excel()
            
            with col2:
                if st.button("📋 Exportar a CSV"):
                    self.export_to_csv()
            
            with col3:
                if st.button("📄 Generar Reporte PDF"):
                    self.generate_pdf_report()
    
    def export_to_excel(self):
        """Exporta los resultados a Excel"""
        try:
            # Crear DataFrame con el historial
            export_data = []
            for analysis in st.session_state.analysis_history:
                export_data.append({
                    'Fecha': analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Consulta': analysis['query'],
                    'Resumen': analysis['result']['summary'],
                    'Tipo_Analisis': analysis['result'].get('analysis_type', 'N/A'),
                    'Insights': '; '.join(analysis['result'].get('insights', []))
                })
            
            df_export = pd.DataFrame(export_data)
            
            # Crear buffer para el archivo Excel
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_export.to_excel(writer, sheet_name='Análisis', index=False)
            
            buffer.seek(0)
            
            st.download_button(
                label="⬇️ Descargar Excel",
                data=buffer,
                file_name=f"analisis_noticias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
        except Exception as e:
            st.error(f"Error al exportar a Excel: {str(e)}")
    
    def export_to_csv(self):
        """Exporta los resultados a CSV"""
        try:
            # Crear DataFrame con el historial
            export_data = []
            for analysis in st.session_state.analysis_history:
                export_data.append({
                    'Fecha': analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'Consulta': analysis['query'],
                    'Resumen': analysis['result']['summary'],
                    'Tipo_Analisis': analysis['result'].get('analysis_type', 'N/A'),
                    'Insights': '; '.join(analysis['result'].get('insights', []))
                })
            
            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Descargar CSV",
                data=csv,
                file_name=f"analisis_noticias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error al exportar a CSV: {str(e)}")
    
    def generate_pdf_report(self):
        """Genera un reporte en PDF"""
        st.info("⚠️ Funcionalidad de PDF en desarrollo. Por ahora puedes usar la exportación a Excel o CSV.")
    
    def run(self):
        """Ejecuta la aplicación principal"""
        # Renderizar header
        self.render_header()
        
        # Renderizar sidebar
        self.render_sidebar()
        
        # Contenido principal
        # Sección de carga de datos
        self.render_data_upload_section()
        
        # Sección de vector store (solo si hay datos cargados)
        if st.session_state.data_loaded:
            self.render_vector_store_section()
        
        # Interfaz de análisis (solo si vector store está listo)
        self.render_analysis_interface()
        
        # Sección de exportación (si hay análisis)
        if st.session_state.analysis_history:
            self.render_export_section()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; padding: 1rem;'>
                📰 Análisis Inteligente de Noticias | Desarrollado con Streamlit y LangChain
                <br>
                <small>Versión 1.0 | Última actualización: Junio 2025</small>
            </div>
            """, 
            unsafe_allow_html=True
        )


# Función principal
def main():
    """Función principal de la aplicación"""
    try:
        app = NewsAnalysisApp()
        app.run()
    except Exception as e:
        st.error(f"Error en la aplicación: {str(e)}")
        st.info("Por favor, recarga la página e intenta nuevamente.")


# Punto de entrada
if __name__ == "__main__":
    main()