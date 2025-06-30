# news_agents_graph.py
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.tools import BaseTool
from langchain.llms import Ollama
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel, Field
import json
import pandas as pd
from datetime import datetime, timedelta

# Estado del sistema de agentes
class NewsAnalysisState(BaseModel):
    """Estado compartido entre agentes"""
    query: str = Field(description="Consulta del usuario")
    analysis_type: str = Field(default="general", description="Tipo de análisis requerido")
    context_data: Dict = Field(default_factory=dict, description="Datos de contexto")
    temporal_analysis: Dict = Field(default_factory=dict, description="Análisis temporal")
    thematic_analysis: Dict = Field(default_factory=dict, description="Análisis temático")
    comparative_analysis: Dict = Field(default_factory=dict, description="Análisis comparativo")
    final_response: str = Field(default="", description="Respuesta final")
    relevant_docs: List = Field(default_factory=list, description="Documentos relevantes")
    metadata: Dict = Field(default_factory=dict, description="Metadatos adicionales")
    next_action: str = Field(default="", description="Próxima acción")

class NewsAgent:
    """Clase base para agentes especializados"""
    
    def __init__(self, name: str, llm: Ollama, vectorstore_manager):
        self.name = name
        self.llm = llm
        self.vectorstore_manager = vectorstore_manager
    
    def _create_prompt(self, system_prompt: str, user_input: str, context: str = "") -> str:
        """Crea prompt estructurado para el agente"""
        return f"""Sistema: {system_prompt}

Contexto: {context}

Consulta del usuario: {user_input}

Responde de manera estructurada y concisa."""

class AnalyticalAgent(NewsAgent):
    """Agente especializado en análisis de patrones y tendencias"""
    
    def __init__(self, llm: Ollama, vectorstore_manager):
        super().__init__("AnalyticalAgent", llm, vectorstore_manager)
    
    def analyze_patterns(self, state: NewsAnalysisState) -> NewsAnalysisState:
        """Analiza patrones en las noticias"""
        
        # Buscar documentos relevantes
        docs = self.vectorstore_manager.semantic_search_with_context(
            state.query, 
            context_type="tematico",
            k=10
        )
        
        # Extraer información para análisis
        medios = [doc.metadata.get('medio', 'Desconocido') for doc in docs]
        secciones = [doc.metadata.get('seccion', 'Desconocido') for doc in docs]
        fechas = [doc.metadata.get('fecha', '') for doc in docs if doc.metadata.get('fecha')]
        
        # Análisis de patrones
        pattern_analysis = {
            'medios_frecuentes': pd.Series(medios).value_counts().head(5).to_dict(),
            'secciones_frecuentes': pd.Series(secciones).value_counts().head(5).to_dict(),
            'total_documentos': len(docs),
            'rango_temporal': {
                'fechas_disponibles': len(fechas),
                'primera_fecha': min(fechas) if fechas else None,
                'ultima_fecha': max(fechas) if fechas else None
            }
        }
        
        # Crear prompt para análisis LLM
        context = f"""
        Documentos encontrados: {len(docs)}
        Medios más frecuentes: {pattern_analysis['medios_frecuentes']}
        Secciones más frecuentes: {pattern_analysis['secciones_frecuentes']}
        """
        
        prompt = self._create_prompt(
            "Eres un analista experto en identificar patrones y tendencias en noticias. "
            "Analiza los datos proporcionados y identifica patrones significativos.",
            state.query,
            context
        )
        
        # Generar análisis con LLM
        response = self.llm.invoke(prompt)
        
        # Actualizar estado
        state.thematic_analysis = pattern_analysis
        state.relevant_docs.extend(docs[:5])  # Agregar documentos más relevantes
        state.metadata['analytical_insights'] = response
        state.next_action = "temporal_analysis"
        
        return state

class TemporalAgent(NewsAgent):
    """Agente especializado en análisis temporal"""
    
    def __init__(self, llm: Ollama, vectorstore_manager):
        super().__init__("TemporalAgent", llm, vectorstore_manager)
    
    def analyze_temporal_patterns(self, state: NewsAnalysisState) -> NewsAnalysisState:
        """Analiza patrones temporales en las noticias"""
        
        # Obtener fechas específicas si la consulta las menciona
        filters = self._extract_temporal_filters(state.query)
        
        # Buscar con filtros temporales
        docs = self.vectorstore_manager.search_with_filters(
            state.query,
            filters=filters,
            k=15
        )
        
        # Análisis temporal
        temporal_data = []
        for doc in docs:
            if 'fecha' in doc.metadata and doc.metadata['fecha']:
                temporal_data.append({
                    'fecha': doc.metadata['fecha'],
                    'medio': doc.metadata.get('medio', 'Desconocido'),
                    'seccion': doc.metadata.get('seccion', 'General'),
                    'titulo': doc.metadata.get('titulo', '')
                })
        
        if temporal_data:
            df_temporal = pd.DataFrame(temporal_data)
            df_temporal['fecha'] = pd.to_datetime(df_temporal['fecha'])
            
            # Análisis por períodos
            temporal_analysis = {
                'distribucion_mensual': df_temporal.groupby(df_temporal['fecha'].dt.to_period('M')).size().to_dict(),
                'distribucion_semanal': df_temporal.groupby(df_temporal['fecha'].dt.to_period('W')).size().to_dict(),
                'tendencia_por_medio': df_temporal.groupby(['medio', df_temporal['fecha'].dt.to_period('M')]).size().to_dict(),
                'picos_actividad': self._identify_activity_peaks(df_temporal),
                'total_documentos_temporales': len(temporal_data)
            }
        else:
            temporal_analysis = {'error': 'No se encontraron datos temporales suficientes'}
        
        # Crear contexto para LLM
        context = f"Análisis temporal de {len(docs)} documentos encontrados"
        
        prompt = self._create_prompt(
            "Eres un analista especializado en patrones temporales de noticias. "
            "Identifica tendencias, picos de actividad y patrones estacionales.",
            state.query,
            context
        )
        
        response = self.llm.invoke(prompt)
        
        # Actualizar estado
        state.temporal_analysis = temporal_analysis
        state.metadata['temporal_insights'] = response
        state.next_action = "comparative_analysis"
        
        return state
    
    def _extract_temporal_filters(self, query: str) -> Dict:
        """Extrae filtros temporales de la consulta"""
        filters = {}
        
        # Buscar palabras clave temporales
        if 'último mes' in query.lower() or 'mes pasado' in query.lower():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            filters['fecha_inicio'] = start_date.strftime('%Y-%m-%d')
            filters['fecha_fin'] = end_date.strftime('%Y-%m-%d')
        
        elif 'última semana' in query.lower() or 'semana pasada' in query.lower():
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            filters['fecha_inicio'] = start_date.strftime('%Y-%m-%d')
            filters['fecha_fin'] = end_date.strftime('%Y-%m-%d')
        
        elif '2024' in query:
            filters['fecha_inicio'] = '2024-01-01'
            filters['fecha_fin'] = '2024-12-31'
        
        elif '2025' in query:
            filters['fecha_inicio'] = '2025-01-01'
            filters['fecha_fin'] = '2025-12-31'
        
        return filters
    
    def _identify_activity_peaks(self, df: pd.DataFrame) -> Dict:
        """Identifica picos de actividad en las noticias"""
        daily_counts = df.groupby(df['fecha'].dt.date).size()
        
        if len(daily_counts) > 0:
            mean_activity = daily_counts.mean()
            std_activity = daily_counts.std()
            threshold = mean_activity + (2 * std_activity)
            
            peaks = daily_counts[daily_counts > threshold]
            
            return {
                'picos_detectados': len(peaks),
                'fechas_pico': peaks.to_dict(),
                'actividad_promedio': mean_activity,
                'umbral_pico': threshold
            }
        
        return {'error': 'Datos insuficientes para detectar picos'}

class ComparativeAgent(NewsAgent):
    """Agente especializado en análisis comparativo"""
    
    def __init__(self, llm: Ollama, vectorstore_manager):
        super().__init__("ComparativeAgent", llm, vectorstore_manager)
    
    def analyze_comparative_patterns(self, state: NewsAnalysisState) -> NewsAnalysisState:
        """Realiza análisis comparativo entre medios, secciones, etc."""
        
        # Usar documentos ya encontrados por otros agentes
        docs = state.relevant_docs
        
        if not docs:
            # Si no hay documentos, buscar algunos
            docs = self.vectorstore_manager.semantic_search_with_context(
                state.query,
                context_type="geografico",
                k=12
            )
        
        # Análisis comparativo por medios
        medios_analysis = self._compare_by_media(docs)
        
        # Análisis comparativo por secciones
        sections_analysis = self._compare_by_sections(docs)
        
        # Análisis de cobertura
        coverage_analysis = self._analyze_coverage_patterns(docs)
        
        comparative_analysis = {
            'analisis_medios': medios_analysis,
            'analisis_secciones': sections_analysis,
            'analisis_cobertura': coverage_analysis,
            'total_fuentes_analizadas': len(set([doc.metadata.get('medio') for doc in docs]))
        }
        
        # Crear contexto para LLM
        context = f"""
        Análisis comparativo de {len(docs)} documentos:
        - Medios analizados: {len(medios_analysis)}
        - Secciones analizadas: {len(sections_analysis)}
        - Patrones de cobertura identificados
        """
        
        prompt = self._create_prompt(
            "Eres un analista especializado en análisis comparativo de medios de comunicación. "
            "Compara la cobertura, enfoques y patrones entre diferentes medios y secciones.",
            state.query,
            context
        )
        
        response = self.llm.invoke(prompt)
        
        # Actualizar estado
        state.comparative_analysis = comparative_analysis
        state.metadata['comparative_insights'] = response
        state.next_action = "synthesis"
        
        return state
    
    def _compare_by_media(self, docs: List) -> Dict:
        """Compara patrones por medio de comunicación"""
        media_data = {}
        
        for doc in docs:
            medio = doc.metadata.get('medio', 'Desconocido')
            if medio not in media_data:
                media_data[medio] = {
                    'count': 0,
                    'secciones': set(),
                    'fechas': [],
                    'titulos_muestra': []
                }
            
            media_data[medio]['count'] += 1
            media_data[medio]['secciones'].add(doc.metadata.get('seccion', 'General'))
            if doc.metadata.get('fecha'):
                media_data[medio]['fechas'].append(doc.metadata['fecha'])
            if doc.metadata.get('titulo'):
                media_data[medio]['titulos_muestra'].append(doc.metadata['titulo'])
        
        # Convertir sets a listas para serialización
        for medio in media_data:
            media_data[medio]['secciones'] = list(media_data[medio]['secciones'])
            media_data[medio]['titulos_muestra'] = media_data[medio]['titulos_muestra'][:3]
        
        return media_data
    
    def _compare_by_sections(self, docs: List) -> Dict:
        """Compara patrones por sección"""
        sections_data = {}
        
        for doc in docs:
            seccion = doc.metadata.get('seccion', 'General')
            if seccion not in sections_data:
                sections_data[seccion] = {
                    'count': 0,
                    'medios': set(),
                    'fechas': []
                }
            
            sections_data[seccion]['count'] += 1
            sections_data[seccion]['medios'].add(doc.metadata.get('medio', 'Desconocido'))
            if doc.metadata.get('fecha'):
                sections_data[seccion]['fechas'].append(doc.metadata['fecha'])
        
        # Convertir sets a listas
        for seccion in sections_data:
            sections_data[seccion]['medios'] = list(sections_data[seccion]['medios'])
        
        return sections_data
    
    def _analyze_coverage_patterns(self, docs: List) -> Dict:
        """Analiza patrones de cobertura"""
        coverage_patterns = {
            'diversidad_tematica': len(set([doc.metadata.get('seccion') for doc in docs])),
            'diversidad_medios': len(set([doc.metadata.get('medio') for doc in docs])),
            'concentracion_cobertura': self._calculate_coverage_concentration(docs),
            'distribucion_temporal': self._analyze_temporal_distribution(docs)
        }
        
        return coverage_patterns
    
    def _calculate_coverage_concentration(self, docs: List) -> Dict:
        """Calcula concentración de cobertura"""
        medio_counts = {}
        for doc in docs:
            medio = doc.metadata.get('medio', 'Desconocido')
            medio_counts[medio] = medio_counts.get(medio, 0) + 1
        
        if medio_counts:
            total_docs = len(docs)
            max_coverage = max(medio_counts.values())
            concentration_ratio = max_coverage / total_docs
            
            return {
                'medio_dominante': max(medio_counts, key=medio_counts.get),
                'porcentaje_dominante': concentration_ratio * 100,
                'distribucion': medio_counts
            }
        
        return {'error': 'No hay datos suficientes'}
    
    def _analyze_temporal_distribution(self, docs: List) -> Dict:
        """Analiza distribución temporal de la cobertura"""
        fechas = [doc.metadata.get('fecha') for doc in docs if doc.metadata.get('fecha')]
        
        if fechas:
            fechas_df = pd.to_datetime(fechas)
            return {
                'rango_temporal': {
                    'inicio': fechas_df.min().strftime('%Y-%m-%d'),
                    'fin': fechas_df.max().strftime('%Y-%m-%d')
                },
                'distribucion_mensual': fechas_df.groupby(fechas_df.dt.to_period('M')).size().to_dict()
            }
        
        return {'error': 'No hay fechas disponibles'}

class SynthesisAgent(NewsAgent):
    """Agente coordinador que sintetiza todos los análisis"""
    
    def __init__(self, llm: Ollama, vectorstore_manager):
        super().__init__("SynthesisAgent", llm, vectorstore_manager)
    
    def synthesize_analysis(self, state: NewsAnalysisState) -> NewsAnalysisState:
        """Sintetiza todos los análisis en una respuesta coherente"""
        
        # Recopilar todos los insights
        insights = {
            'analytical': state.metadata.get('analytical_insights', ''),
            'temporal': state.metadata.get('temporal_insights', ''),
            'comparative': state.metadata.get('comparative_insights', '')
        }
        
        # Crear contexto comprehensivo
        context = f"""
        ANÁLISIS TEMÁTICO:
        - Documentos analizados: {state.thematic_analysis.get('total_documentos', 0)}
        - Medios frecuentes: {state.thematic_analysis.get('medios_frecuentes', {})}
        - Secciones frecuentes: {state.thematic_analysis.get('secciones_frecuentes', {})}
        
        ANÁLISIS TEMPORAL:
        - Documentos temporales: {state.temporal_analysis.get('total_documentos_temporales', 0)}
        - Picos de actividad: {state.temporal_analysis.get('picos_actividad', {})}
        
        ANÁLISIS COMPARATIVO:
        - Fuentes analizadas: {state.comparative_analysis.get('total_fuentes_analizadas', 0)}
        - Diversidad temática: {state.comparative_analysis.get('analisis_cobertura', {}).get('diversidad_tematica', 0)}
        """
        
        # Prompt para síntesis final
        prompt = f"""
        Eres un analista senior especializado en síntesis de información periodística.
        
        CONSULTA ORIGINAL: {state.query}
        
        CONTEXTO DE ANÁLISIS:
        {context}
        
        INSIGHTS DE AGENTES ESPECIALIZADOS:
        Análisis Temático: {insights['analytical']}
        Análisis Temporal: {insights['temporal']}
        Análisis Comparativo: {insights['comparative']}
        
        INSTRUCCIONES:
        1. Sintetiza los hallazgos más importantes
        2. Identifica patrones y tendencias clave
        3. Proporciona insights accionables
        4. Mantén un tono profesional y analítico
        5. Estructura la respuesta de forma clara y concisa
        6. Incluye datos específicos cuando sea relevante
        
        RESPUESTA FINAL:
        """
        
        # Generar síntesis
        final_response = self.llm.invoke(prompt)
        
        # Actualizar estado final
        state.final_response = final_response
        state.next_action = "complete"
        
        return state

# Coordinador principal del grafo
class NewsAnalysisCoordinator:
    """Coordinador principal que orquesta el flujo de análisis"""
    
    def __init__(self, vectorstore_manager):
        self.vectorstore_manager = vectorstore_manager
        self.llm = vectorstore_manager.llm
        
        # Inicializar agentes
        self.analytical_agent = AnalyticalAgent(self.llm, vectorstore_manager)
        self.temporal_agent = TemporalAgent(self.llm, vectorstore_manager)
        self.comparative_agent = ComparativeAgent(self.llm, vectorstore_manager)
        self.synthesis_agent = SynthesisAgent(self.llm, vectorstore_manager)
        
        # Crear grafo
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Crea el workflow con LangGraph"""
        
        workflow = StateGraph(NewsAnalysisState)
        
        # Agregar nodos
        workflow.add_node("analytical", self.analytical_agent.analyze_patterns)
        workflow.add_node("temporal", self.temporal_agent.analyze_temporal_patterns)
        workflow.add_node("comparative", self.comparative_agent.analyze_comparative_patterns)
        workflow.add_node("synthesis", self.synthesis_agent.synthesize_analysis)
        
        # Definir flujo
        workflow.add_edge("analytical", "temporal")
        workflow.add_edge("temporal", "comparative")
        workflow.add_edge("comparative", "synthesis")
        workflow.add_edge("synthesis", END)
        
        # Punto de entrada
        workflow.set_entry_point("analytical")
        
        return workflow.compile()
    
    def analyze_query(self, query: str, analysis_type: str = "comprehensive") -> Dict:
        """Ejecuta el análisis completo de una consulta"""
        
        # Estado inicial
        initial_state = NewsAnalysisState(
            query=query,
            analysis_type=analysis_type
        )
        
        try:
            # Ejecutar workflow
            result = self.workflow.invoke(initial_state)
            
            # Preparar respuesta
            response = {
                "query": query,
                "analysis_type": analysis_type,
                "final_response": result.final_response,
                "detailed_analysis": {
                    "thematic": result.thematic_analysis,
                    "temporal": result.temporal_analysis,
                    "comparative": result.comparative_analysis
                },
                "relevant_documents": [
                    {
                        "titulo": doc.metadata.get('titulo', 'Sin título'),
                        "medio": doc.metadata.get('medio', 'Desconocido'),
                        "fecha": doc.metadata.get('fecha', 'Sin fecha'),
                        "seccion": doc.metadata.get('seccion', 'Sin sección')
                    }
                    for doc in result.relevant_docs[:5]
                ],
                "metadata": result.metadata,
                "status": "completed"
            }
            
            return response
            
        except Exception as e:
            return {
                "query": query,
                "error": f"Error en el análisis: {str(e)}",
                "status": "failed"
            }
    
    def get_workflow_status(self) -> Dict:
        """Obtiene el estado del workflow"""
        return {
            "agents": [
                "AnalyticalAgent",
                "TemporalAgent", 
                "ComparativeAgent",
                "SynthesisAgent"
            ],
            "vectorstore_status": self.vectorstore_manager.get_collection_stats(),
            "workflow_ready": self.workflow is not None
        }

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de inicialización y uso
    print("🚀 Inicializando sistema de agentes...")
    
    # Nota: Esto requiere que ya tengas el vectorstore_manager configurado
    # coordinator = NewsAnalysisCoordinator(vectorstore_manager)
    # 
    # result = coordinator.analyze_query(
    #     "¿Cuáles son las principales tendencias en las noticias deportivas del último mes?"
    # )
    # 
    # print("📊 Resultado del análisis:")
    # print(result['final_response'])