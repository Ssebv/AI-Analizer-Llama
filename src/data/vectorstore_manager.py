# vectorstore_manager.py - Versión sin warnings de deprecación
import os
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.schema import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List, Dict, Optional, Any
import tempfile
import shutil

class InMemoryChatHistory(BaseChatMessageHistory):
    """Implementación simple de historial de chat en memoria"""
    
    def __init__(self, max_messages: int = 6):
        self.messages: List[BaseMessage] = []
        self.max_messages = max_messages
    
    def add_message(self, message: BaseMessage) -> None:
        """Añade un mensaje al historial"""
        self.messages.append(message)
        # Mantener solo los últimos max_messages mensajes
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def clear(self) -> None:
        """Limpia el historial"""
        self.messages = []

class NewsVectorStoreManager:
    """Gestor de vector store especializado para noticias - Versión sin warnings"""
    
    def __init__(self, 
                 model_name: str = "llama3:8b",
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "news_collection"):
        
        # Configurar embeddings
        self.embeddings = OllamaEmbeddings(
            model=model_name,
            base_url="http://localhost:11434"
        )
        
        # Configurar LLM para compresión contextual
        self.llm = OllamaLLM(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.1
        )
        
        self.persist_directory = os.path.abspath(persist_directory)
        self.collection_name = collection_name
        self.vectorstore = None
        self.retriever = None
        
        # Crear directorio si no existe
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Crea el vector store con los documentos - Método simplificado"""
        
        print(f"🔄 Creando vector store con {len(documents)} documentos...")
        
        try:
            # Limpiar directorio existente para evitar conflictos
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Crear cliente ChromaDB con configuración mínima
            chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Crear vector store
            self.vectorstore = Chroma(
                client=chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            
            # Agregar documentos por lotes más pequeños
            batch_size = 25  # Reducido para mayor estabilidad
            total_batches = (len(documents) - 1) // batch_size + 1
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                try:
                    self.vectorstore.add_documents(batch)
                    print(f"📝 Procesado lote {batch_num}/{total_batches} ({len(batch)} docs)")
                except Exception as e:
                    print(f"⚠️ Error en lote {batch_num}: {str(e)[:100]}...")
                    # Intentar documento por documento en caso de error
                    for j, doc in enumerate(batch):
                        try:
                            self.vectorstore.add_documents([doc])
                        except Exception as doc_error:
                            print(f"❌ Error en documento {i+j+1}: {str(doc_error)[:50]}...")
                            continue

            print("✅ Vector store creado exitosamente")
            return self.vectorstore
            
        except Exception as e:
            print(f"❌ Error crítico creando vector store: {e}")
            # Fallback: usar vector store en memoria
            return self._create_memory_vectorstore(documents)
    
    def _create_memory_vectorstore(self, documents: List[Document]) -> Chroma:
        """Fallback: crear vector store en memoria"""
        print("🔄 Creando vector store en memoria como fallback...")
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=f"{self.collection_name}_memory"
            )
            print("✅ Vector store en memoria creado")
            return self.vectorstore
        except Exception as e:
            print(f"❌ Error crítico: {e}")
            raise e
    
    def load_existing_vectorstore(self) -> Optional[Chroma]:
        """Carga un vector store existente"""
        try:
            if not os.path.exists(self.persist_directory):
                print("📁 No existe directorio de persistencia")
                return None
                
            chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Verificar si la colección existe
            try:
                collection = chroma_client.get_collection(self.collection_name)
                if collection.count() == 0:
                    print("📊 Colección vacía")
                    return None
            except:
                print("📊 Colección no encontrada")
                return None
            
            self.vectorstore = Chroma(
                client=chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            
            print("✅ Vector store cargado exitosamente")
            return self.vectorstore
            
        except Exception as e:
            print(f"❌ Error cargando vector store: {e}")
            return None
    
    def create_advanced_retriever(self, 
                                search_type: str = "similarity",  # Cambiado de "mmr" a "similarity"
                                k: int = 6,
                                use_compression: bool = False) -> Any:  # Deshabilitado por defecto
        """Crea un retriever - Versión simplificada"""
        
        if self.vectorstore is None:
            raise ValueError("Vector store no inicializado. Ejecuta create_vectorstore primero.")
        
        # Retriever base simplificado
        search_kwargs = {"k": k}
        
        if search_type == "mmr":
            search_kwargs.update({
                "fetch_k": min(k * 3, 20),
                "lambda_mult": 0.7
            })
        
        base_retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        if use_compression:
            try:
                compressor = LLMChainExtractor.from_llm(self.llm)
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
            except Exception as e:
                print(f"⚠️ Error con compresión, usando retriever simple: {e}")
                self.retriever = base_retriever
        else:
            self.retriever = base_retriever
        
        return self.retriever
    
    def search_with_filters(self, 
                           query: str,
                           filters: Optional[Dict] = None,
                           k: int = 6) -> List[Document]:
        """Búsqueda con filtros - Versión simplificada"""
        
        if self.vectorstore is None:
            raise ValueError("Vector store no inicializado")
        
        try:
            # Búsqueda simple sin filtros complejos por ahora
            docs = self.vectorstore.similarity_search(query, k=k)
            
            # Filtrado post-búsqueda si se proporcionan filtros
            if filters and docs:
                filtered_docs = []
                for doc in docs:
                    include_doc = True
                    
                    # Filtro por medios
                    if 'medios' in filters and doc.metadata.get('medio'):
                        if doc.metadata['medio'] not in filters['medios']:
                            include_doc = False
                    
                    # Filtro por secciones
                    if 'secciones' in filters and doc.metadata.get('seccion'):
                        if doc.metadata['seccion'] not in filters['secciones']:
                            include_doc = False
                    
                    if include_doc:
                        filtered_docs.append(doc)
                
                return filtered_docs[:k]
            
            return docs
            
        except Exception as e:
            print(f"⚠️ Error en búsqueda: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """Obtiene estadísticas de la colección"""
        if self.vectorstore is None:
            return {"error": "Vector store no inicializado"}
        
        try:
            # Intentar obtener estadísticas básicas
            test_docs = self.vectorstore.similarity_search("test", k=1)
            
            if not test_docs:
                return {
                    "total_documents": 0,
                    "status": "vacío"
                }
            
            # Obtener muestra para estadísticas
            sample_docs = self.vectorstore.similarity_search("", k=5)
            
            medios = set()
            secciones = set()
            fechas = []
            
            for doc in sample_docs:
                if 'medio' in doc.metadata:
                    medios.add(doc.metadata['medio'])
                if 'seccion' in doc.metadata:
                    secciones.add(doc.metadata['seccion'])
                if 'fecha' in doc.metadata:
                    fechas.append(doc.metadata['fecha'])
            
            return {
                "total_documents": len(sample_docs),
                "sample_medios": list(medios),
                "sample_secciones": list(secciones),
                "sample_fechas": fechas,
                "status": "activo"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def reset_collection(self):
        """Resetea la colección"""
        try:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                os.makedirs(self.persist_directory, exist_ok=True)
                print("✅ Colección reseteada")
        except Exception as e:
            print(f"❌ Error reseteando: {e}")


class NewsRAGChain:
    """Chain RAG especializada para análisis de noticias - Sin warnings de deprecación"""
    
    def __init__(self, vectorstore_manager: NewsVectorStoreManager):
        self.vectorstore_manager = vectorstore_manager
        self.llm = vectorstore_manager.llm
        
        # Usar historial personalizado en lugar de ConversationBufferWindowMemory
        self.chat_history = InMemoryChatHistory(max_messages=6)
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template="""Eres un analista experto de noticias chilenas. 
Analiza el contexto de noticias para responder la pregunta.

Contexto: {context}

Historial de conversación: {chat_history}

Pregunta: {question}

Responde de forma clara y concisa, mencionando fuentes cuando sea relevante.

Respuesta:"""
        )
        
        self.chain = None
    
    def _format_chat_history(self) -> str:
        """Formatea el historial de chat para el prompt"""
        if not self.chat_history.messages:
            return "Sin historial previo."
        
        formatted_history = []
        for message in self.chat_history.messages[-4:]:  # Solo últimos 4 mensajes
            if isinstance(message, HumanMessage):
                formatted_history.append(f"Usuario: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Asistente: {message.content}")
        
        return "\n".join(formatted_history)
    
    def create_chain(self):
        """Crea la cadena conversacional - Sin ConversationBufferWindowMemory"""
        retriever = self.vectorstore_manager.create_advanced_retriever(
            use_compression=False  # Deshabilitado para mayor estabilidad
        )
        
        # Usar ConversationalRetrievalChain sin memory (manejamos el historial manualmente)
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            combine_docs_chain_kwargs={"prompt": self.prompt_template},
            return_source_documents=True,
            output_key="answer"
        )
        
        return self.chain
    
    def query(self, question: str) -> Dict:
        """Realiza una consulta al sistema RAG"""
        if self.chain is None:
            self.create_chain()
        
        try:
            # Usar la lista real de mensajes; si no hay, pasa lista vacía
            history_messages = self.chat_history.messages if self.chat_history.messages else []
            
            result = self.chain.invoke({
                "question": question,
                "chat_history": history_messages
            })
            
            # Guardar en historial personalizado
            self.chat_history.add_message(HumanMessage(content=question))
            self.chat_history.add_message(AIMessage(content=result['answer']))
            
            sources = []
            for doc in result.get('source_documents', []):
                source_info = {
                    "titulo": doc.metadata.get('titulo', 'Sin título'),
                    "medio": doc.metadata.get('medio', 'Desconocido'),
                    "fecha": doc.metadata.get('fecha', 'Sin fecha'),
                    "seccion": doc.metadata.get('seccion', 'Sin sección'),
                    "texto_muestra": doc.page_content[:150] + "..."
                }
                sources.append(source_info)
            
            return {
                "answer": result['answer'],
                "sources": sources,
                "question": question
            }
            
        except Exception as e:
            return {
                "error": f"Error procesando consulta: {str(e)}",
                "question": question
            }
    
    def clear_history(self):
        """Limpia el historial de conversación"""
        self.chat_history.clear()
        print("✅ Historial de conversación limpiado")
    
    def get_history_summary(self) -> Dict:
        """Obtiene un resumen del historial actual"""
        return {
            "total_messages": len(self.chat_history.messages),
            "last_messages": [
                {
                    "type": type(msg).__name__,
                    "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                }
                for msg in self.chat_history.messages[-4:]
            ]
        }

# Alias para mantener compatibilidad
Ollama = OllamaLLM