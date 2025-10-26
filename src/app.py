import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import sys
import os

load_dotenv()

# --- Verificación de Entorno y Configuración ---
# Control de errores: Asegura que la clave API esté configurada antes de ejecutar
if not os.getenv("GOOGLE_API_KEY"):
    print("Se necesita la variable de entorno \"GOOGLE_API_KEY\"", file=sys.stderr)
    sys.exit(1)

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# --- Plantilla de Prompt (Prompt Engineering) ---
# Define el rol del LLM y las instrucciones clave para el RAG.
PROMPT_TEMPLATE = """
Eres un Consultor de Buenas Prácticas veterinarias de SENASA.
Responde la pregunta del usuario basándote **únicamente** en los siguientes documentos de contexto.
Si la información no se encuentra en el contexto, indica amablemente que no tienes la información.
Tu respuesta debe ser completa, profesional y citar **todas las fuentes** de donde extrajiste la información.

Contexto:
{context}

Pregunta: {question}
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# Inicializa el LLM de Google. Una temperatura baja (0.1) promueve respuestas más determinísticas y fácticas
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.1)

# Define la cadena de procesamiento de generación: Prompt -> LLM -> Parser de salida
rag_chain_generation = prompt | llm | StrOutputParser()

# --- Carga de Componentes de RAG (Eficiencia con Streamlit) ---
# Decorador de Streamlit para cachear el resultado de la función.
# Esto asegura que la base de datos se cargue solo una vez, incluso con las interacciones del usuario.
@st.cache_resource
def load_rag_components():
    """Carga la función de embeddings y la base de datos vectorial ChromaDB."""

    print(f"🔄 Cargando base de datos Chroma desde: {CHROMA_PATH}")
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # Inicializa la base de datos ChromaDB desde el disco.
    vector_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
    print("✅ Base de datos cargada.")

    # Crea un 'retriever' que es la interfaz para buscar documentos relevantes.
    # search_kwargs={"k": 3} indica que se deben obtener los 3 fragmentos (chunks) más relevantes.
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    return retriever

retriever = load_rag_components()

def rag_invoke(query: str):
    """
    Realiza los pasos manuales de RAG:
    1. Retrieval: Obtiene los documentos relevantes.
    2. Generation: Pasa el contexto y la pregunta al LLM.
    """
    
    retrieved_docs = retriever.invoke(query)
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    response_text = rag_chain_generation.invoke(
        {"context": context_text, "question": query}
    )
    
    return response_text, retrieved_docs


st.set_page_config(
    page_title="Consultor RAG de Guías Técnicas",
    layout="centered"
)

st.title("🌿 Consultor de Buenas Prácticas veterinarias SENASA")
st.markdown("Pregunta sobre los manuales técnicos cargados.")

# Inicializa el historial del chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Muestra mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Manejo de nueva entrada del usuario
if user_input := st.chat_input("Escribí tu pregunta..."):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Buscando y generando respuesta..."):
            
            full_answer, sources = rag_invoke(user_input)
            
            # Extracción y formato de las fuentes
            # Obtenemos los nombres de archivo únicos de la metadata de los documentos
            unique_sources = set([doc.metadata.get('source', 'Fuente Desconocida') for doc in sources])
            
            # Construir el bloque de fuentes para el usuario
            sources_text = "\n\n---\n\n**Fuentes consultadas:**\n"
            for source in sorted(list(unique_sources)):
                # Obtiene solo el nombre del archivo para mostrar
                file_name = os.path.basename(source)
                sources_text += f"- `{file_name}`\n"
            
            final_output = full_answer + sources_text
            
            st.markdown(final_output)

        # Añadir respuesta del asistente al historial
        st.session_state.messages.append({"role": "assistant", "content": final_output})