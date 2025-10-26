# Importación de librerías esenciales de LangChain y utilidades
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os

# --- Configuración de Rutas ---
# Directorio donde se almacenará la base de datos vectorial
CHROMA_PATH = "chroma_db"
# Directorio que contiene los documentos PDF de origen (manuales SENASA)
FILE_PATH = "data/documents"
all_documents = list()

# --- Carga de Documentos ---
# Iteramos sobre todos los archivos en el directorio de documentos
for path in os.listdir(FILE_PATH):
    # Utilizamos PyPDFLoader para manejar la extracción de texto de los PDFs
    loader = PyPDFLoader(f"{FILE_PATH}/{path}")
    document = loader.load()
    all_documents.extend(document)

# --- División de Texto (Chunking) ---
# Creamos un separador de texto recursivo para dividir documentos grandes en fragmentos (chunks)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)

# Aplicamos el splitter a todos los documentos
all_chunks = text_splitter.split_documents(all_documents)

# --- Generación de Embeddings ---
# Cargamos el modelo de embeddings. Este modelo convierte texto en vectores numéricos.
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Creación y Persistencia de la Base de Datos Vectorial ---
# Creamos la base de datos ChromaDB desde los fragmentos de texto
vector_db = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_function,
    persist_directory=CHROMA_PATH
)

# Guardamos los datos en el disco para que puedan ser cargados por 'app.py' sin reprocesar.
vector_db.persist()