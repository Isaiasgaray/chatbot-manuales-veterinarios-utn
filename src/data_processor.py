from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import os

CHROMA_PATH = "chroma_db"
FILE_PATH = "data/documents"
all_documents = list()

for path in os.listdir(FILE_PATH):
    loader = PyPDFLoader(f"{FILE_PATH}/{path}")
    document = loader.load()
    all_documents.extend(document)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""],
)

all_chunks = text_splitter.split_documents(all_documents)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_function,
    persist_directory=CHROMA_PATH
)

vector_db.persist()