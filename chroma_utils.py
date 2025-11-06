import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import STEmbeddingsWrapper
from config import BASE_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP

emb_wrapper = STEmbeddingsWrapper()

def safe_chroma_load(persist_dir: str):
    try:
        return Chroma(persist_directory=persist_dir, embedding=emb_wrapper)
    except Exception:
        return None

def build_or_load_chroma(agent_name: str, url: str):
    persist_dir = os.path.join(BASE_DB_DIR, agent_name)
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        db = safe_chroma_load(persist_dir)
        if db:
            return db

    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)

    db = Chroma.from_documents(split_docs, embedding=emb_wrapper, persist_directory=persist_dir)
    db.persist()
    return db
