import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from embeddings import STEmbeddingsWrapper
from config import BASE_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP

emb_wrapper = STEmbeddingsWrapper()


def running_in_streamlit_cloud():
    return "STREAMLIT_SERVER_CONFIG" in os.environ


def safe_chroma_load(persist_dir: str):
    if running_in_streamlit_cloud():
        return None
    try:
        return Chroma(persist_directory=persist_dir, embedding=emb_wrapper)
    except Exception:
        return None


def build_or_load_chroma(agent_name: str, url):
    persist_dir = os.path.join(BASE_DB_DIR, agent_name)

    # CASE 1: try to load local chroma persist
    if not running_in_streamlit_cloud():
        if os.path.isdir(persist_dir) and os.listdir(persist_dir):
            db = safe_chroma_load(persist_dir)
            if db:
                return db

    # CASE 2: JSON persona file
    if isinstance(url, str) and url.endswith(".json") and os.path.exists(url):
        with open(url, "r", encoding="utf-8") as f:
            data = f.read()

        split_docs = [Document(page_content=data)]

        if running_in_streamlit_cloud():
            return Chroma.from_documents(split_docs, embedding=emb_wrapper)

        db = Chroma.from_documents(split_docs, embedding=emb_wrapper, persist_directory=persist_dir)
        db.persist()
        return db

    # CASE 3: URLs (single or list)
    all_docs = []

    if isinstance(url, list):
        urls = url
    else:
        urls = [url]

    for u in urls:
        try:
            loader = WebBaseLoader(u)
            docs = loader.load()
            docs = [d for d in docs if d.page_content and d.page_content.strip()]
            all_docs.extend(docs)
        except:
            pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(all_docs)
    split_docs = [d for d in split_docs if d.page_content and d.page_content.strip()]

    if running_in_streamlit_cloud():
        return Chroma.from_documents(split_docs, embedding=emb_wrapper)

    db = Chroma.from_documents(split_docs, embedding=emb_wrapper, persist_directory=persist_dir)
    db.persist()
    return db
