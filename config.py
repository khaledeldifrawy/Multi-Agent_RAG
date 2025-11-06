import os

# ---------------- Config ----------------
AGENT_URLS = {
    "Python": "https://docs.python.org/3/",
    "ML": "https://scikit-learn.org/stable/user_guide.html",
    "DL": "https://huggingface.co/docs/transformers/index",
    "LangChain": "https://python.langchain.com/docs/",
    "LLM": "https://docs.llamaindex.ai/",
    "Deploy": "https://groq.com/blog/",
}

BASE_DB_DIR = "./agent_dbs"
os.makedirs(BASE_DB_DIR, exist_ok=True)

MEMORY_FILE = os.path.join(BASE_DB_DIR, "agent_memory.json")

SENTENCE_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 4
