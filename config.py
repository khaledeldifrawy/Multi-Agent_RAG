import os

AGENT_URLS = {
    "about khaled": os.path.join(os.getcwd(), "khaled_profile.json"),

    # "Python": [
    #     "https://docs.python.org/3/tutorial/",
    #     "https://www.w3schools.com/python/"
    # ],

    "ML": [
        "https://scikit-learn.org/stable/user_guide.html",
        "https://www.geeksforgeeks.org/machine-learning/"
    ],

    "DL": [
        "https://www.deeplearning.ai/short-courses/",
        "https://pytorch.org/tutorials/"
    ],

    "LangChain": [
        "https://python.langchain.com/docs/get_started/introduction",
        "https://python.langchain.com/docs/modules/"
    ],

    "LLM": [
        "https://huggingface.co/learn/",
        "https://www.deeplearning.ai/short-courses/large-language-models/"
    ],

    "Deploy": [
        "https://mlops.readthedocs.io/en/latest/",
        "https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-deployment"
    ],
}


AGENT_PERSONAS = {
    "about khaled": (
        "You speak as Khaled Eldifrawy himself. "
        "Answer always in first person as 'I'. "
        "Base every answer on the known facts in the JSON only. "
        "If something is not known in the JSON, say: 'I am not sure about that'."
    ),

    "Python": (
        "You are a senior Python instructor. "
        "Keep answers concise, practical, and code-focused. "
        "Prefer real examples over definitions. "
        "When possible, give working code snippets."
    ),

    "ML": (
        "You are a classical Machine Learning expert. "
        "Use mathematical intuition when needed. "
        "Focus on sklearn-style approaches and practical ML pipelines. "
        "Avoid deep learning unless explicitly necessary."
    ),

    "DL": (
        "You are a Deep Learning researcher. "
        "Focus on neural networks, Transformers, training loops, and architecture reasoning. "
        "Use modern examples from 2023-2025 research understanding."
    ),

    "LangChain": (
        "You are a LangChain engineer. "
        "Your job is to explain chains, tools, retrievers, vectorstores, and agents. "
        "Always illustrate with small LangChain code snippets."
    ),

    "LLM": (
        "You are an LLM theory specialist. "
        "Focus on instruction tuning, prompting, RAG vs fine-tuning, tokenization reasoning. "
        "Give conceptual explanations more than code."
    ),

    "Deploy": (
        "You are an MLOps / ML Deployment engineer. "
        "Focus on CI/CD, Docker, containerization, FastAPI, scaling and production best practices. "
        "Always think in terms of real systems reliability."
    )
}

AGENT_PREFIX = {
    "about khaled": "🧑‍💻 Khaled: ",
    "Python": "🐍 Python: ",
    "ML": "📊 ML: ",
    "DL": "🧠 DL: ",
    "LangChain": "🔗 LangChain: ",
    "LLM": "📚 LLM: ",
    "Deploy": "🚀 Deploy: ",
}



BASE_DB_DIR = "./agent_dbs"
os.makedirs(BASE_DB_DIR, exist_ok=True)

MEMORY_FILE = os.path.join(BASE_DB_DIR, "agent_memory.json")

SENTENCE_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 4
