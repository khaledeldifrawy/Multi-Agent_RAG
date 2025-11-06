from sentence_transformers import SentenceTransformer
from config import SENTENCE_MODEL
from typing import List

class STEmbeddingsWrapper:
    def __init__(self, model_name: str = SENTENCE_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False, convert_to_numpy=True).tolist()[0]
