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
    
class CleanRetriever:
    def __init__(self, base_retriever):
        self.base_retriever = base_retriever

    def get_relevant_documents(self, query: str):
        docs = self.base_retriever.get_relevant_documents(query)
        # فلتر بسيط يحذف None / non-str / empty
        docs = [d for d in docs if d.page_content and isinstance(d.page_content, str) and d.page_content.strip()]
        return docs

    async def aget_relevant_documents(self, query: str):
        docs = await self.base_retriever.aget_relevant_documents(query)
        docs = [d for d in docs if d.page_content and isinstance(d.page_content, str) and d.page_content.strip()]
        return docs

