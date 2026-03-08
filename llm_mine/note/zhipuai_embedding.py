from typing import List
from langchain_core.embeddings import Embeddings

class ZhipuAIEmbedding(Embeddings):
    def __init__(self):
        from zhipuai import ZhipuAI 
        self.client = ZhipuAI()
    
    def embed_documents(self,texts: List[str]) -> List[List[float]]:
        result = []
        for i in range(0,len(texts),64):
            embeddings = self.client.embeddings.create(
                model = 'embedding-3',
                input = texts[i:i+64]                
            )
            result.extend([embedding.embedding for embedding in embeddings.data])
        return result
    def embed_query(self, text):
        return self.embed_documents([text])[0]