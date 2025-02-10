from typing import List
from llama_index.embeddings.ollama import OllamaEmbedding
from src.utils.logging import logger

class EmbeddingService:
    def __init__(self, embed_model: OllamaEmbedding):
        self.embed_model = embed_model

    async def get_embedding(self, text: str) -> List[float]:
        try:
            text = text.replace("\n", " ")
            if len(text) > 8000:
                text = text[:8000] + "..."
                
            return await self.embed_model.aget_general_text_embedding(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise