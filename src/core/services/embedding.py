from typing import List
from openai import AsyncOpenAI
from src.utils.logging import logger
from src.config.settings import settings
from langchain_ollama.embeddings import OllamaEmbeddings

class EmbeddingService:
    def __init__(self, client: AsyncOpenAI):
        # self.client = client
        self.ollama_embeddings = OllamaEmbeddings(model=settings.EMBEDDING_MODEL)

    async def get_embedding(self, text: str) -> List[float]:
        try:
            text = text.replace("\n", " ")
            if len(text) > 8000:
                text = text[:8000] + "..."
                
            # response = await self.client.embeddings.create(
            #     model="text-embedding-3-small",
            #     input=text
            # )
            # return response.data[0].embedding
            return await self.ollama_embeddings.aembed_query(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise