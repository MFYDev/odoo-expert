from typing import List, Dict, Optional, Tuple
from llama_index.llms.ollama.base import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from src.core.services.embedding import EmbeddingService
from src.core.services.db_service import DatabaseService
from src.config.settings import settings
from src.utils.logging import logger

class ChatService:
    def __init__(
        self,
        llm: Ollama,
        db_service: DatabaseService,
        embedding_service: EmbeddingService
    ):
        self.llm = llm
        self.db_service = db_service
        self.embedding_service = embedding_service

    async def retrieve_relevant_chunks(
        self,
        query: str,
        version: int,
        limit: int = 6
    ) -> List[Dict]:
        try:
            query_embedding = await self.embedding_service.get_embedding(query)
            chunks = await self.db_service.search_documents(
                query_embedding,
                version,
                limit
            )
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise

    def prepare_context(self, chunks: List[Dict]) -> Tuple[str, List[Dict[str, str]]]:
        """Prepare context and sources from retrieved chunks."""
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            source_info = (
                f"Context:\n"
                f"Document: {chunk['url']}\n"
                f"Title: {chunk['title']}\n"
                f"Content: {chunk['content']}"
            )
            context_parts.append(source_info)
            sources.append({
                "url": chunk["url"],
                "title": chunk["title"]
            })
        
        return "\n\n---\n\n".join(context_parts), sources

    async def generate_response(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict]] = None,
        stream: bool = False
    ):
        """Generate AI response based on query and context."""
        try:
            messages = [ChatMessage(
                role=MessageRole.SYSTEM,
                content=settings.SYSTEM_PROMPT,
            )]
            
            if conversation_history:
                history_text = "\n".join([
                    f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                    for msg in conversation_history[-3:]
                ])
                messages.append(ChatMessage(
                    role=MessageRole.USER,
                    content=f"Previous conversation:\n{history_text}",
                ))

            messages.append(ChatMessage(
                role=MessageRole.USER,
                content=f"Question: {query}\n\nRelevant documentation:\n{context}",
            ))

            if stream:
                response_stream = await self.llm.astream_chat(messages)
                return response_stream
            else:
                response = self.llm.chat(messages)
                return response.message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise