from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from src.config.model_provider import ModelProvider
from src.core.services.db_service import DatabaseService
from src.api.models.chat import ChatRequest, ChatResponse
from src.api.dependencies.auth import verify_token
from src.core.services.chat_service import ChatService
from src.core.services.embedding import EmbeddingService
from src.config.settings import settings
from src.utils.logging import logger

router = APIRouter()

# Create dependency for services
async def get_services():
    model_provider = ModelProvider()
    
    db_service = DatabaseService()
    embedding_service = EmbeddingService(model_provider.embed_model)
    chat_service = ChatService(model_provider.llm, db_service, embedding_service)
    
    return chat_service

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    authenticated: bool = Depends(verify_token),
    chat_service: ChatService = Depends(get_services)
):
    try:
        chunks = await chat_service.retrieve_relevant_chunks(
            request.query, 
            request.version
        )
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documentation found"
            )
        
        context, sources = chat_service.prepare_context(chunks)
        
        response = await chat_service.generate_response(
            query=request.query,
            context=context,
            conversation_history=request.conversation_history,
            stream=False
        )
        
        if not response:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate response"
            )
        
        return ChatResponse(
            answer=response,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/stream", response_class=StreamingResponse)
async def stream_endpoint(
    request: ChatRequest,
    authenticated: bool = Depends(verify_token),
    chat_service: ChatService = Depends(get_services)
):
    try:
        chunks = await chat_service.retrieve_relevant_chunks(
            request.query, 
            request.version
        )
        
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant documentation found"
            )
        
        context, sources = chat_service.prepare_context(chunks)
        
        stream = await chat_service.generate_response(
            query=request.query,
            context=context,
            conversation_history=request.conversation_history,
            stream=True
        )
        
        async def generate():
            try:
                async for chunk in stream:
                    if (hasattr(chunk, 'choices') and 
                        chunk.choices and 
                        hasattr(chunk.choices[0], 'delta') and 
                        hasattr(chunk.choices[0].delta, 'content') and 
                        chunk.choices[0].delta.content):
                        yield chunk.choices[0].delta.content
            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                raise
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        logger.error(f"Error in stream endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
