"""
Chat endpoints for RAG-based question answering.
"""

import time
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from app.utils.logger import get_logger
from app.config import settings
from app.models.chat import ChatRequest, ChatResponse, SourceDocument, ChatError, MessageRole
from app.core.llm_client import LLMClient, LLMClientError
from app.core.embeddings import EmbeddingProcessor
from app.core.vectorstore import VectorStore, VectorStoreError

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])

# Initialize components
llm_client = LLMClient()
embedding_processor = EmbeddingProcessor()
vector_store = VectorStore()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest) -> ChatResponse:
    """
    Chat with documents using RAG (Retrieval-Augmented Generation).
    
    Args:
        request: Chat request with question and parameters
        
    Returns:
        AI-generated response with sources
    """
    start_time = time.time()
    
    try:
        logger.info(
            "Processing chat request",
            question_length=len(request.message),
            collection_id=request.collection_id,
            include_sources=request.include_sources
        )
        
        # Validate collection if specified
        relevant_docs = []
        if request.collection_id:
            try:
                # Check if collection exists
                collections = vector_store.list_collections()
                if request.collection_id not in collections:
                    raise HTTPException(
                        status_code=404,
                        detail=ChatError(
                            error=f"Collection '{request.collection_id}' not found",
                            error_code="COLLECTION_NOT_FOUND"
                        ).dict()
                    )
                
                # Generate query embedding
                query_embedding = embedding_processor.generate_query_embedding(
                    request.message
                )
                
                # Search for relevant documents
                search_results = vector_store.similarity_search(
                    collection_id=request.collection_id,
                    query_embedding=query_embedding,
                    n_results=5
                )
                
                relevant_docs = search_results
                
                logger.info(
                    "Document search completed",
                    results_found=len(relevant_docs),
                    collection_id=request.collection_id
                )
                
            except VectorStoreError as e:
                logger.error(f"Vector store error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=ChatError(
                        error="Failed to search documents",
                        details=str(e),
                        error_code="SEARCH_ERROR"
                    ).dict()
                )
        
        # Generate response using LLM
        try:
            llm_result = await llm_client.generate_answer(
                question=request.message,
                context_docs=relevant_docs,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            answer = llm_result["answer"]
            tokens_used = llm_result["tokens_used"]
            
        except LLMClientError as e:
            logger.error(f"LLM client error: {e}")
            raise HTTPException(
                status_code=500,
                detail=ChatError(
                    error=str(e),
                    error_code="LLM_ERROR"
                ).dict()
            )
        
        # Prepare source documents
        sources = []
        if request.include_sources and relevant_docs:
            for doc in relevant_docs:
                metadata = doc.get("metadata", {})
                source = SourceDocument(
                    filename=metadata.get("filename", "Unknown"),
                    page=metadata.get("page"),
                    chunk_id=doc.get("id", ""),
                    similarity_score=round(doc.get("similarity_score", 0.0), 3),
                    content_preview=doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
                )
                sources.append(source)
        
        response_time = time.time() - start_time
        
        # Prepare metadata
        metadata = {
            "model_used": llm_result.get("model_used", settings.openai_model_name),
            "context_docs_count": len(relevant_docs),
            "search_performed": bool(request.collection_id),
            "processing_steps": {
                "embedding_generation": bool(request.collection_id),
                "document_search": bool(request.collection_id and relevant_docs),
                "llm_generation": True
            }
        }
        
        logger.info(
            "Chat response generated successfully",
            tokens_used=tokens_used,
            response_time=round(response_time, 2),
            sources_count=len(sources)
        )
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            tokens_used=tokens_used,
            response_time=round(response_time, 2),
            collection_id=request.collection_id,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=ChatError(
                error="Internal server error during chat",
                details=str(e),
                error_code="INTERNAL_ERROR"
            ).dict()
        )


@router.post("/summarize")
async def summarize_collection(
    collection_id: str,
    max_length: int = Query(default=500, ge=100, le=2000, description="Maximum summary length")
) -> Dict[str, Any]:
    """
    Generate a summary of documents in a collection.
    
    Args:
        collection_id: Collection to summarize
        max_length: Maximum length of summary
        
    Returns:
        Collection summary
    """
    try:
        start_time = time.time()
        
        # Check if collection exists
        collections = vector_store.list_collections()
        if collection_id not in collections:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Collection '{collection_id}' not found"}
            )
        
        # Get collection stats
        stats = vector_store.get_collection_stats(collection_id)
        
        # Get a sample of documents for summarization
        # Use a broad query to get diverse content
        query_embedding = embedding_processor.generate_query_embedding(
            "overview summary main topics key points"
        )
        
        sample_docs = vector_store.similarity_search(
            collection_id=collection_id,
            query_embedding=query_embedding,
            n_results=10
        )
        
        if not sample_docs:
            return {
                "summary": "No documents found in collection to summarize.",
                "collection_id": collection_id,
                "document_count": stats.get("document_count", 0),
                "processing_time": time.time() - start_time
            }
        
        # Combine document content for summarization
        combined_text = "\n\n".join([
            doc.get("content", "") for doc in sample_docs
        ])
        
        # Generate summary
        summary = await llm_client.summarize_document(
            text=combined_text,
            max_length=max_length
        )
        
        processing_time = time.time() - start_time
        
        logger.info(
            "Collection summary generated",
            collection_id=collection_id,
            documents_sampled=len(sample_docs),
            summary_length=len(summary),
            processing_time=round(processing_time, 2)
        )
        
        return {
            "summary": summary,
            "collection_id": collection_id,
            "document_count": stats.get("document_count", 0),
            "documents_sampled": len(sample_docs),
            "processing_time": round(processing_time, 2),
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to generate summary",
                "details": str(e)
            }
        )


@router.post("/key-points")
async def extract_key_points(
    collection_id: str,
    num_points: int = Query(default=5, ge=3, le=10, description="Number of key points to extract")
) -> Dict[str, Any]:
    """
    Extract key points from documents in a collection.
    
    Args:
        collection_id: Collection to analyze
        num_points: Number of key points to extract
        
    Returns:
        List of key points
    """
    try:
        start_time = time.time()
        
        # Check if collection exists
        collections = vector_store.list_collections()
        if collection_id not in collections:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Collection '{collection_id}' not found"}
            )
        
        # Get collection stats
        stats = vector_store.get_collection_stats(collection_id)
        
        # Get diverse document sample
        query_embedding = embedding_processor.generate_query_embedding(
            "important key points main ideas concepts"
        )
        
        sample_docs = vector_store.similarity_search(
            collection_id=collection_id,
            query_embedding=query_embedding,
            n_results=8
        )
        
        if not sample_docs:
            return {
                "key_points": ["No documents found in collection."],
                "collection_id": collection_id,
                "document_count": stats.get("document_count", 0),
                "processing_time": time.time() - start_time
            }
        
        # Combine document content
        combined_text = "\n\n".join([
            doc.get("content", "") for doc in sample_docs
        ])
        
        # Extract key points
        key_points = await llm_client.extract_key_points(
            text=combined_text,
            num_points=num_points
        )
        
        processing_time = time.time() - start_time
        
        logger.info(
            "Key points extracted",
            collection_id=collection_id,
            documents_sampled=len(sample_docs),
            points_extracted=len(key_points),
            processing_time=round(processing_time, 2)
        )
        
        return {
            "key_points": key_points,
            "collection_id": collection_id,
            "document_count": stats.get("document_count", 0),
            "documents_sampled": len(sample_docs),
            "processing_time": round(processing_time, 2),
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Key point extraction error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to extract key points", 
                "details": str(e)
            }
        )


@router.get("/chat/history/{collection_id}")
async def get_chat_context(collection_id: str) -> Dict[str, Any]:
    """
    Get context information for a collection (for chat UI).
    
    Args:
        collection_id: Collection ID
        
    Returns:
        Collection context information
    """
    try:
        # Check if collection exists
        collections = vector_store.list_collections()
        if collection_id not in collections:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Collection '{collection_id}' not found"}
            )
        
        # Get collection stats
        stats = vector_store.get_collection_stats(collection_id)
        
        # Get sample documents to show available content
        query_embedding = embedding_processor.generate_query_embedding("overview")
        
        sample_docs = vector_store.similarity_search(
            collection_id=collection_id,
            query_embedding=query_embedding,
            n_results=3
        )
        
        # Get unique filenames
        filenames = set()
        for doc in sample_docs:
            filename = doc.get("metadata", {}).get("filename")
            if filename:
                filenames.add(filename)
        
        return {
            "collection_id": collection_id,
            "document_count": stats.get("document_count", 0),
            "created_at": stats.get("created_at"),
            "filenames": list(filenames),
            "sample_content": [
                {
                    "filename": doc.get("metadata", {}).get("filename", "Unknown"),
                    "page": doc.get("metadata", {}).get("page"),
                    "preview": doc.get("content", "")[:150] + "..." if len(doc.get("content", "")) > 150 else doc.get("content", "")
                }
                for doc in sample_docs[:2]
            ],
            "suggested_questions": [
                "What are the main topics covered in these documents?",
                "Can you summarize the key points?",
                "What are the most important concepts to understand?",
                f"Tell me about the content in {list(filenames)[0] if filenames else 'these documents'}"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat context: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve chat context",
                "details": str(e)
            }
        )