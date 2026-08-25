"""
PDF upload and processing endpoints.
"""

import os
import time
import asyncio
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.utils.logger import get_logger
from app.config import settings
from app.models.upload import UploadResponse, DocumentInfo, ProcessingStatus, UploadError
from app.core.pdf_processor import PDFProcessor, PDFProcessingError
from app.core.embeddings import EmbeddingProcessor
from app.core.vectorstore import VectorStore, VectorStoreError

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

# Initialize processors
pdf_processor = PDFProcessor()
embedding_processor = EmbeddingProcessor()
vector_store = VectorStore()


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
) -> UploadResponse:
    """
    Upload and process PDF documents.
    
    Args:
        files: List of PDF files to upload
        background_tasks: Background task queue
        
    Returns:
        Processing results with document information
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not files:
            raise HTTPException(
                status_code=400,
                detail=UploadError(error="No files provided").dict()
            )
        
        if len(files) > settings.max_files_per_upload:
            raise HTTPException(
                status_code=400,
                detail=UploadError(
                    error=f"Too many files. Maximum allowed: {settings.max_files_per_upload}"
                ).dict()
            )
        
        # Validate file types and sizes
        for file in files:
            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail=UploadError(error="Invalid filename").dict()
                )
            
            # Check file extension
            extension = file.filename.split('.')[-1].lower()
            if extension not in settings.allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=UploadError(
                        error=f"Unsupported file type: {extension}",
                        filename=file.filename
                    ).dict()
                )
        
        logger.info(f"Processing {len(files)} uploaded files")
        
        # Create collection for this upload session
        collection_id = vector_store.create_collection()
        
        # Process files concurrently
        processing_tasks = []
        for file in files:
            task = process_single_file(file, collection_id)
            processing_tasks.append(task)
        
        # Wait for all files to process
        results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Aggregate results
        successful_documents = []
        failed_documents = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"File processing exception: {result}")
                failed_documents.append({
                    "filename": "unknown",
                    "error": str(result)
                })
            elif result.get("success", False):
                successful_documents.append(result)
            else:
                failed_documents.append(result)
        
        # Prepare response
        total_processing_time = time.time() - start_time
        
        if successful_documents:
            # Build document info list
            document_infos = []
            for doc in successful_documents:
                doc_info = DocumentInfo(
                    filename=doc["filename"],
                    file_size=doc["file_size"],
                    page_count=doc["page_count"],
                    chunk_count=doc["chunk_count"],
                    processing_time=doc["processing_time"]
                )
                document_infos.append(doc_info)
            
            status = ProcessingStatus.COMPLETED
            message = f"Successfully processed {len(successful_documents)} documents"
            
            if failed_documents:
                message += f" ({len(failed_documents)} failed)"
                status = ProcessingStatus.COMPLETED  # Partial success still counts as completed
            
            logger.info(
                "Document upload completed",
                successful=len(successful_documents),
                failed=len(failed_documents),
                collection_id=collection_id,
                total_time=round(total_processing_time, 2)
            )
            
            return UploadResponse(
                status=status,
                message=message,
                document_count=len(successful_documents),
                documents=document_infos,
                collection_id=collection_id
            )
        
        else:
            # All files failed
            error_messages = [doc.get("error", "Unknown error") for doc in failed_documents]
            error_summary = "; ".join(error_messages[:3])  # First 3 errors
            if len(error_messages) > 3:
                error_summary += f" ... and {len(error_messages) - 3} more"
            
            logger.error(
                "All documents failed processing",
                failed_count=len(failed_documents),
                errors=error_messages
            )
            
            return UploadResponse(
                status=ProcessingStatus.FAILED,
                message="All documents failed to process",
                document_count=0,
                documents=[],
                error=error_summary
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=UploadError(
                error="Internal server error during upload",
                details=str(e)
            ).dict()
        )


async def process_single_file(file: UploadFile, collection_id: str) -> Dict[str, Any]:
    """
    Process a single uploaded PDF file.
    
    Args:
        file: Uploaded file object
        collection_id: Target collection ID
        
    Returns:
        Processing result dictionary
    """
    temp_file_path = None
    
    try:
        # Save file to temporary location
        temp_file_path = f"/tmp/{file.filename}_{int(time.time())}"
        
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Process PDF
        pdf_result = await pdf_processor.process_pdf_async(
            temp_file_path, 
            file.filename
        )
        
        if not pdf_result["success"]:
            return pdf_result
        
        # Generate text chunks
        chunks = embedding_processor.chunk_text(
            pdf_result["text"], 
            file.filename
        )
        
        if not chunks:
            return {
                "filename": file.filename,
                "error": "No text chunks generated",
                "success": False
            }
        
        # Generate embeddings
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = embedding_processor.generate_embeddings(chunk_texts)
        
        # Store in vector database
        chunks_added = vector_store.add_documents(
            collection_id, 
            chunks, 
            embeddings
        )
        
        # Return success result
        return {
            "filename": file.filename,
            "file_size": pdf_result["file_size"],
            "page_count": pdf_result["page_count"],
            "chunk_count": chunks_added,
            "processing_time": pdf_result["processing_time"],
            "success": True
        }
        
    except PDFProcessingError as e:
        logger.error(f"PDF processing error for {file.filename}: {e}")
        return {
            "filename": file.filename,
            "error": str(e),
            "success": False
        }
    except VectorStoreError as e:
        logger.error(f"Vector store error for {file.filename}: {e}")
        return {
            "filename": file.filename,
            "error": f"Database error: {str(e)}",
            "success": False
        }
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {e}")
        return {
            "filename": file.filename,
            "error": f"Processing failed: {str(e)}",
            "success": False
        }
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file_path}: {e}")


@router.get("/collections")
async def list_collections() -> Dict[str, Any]:
    """
    List available document collections.
    
    Returns:
        List of collection information
    """
    try:
        collections = vector_store.list_collections()
        
        collection_details = []
        for collection_id in collections:
            try:
                stats = vector_store.get_collection_stats(collection_id)
                collection_details.append(stats)
            except Exception as e:
                logger.warning(f"Failed to get stats for collection {collection_id}: {e}")
                collection_details.append({
                    "collection_id": collection_id,
                    "document_count": 0,
                    "error": str(e)
                })
        
        return {
            "collections": collection_details,
            "total_collections": len(collections),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve collections",
                "details": str(e)
            }
        )


@router.delete("/collections/{collection_id}")
async def delete_collection(collection_id: str) -> Dict[str, Any]:
    """
    Delete a document collection.
    
    Args:
        collection_id: Collection to delete
        
    Returns:
        Deletion confirmation
    """
    try:
        success = vector_store.delete_collection(collection_id)
        
        if success:
            return {
                "message": f"Collection {collection_id} deleted successfully",
                "collection_id": collection_id,
                "timestamp": time.time()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Collection {collection_id} not found",
                    "collection_id": collection_id
                }
            )
            
    except VectorStoreError as e:
        logger.error(f"Failed to delete collection {collection_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Failed to delete collection: {str(e)}",
                "collection_id": collection_id
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting collection {collection_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "details": str(e),
                "collection_id": collection_id
            }
        )