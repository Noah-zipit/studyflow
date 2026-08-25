"""
ChromaDB vector store operations.
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    pass


class VectorStore:
    """ChromaDB vector store manager."""
    
    def __init__(self):
        self.persist_directory = settings.chroma_persist_directory
        self._client = None
        
    @property
    def client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            try:
                logger.info("Initializing ChromaDB client")
                
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )
                
                logger.info(
                    "ChromaDB client initialized successfully",
                    persist_directory=self.persist_directory
                )
                
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {e}")
                raise VectorStoreError(f"Cannot initialize ChromaDB: {e}")
                
        return self._client

    def create_collection(self, collection_name: str = None) -> str:
        """
        Create a new document collection.
        
        Args:
            collection_name: Optional collection name, auto-generated if None
            
        Returns:
            Collection ID
        """
        try:
            if collection_name is None:
                collection_name = f"documents_{uuid.uuid4().hex[:8]}"
            
            # Ensure collection name is valid
            collection_name = self._sanitize_collection_name(collection_name)
            
            logger.info(f"Creating ChromaDB collection: {collection_name}")
            
            # Try to get existing collection first
            try:
                collection = self.client.get_collection(name=collection_name)
                logger.info(f"Collection {collection_name} already exists, using existing")
                return collection_name
            except Exception:
                # Collection doesn't exist, create new one
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"created_at": time.time()}
                )
                
                logger.info(f"Created new collection: {collection_name}")
                return collection_name
                
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise VectorStoreError(f"Cannot create collection: {e}")

    def add_documents(
        self, 
        collection_id: str, 
        chunks: List[Dict[str, Any]], 
        embeddings: List[List[float]]
    ) -> int:
        """
        Add document chunks to collection.
        
        Args:
            collection_id: Target collection ID
            chunks: List of text chunks with metadata
            embeddings: Corresponding embedding vectors
            
        Returns:
            Number of documents added
        """
        try:
            start_time = time.time()
            
            if len(chunks) != len(embeddings):
                raise VectorStoreError("Chunks and embeddings count mismatch")
            
            if not chunks:
                return 0
            
            collection = self.client.get_collection(name=collection_id)
            
            # Prepare data for ChromaDB
            ids = [chunk["id"] for chunk in chunks]
            documents = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Add documents to collection
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            processing_time = time.time() - start_time
            
            logger.info(
                "Documents added to collection",
                collection_id=collection_id,
                document_count=len(chunks),
                processing_time=round(processing_time, 2)
            )
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Failed to add documents to collection {collection_id}: {e}")
            raise VectorStoreError(f"Cannot add documents: {e}")

    def similarity_search(
        self, 
        collection_id: str, 
        query_embedding: List[float], 
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            collection_id: Collection to search
            query_embedding: Query embedding vector
            n_results: Number of results to return
            
        Returns:
            List of similar documents with metadata
        """
        try:
            start_time = time.time()
            
            collection = self.client.get_collection(name=collection_id)
            
            # Perform similarity search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, 20),  # Limit max results
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    result = {
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                    }
                    formatted_results.append(result)
            
            search_time = time.time() - start_time
            
            logger.info(
                "Similarity search completed",
                collection_id=collection_id,
                results_found=len(formatted_results),
                search_time=round(search_time, 3)
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Similarity search failed for collection {collection_id}: {e}")
            raise VectorStoreError(f"Cannot perform similarity search: {e}")

    def get_collection_stats(self, collection_id: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_id: Collection ID
            
        Returns:
            Collection statistics
        """
        try:
            collection = self.client.get_collection(name=collection_id)
            
            # Get collection info
            count = collection.count()
            metadata = collection.metadata
            
            stats = {
                "collection_id": collection_id,
                "document_count": count,
                "created_at": metadata.get("created_at"),
                "last_updated": time.time()
            }
            
            logger.debug(f"Collection stats retrieved for {collection_id}", stats=stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats for {collection_id}: {e}")
            raise VectorStoreError(f"Cannot get collection stats: {e}")

    def list_collections(self) -> List[str]:
        """
        List all available collections.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]
            
            logger.debug(f"Listed {len(collection_names)} collections")
            
            return collection_names
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise VectorStoreError(f"Cannot list collections: {e}")

    def delete_collection(self, collection_id: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_id: Collection to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=collection_id)
            logger.info(f"Collection deleted: {collection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_id}: {e}")
            raise VectorStoreError(f"Cannot delete collection: {e}")

    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitize collection name for ChromaDB."""
        # ChromaDB collection names must be alphanumeric + underscores/hyphens
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        sanitized = sanitized.strip('_-')
        
        # Ensure it starts with letter or number
        if sanitized and not sanitized[0].isalnum():
            sanitized = f"col_{sanitized}"
        
        # Fallback if empty
        if not sanitized:
            sanitized = f"collection_{uuid.uuid4().hex[:8]}"
            
        return sanitized[:50]  # Limit length