"""
Document embedding and text chunking utilities.
"""

import time
import hashlib
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)


class EmbeddingProcessor:
    """Handles text chunking and embedding generation."""
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.embedding_model_name = settings.embedding_model
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True
        )
        
        # Initialize embeddings model (lazy loading)
        self._embeddings = None
        
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy load embeddings model."""
        if self._embeddings is None:
            try:
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                start_time = time.time()
                
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={
                        'device': 'cpu',  # Force CPU for HF Spaces compatibility
                        'trust_remote_code': False
                    },
                    encode_kwargs={
                        'normalize_embeddings': True,
                        'batch_size': 32
                    }
                )
                
                load_time = time.time() - start_time
                logger.info(
                    "Embedding model loaded successfully",
                    model=self.embedding_model_name,
                    load_time=round(load_time, 2)
                )
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise RuntimeError(f"Cannot initialize embedding model: {e}")
                
        return self._embeddings

    def chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text content to chunk
            filename: Source filename for metadata
            
        Returns:
            List of text chunks with metadata
        """
        try:
            start_time = time.time()
            
            # Create chunks
            chunks = self.text_splitter.create_documents([text])
            
            # Process chunks and add metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                # Generate unique chunk ID
                chunk_content = chunk.page_content
                chunk_id = self._generate_chunk_id(chunk_content, filename, i)
                
                # Extract page number from content if available
                page_number = self._extract_page_number(chunk_content)
                
                # Create chunk metadata
                chunk_data = {
                    "id": chunk_id,
                    "content": chunk_content,
                    "metadata": {
                        "filename": filename,
                        "chunk_index": i,
                        "page": page_number,
                        "char_count": len(chunk_content),
                        "start_index": chunk.metadata.get("start_index", 0)
                    }
                }
                
                processed_chunks.append(chunk_data)
            
            processing_time = time.time() - start_time
            
            logger.info(
                "Text chunking completed",
                filename=filename,
                total_chunks=len(processed_chunks),
                avg_chunk_size=sum(len(c["content"]) for c in processed_chunks) // len(processed_chunks),
                processing_time=round(processing_time, 2)
            )
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Text chunking failed for {filename}: {e}")
            raise RuntimeError(f"Failed to chunk text: {e}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text chunks.
        
        Args:
            texts: List of text chunks
            
        Returns:
            List of embedding vectors
        """
        try:
            start_time = time.time()
            
            if not texts:
                return []
            
            logger.info(f"Generating embeddings for {len(texts)} text chunks")
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            processing_time = time.time() - start_time
            
            logger.info(
                "Embedding generation completed",
                chunk_count=len(texts),
                embedding_dim=len(embeddings[0]) if embeddings else 0,
                processing_time=round(processing_time, 2)
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector for the query
        """
        try:
            start_time = time.time()
            
            embedding = self.embeddings.embed_query(query)
            
            processing_time = time.time() - start_time
            
            logger.debug(
                "Query embedding generated",
                query_length=len(query),
                embedding_dim=len(embedding),
                processing_time=round(processing_time, 4)
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate query embedding: {e}")

    def _generate_chunk_id(self, content: str, filename: str, index: int) -> str:
        """Generate unique ID for a text chunk."""
        # Create hash from content + filename + index for uniqueness
        content_hash = hashlib.md5(
            f"{filename}:{index}:{content[:100]}".encode()
        ).hexdigest()[:12]
        
        return f"chunk_{content_hash}_{index}"

    def _extract_page_number(self, content: str) -> int:
        """Extract page number from chunk content."""
        try:
            import re
            # Look for page markers added during PDF processing
            match = re.search(r'--- Page (\d+) ---', content)
            if match:
                return int(match.group(1))
            return 1
        except Exception:
            return 1

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            "model_name": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_dim": 384,  # all-MiniLM-L6-v2 dimension
            "model_loaded": self._embeddings is not None
        }