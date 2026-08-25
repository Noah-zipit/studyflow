"""
OpenAI-compatible LLM client for StudyFlow.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
import httpx

from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)


class LLMClientError(Exception):
    """Custom exception for LLM client operations."""
    pass


class LLMClient:
    """OpenAI-compatible LLM client with error handling and retries."""
    
    def __init__(self):
        self.model_name = settings.openai_model_name
        self.base_url = settings.openai_base_url
        self.api_key = settings.openai_api_key
        self.max_tokens = settings.max_tokens_per_request
        
        # Initialize async client
        self._client = None
        
    @property
    def client(self) -> AsyncOpenAI:
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=httpx.Timeout(60.0, read=120.0),  # Extended timeout for LLM responses
                    max_retries=3
                )
                logger.info(
                    "LLM client initialized",
                    base_url=self.base_url,
                    model=self.model_name
                )
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                raise LLMClientError(f"Cannot initialize LLM client: {e}")
                
        return self._client

    async def generate_answer(
        self,
        question: str,
        context_docs: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate answer using LLM with RAG context.
        
        Args:
            question: User's question
            context_docs: Retrieved context documents
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0.0 to 2.0)
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            start_time = time.time()
            
            # Prepare context from retrieved documents
            context = self._prepare_context(context_docs)
            
            # Build prompt
            prompt = self._build_rag_prompt(question, context)
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": self._get_system_prompt()
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # Call LLM
            logger.info(
                "Generating LLM response",
                model=self.model_name,
                context_docs=len(context_docs),
                question_length=len(question)
            )
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature,
                stream=False
            )
            
            # Extract response
            answer = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            response_time = time.time() - start_time
            
            logger.info(
                "LLM response generated successfully",
                tokens_used=tokens_used,
                response_time=round(response_time, 2),
                answer_length=len(answer)
            )
            
            return {
                "answer": answer,
                "tokens_used": tokens_used,
                "response_time": response_time,
                "model_used": self.model_name,
                "context_docs_count": len(context_docs)
            }
            
        except asyncio.TimeoutError:
            logger.error("LLM request timed out")
            raise LLMClientError("Request timed out. Please try again.")
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            
            # Handle specific error types
            if "rate limit" in str(e).lower():
                raise LLMClientError("Rate limit exceeded. Please wait a moment and try again.")
            elif "token" in str(e).lower() and "limit" in str(e).lower():
                raise LLMClientError("Request too long. Please try a shorter question or upload smaller documents.")
            elif "api key" in str(e).lower() or "unauthorized" in str(e).lower():
                raise LLMClientError("API authentication failed. Please check your configuration.")
            else:
                raise LLMClientError(f"Failed to generate response: {str(e)}")

    async def summarize_document(
        self,
        text: str,
        max_length: int = 500
    ) -> str:
        """
        Generate a summary of document text.
        
        Args:
            text: Document text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        try:
            # Truncate text if too long
            max_input_chars = 8000  # Conservative limit
            if len(text) > max_input_chars:
                text = text[:max_input_chars] + "..."
                
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that creates concise, informative summaries. Focus on key points and main ideas."
                },
                {
                    "role": "user",
                    "content": f"Please provide a concise summary (max {max_length} characters) of the following text:\n\n{text}"
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=min(max_length // 2, 500),  # Conservative token estimate
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            
            logger.info(
                "Document summary generated",
                input_length=len(text),
                summary_length=len(summary)
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Document summarization failed: {e}")
            return "Summary generation failed."

    async def extract_key_points(
        self,
        text: str,
        num_points: int = 5
    ) -> List[str]:
        """
        Extract key points from document text.
        
        Args:
            text: Document text
            num_points: Number of key points to extract
            
        Returns:
            List of key points
        """
        try:
            # Truncate text if too long
            max_input_chars = 8000
            if len(text) > max_input_chars:
                text = text[:max_input_chars] + "..."
                
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that extracts key points from documents. Return exactly the requested number of concise, important points."
                },
                {
                    "role": "user",
                    "content": f"Extract {num_points} key points from this text. Return only the points, one per line, numbered:\n\n{text}"
                }
            ]
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=400,
                temperature=0.2
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse numbered points
            points = []
            for line in content.split('\n'):
                line = line.strip()
                if line and any(line.startswith(f"{i}.") for i in range(1, num_points + 2)):
                    # Remove numbering
                    point = line.split('.', 1)[1].strip() if '.' in line else line
                    if point:
                        points.append(point)
            
            # Fallback if parsing fails
            if not points:
                points = [line.strip() for line in content.split('\n') if line.strip()][:num_points]
                
            logger.info(
                "Key points extracted",
                input_length=len(text),
                points_extracted=len(points)
            )
            
            return points[:num_points]
            
        except Exception as e:
            logger.error(f"Key point extraction failed: {e}")
            return ["Key point extraction failed."]

    def _prepare_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents."""
        if not context_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(context_docs[:5], 1):  # Limit to top 5 documents
            content = doc.get("content", "").strip()
            filename = doc.get("metadata", {}).get("filename", "Unknown")
            page = doc.get("metadata", {}).get("page")
            
            if content:
                page_info = f" (Page {page})" if page else ""
                context_parts.append(f"[Source {i} - {filename}{page_info}]:\n{content}")
        
        return "\n\n".join(context_parts)

    def _build_rag_prompt(self, question: str, context: str) -> str:
        """Build RAG prompt with question and context."""
        if not context:
            return f"""Please answer the following question based on your general knowledge:

Question: {question}

Please provide a helpful and accurate answer."""
        
        return f"""Please answer the following question based on the provided context. If the context doesn't contain enough information to answer the question completely, say so and provide what information you can.

Context:
{context}

Question: {question}

Answer:"""

    def _get_system_prompt(self) -> str:
        """Get system prompt for the assistant."""
        return """You are StudyFlow, an AI study assistant that helps users understand and learn from their documents. 

Guidelines:
- Provide clear, accurate, and helpful answers
- When answering based on provided context, cite the source documents
- If information is not in the context, clearly state this
- Break down complex topics into understandable explanations
- Use examples when helpful for learning
- Be encouraging and supportive in your responses
- If you're uncertain about something, express that uncertainty

Your goal is to help users learn effectively from their study materials."""

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on LLM client.
        
        Returns:
            Health status information
        """
        try:
            start_time = time.time()
            
            # Simple test request
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0
            )
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "model": self.model_name,
                "base_url": self.base_url,
                "response_time": round(response_time, 3),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model_name,
                "base_url": self.base_url,
                "timestamp": time.time()
            }