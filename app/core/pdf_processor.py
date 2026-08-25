"""
PDF text extraction and processing utilities.
"""

import os
import time
import filetype
from typing import List, Tuple, Dict
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


class PDFProcessor:
    """Handles PDF text extraction and validation."""
    
    def __init__(self):
        self.max_file_size = settings.max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.allowed_extensions = settings.allowed_extensions
        
    def validate_file(self, file_path: str, original_filename: str) -> None:
        """
        Validate uploaded file.
        
        Args:
            file_path: Path to uploaded file
            original_filename: Original filename from upload
            
        Raises:
            PDFProcessingError: If file validation fails
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                raise PDFProcessingError(f"File not found: {file_path}")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                raise PDFProcessingError(
                    f"File too large: {file_size / (1024*1024):.1f}MB "
                    f"(max: {settings.max_file_size_mb}MB)"
                )
            
            if file_size == 0:
                raise PDFProcessingError("File is empty")
            
            # Check file extension
            extension = Path(original_filename).suffix.lower().lstrip('.')
            if extension not in self.allowed_extensions:
                raise PDFProcessingError(
                    f"Unsupported file type: {extension}. "
                    f"Allowed: {', '.join(self.allowed_extensions)}"
                )
            
            # Validate file type by content
            kind = filetype.guess(file_path)
            if not kind or kind.extension != 'pdf':
                raise PDFProcessingError(
                    "File content is not a valid PDF"
                )
                
            logger.info(
                "File validation successful",
                filename=original_filename,
                size_mb=round(file_size / (1024*1024), 2)
            )
            
        except PDFProcessingError:
            raise
        except Exception as e:
            logger.error("File validation error", error=str(e), filename=original_filename)
            raise PDFProcessingError(f"File validation failed: {str(e)}")

    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, int]:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, page_count)
            
        Raises:
            PDFProcessingError: If text extraction fails
        """
        try:
            start_time = time.time()
            
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                if page_count == 0:
                    raise PDFProcessingError("PDF has no pages")
                
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            # Add page marker for source tracking
                            text_content.append(f"\n--- Page {page_num} ---\n{page_text}")
                        else:
                            logger.warning(f"No text found on page {page_num}")
                            
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract text from page {page_num}",
                            error=str(e)
                        )
                        continue
                
                full_text = "\n".join(text_content)
                
                if not full_text.strip():
                    raise PDFProcessingError("No text content found in PDF")
                
                processing_time = time.time() - start_time
                logger.info(
                    "PDF text extraction completed",
                    page_count=page_count,
                    text_length=len(full_text),
                    processing_time=round(processing_time, 2)
                )
                
                return full_text, page_count
                
        except PDFProcessingError:
            raise
        except Exception as e:
            logger.error("PDF text extraction failed", error=str(e))
            raise PDFProcessingError(f"Failed to extract text from PDF: {str(e)}")

    async def process_pdf_async(self, file_path: str, original_filename: str) -> Dict:
        """
        Asynchronously process a PDF file.
        
        Args:
            file_path: Path to PDF file
            original_filename: Original filename
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # Run CPU-intensive work in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                # Validate file
                await loop.run_in_executor(
                    executor, self.validate_file, file_path, original_filename
                )
                
                # Extract text
                text, page_count = await loop.run_in_executor(
                    executor, self.extract_text_from_pdf, file_path
                )
            
            processing_time = time.time() - start_time
            file_size = os.path.getsize(file_path)
            
            result = {
                "filename": original_filename,
                "text": text,
                "page_count": page_count,
                "file_size": file_size,
                "processing_time": processing_time,
                "success": True
            }
            
            logger.info(
                "PDF processing completed successfully",
                filename=original_filename,
                page_count=page_count,
                processing_time=round(processing_time, 2)
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            logger.error(
                "PDF processing failed", 
                filename=original_filename,
                error=error_msg,
                processing_time=round(processing_time, 2)
            )
            
            return {
                "filename": original_filename,
                "error": error_msg,
                "processing_time": processing_time,
                "success": False
            }
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {file_path}: {e}")

    def extract_page_number_from_text(self, text: str) -> int:
        """
        Extract page number from text chunk with page markers.
        
        Args:
            text: Text chunk that may contain page markers
            
        Returns:
            Page number or 1 if not found
        """
        try:
            import re
            # Look for page markers added during extraction
            match = re.search(r'--- Page (\d+) ---', text)
            if match:
                return int(match.group(1))
            return 1
        except Exception:
            return 1