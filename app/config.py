"""
Configuration settings for StudyFlow application.
Compatible with both FastAPI and Streamlit deployments.
"""

import os
import sys
from typing import List, Optional, Union
from pydantic_settings import BaseSettings
from pydantic import Field, validator

# Check if running in Streamlit
def is_streamlit_environment():
    """Check if code is running in Streamlit environment."""
    try:
        import streamlit as st
        return hasattr(st, 'secrets')
    except ImportError:
        return False

# Try to import streamlit for secrets
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    st = None


class Settings(BaseSettings):
    """Application settings with validation and Streamlit compatibility."""
    
    # App Info
    app_name: str = Field(default="StudyFlow", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    
    # AI API Configuration
    _openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    _openai_base_url: Optional[str] = Field(default=None, env="OPENAI_BASE_URL")
    _openai_model_name: Optional[str] = Field(default=None, env="OPENAI_MODEL_NAME")
    
    # File Upload Limits
    max_file_size_mb: int = Field(default=10, env="MAX_FILE_SIZE_MB")
    max_files_per_upload: int = Field(default=5, env="MAX_FILES_PER_UPLOAD")
    allowed_extensions: List[str] = Field(default=["pdf"], env="ALLOWED_EXTENSIONS")
    
    # Text Processing
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_tokens_per_request: int = Field(default=4000, env="MAX_TOKENS_PER_REQUEST")
    
    # Vector Database
    chroma_persist_directory: str = Field(default="/tmp/chromadb", env="CHROMA_PERSIST_DIRECTORY")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Security
    secret_key: str = Field(default="change-this-in-production", env="SECRET_KEY")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_burst: int = Field(default=10, env="RATE_LIMIT_BURST")
    
    # Additional settings for different deployment modes
    deployment_mode: str = Field(default="auto", env="DEPLOYMENT_MODE")  # auto, fastapi, streamlit
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=7860, env="PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._detect_deployment_mode()
    
    def _detect_deployment_mode(self):
        """Automatically detect deployment mode."""
        if self.deployment_mode == "auto":
            if STREAMLIT_AVAILABLE and is_streamlit_environment():
                self.deployment_mode = "streamlit"
            else:
                self.deployment_mode = "fastapi"
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from Streamlit secrets or environment."""
        # Try Streamlit secrets first
        if self.deployment_mode == "streamlit" and STREAMLIT_AVAILABLE:
            try:
                if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
                    return str(st.secrets['OPENAI_API_KEY'])
            except Exception:
                pass
        
        # Fall back to environment variable or pydantic field
        if self._openai_api_key:
            return self._openai_api_key
        
        # Try direct environment access
        env_key = os.getenv('OPENAI_API_KEY')
        if env_key:
            return env_key
        
        # Default/error case
        raise ValueError(
            "OPENAI_API_KEY not found. Please set it in environment variables or Streamlit secrets."
        )
    
    @property
    def openai_base_url(self) -> str:
        """Get OpenAI base URL from Streamlit secrets or environment."""
        # Try Streamlit secrets first
        if self.deployment_mode == "streamlit" and STREAMLIT_AVAILABLE:
            try:
                if hasattr(st, 'secrets') and 'OPENAI_BASE_URL' in st.secrets:
                    return str(st.secrets['OPENAI_BASE_URL'])
            except Exception:
                pass
        
        # Fall back to environment variable or pydantic field
        if self._openai_base_url:
            return self._openai_base_url
        
        # Try direct environment access
        env_url = os.getenv('OPENAI_BASE_URL')
        if env_url:
            return env_url
        
        # Default fallback
        return "https://tokenin.my.id/v1"
    
    @property
    def openai_model_name(self) -> str:
        """Get OpenAI model name from Streamlit secrets or environment."""
        # Try Streamlit secrets first
        if self.deployment_mode == "streamlit" and STREAMLIT_AVAILABLE:
            try:
                if hasattr(st, 'secrets') and 'OPENAI_MODEL_NAME' in st.secrets:
                    return str(st.secrets['OPENAI_MODEL_NAME'])
            except Exception:
                pass
        
        # Fall back to environment variable or pydantic field
        if self._openai_model_name:
            return self._openai_model_name
        
        # Try direct environment access
        env_model = os.getenv('OPENAI_MODEL_NAME')
        if env_model:
            return env_model
        
        # Default fallback
        return "myt/gemini-3.5-flash-free"
    
    @validator('allowed_extensions')
    def validate_extensions(cls, v):
        """Validate file extensions."""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(',')]
        return [ext.lower() for ext in v]
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator('chunk_size', 'chunk_overlap')
    def validate_chunk_settings(cls, v):
        """Validate chunk settings."""
        if v <= 0:
            raise ValueError("Chunk size and overlap must be positive integers")
        return v
    
    @validator('max_file_size_mb')
    def validate_file_size(cls, v):
        """Validate max file size."""
        if v <= 0 or v > 100:  # Reasonable limits
            raise ValueError("Max file size must be between 1-100 MB")
        return v
    
    @validator('max_files_per_upload')
    def validate_file_count(cls, v):
        """Validate max files per upload."""
        if v <= 0 or v > 20:  # Reasonable limits
            raise ValueError("Max files per upload must be between 1-20")
        return v
    
    def get_streamlit_secrets_status(self) -> dict:
        """Get status of Streamlit secrets (for debugging)."""
        if not STREAMLIT_AVAILABLE:
            return {"streamlit_available": False}
        
        try:
            secrets_available = hasattr(st, 'secrets')
            secrets_keys = list(st.secrets.keys()) if secrets_available else []
            
            return {
                "streamlit_available": True,
                "secrets_available": secrets_available,
                "secrets_keys": secrets_keys,
                "has_openai_key": "OPENAI_API_KEY" in secrets_keys,
                "has_openai_url": "OPENAI_BASE_URL" in secrets_keys,
                "has_openai_model": "OPENAI_MODEL_NAME" in secrets_keys,
            }
        except Exception as e:
            return {
                "streamlit_available": True,
                "error": str(e)
            }
    
    def get_config_summary(self) -> dict:
        """Get configuration summary (without sensitive data)."""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "deployment_mode": self.deployment_mode,
            "debug": self.debug,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_file_size_mb": self.max_file_size_mb,
            "max_files_per_upload": self.max_files_per_upload,
            "allowed_extensions": self.allowed_extensions,
            "embedding_model": self.embedding_model,
            "chroma_persist_directory": self.chroma_persist_directory,
            "openai_base_url": self.openai_base_url,
            "openai_model_name": self.openai_model_name,
            "has_api_key": bool(self._get_api_key_safely()),
            "log_level": self.log_level,
            "streamlit_status": self.get_streamlit_secrets_status()
        }
    
    def _get_api_key_safely(self) -> Optional[str]:
        """Safely get API key without exposing it."""
        try:
            key = self.openai_api_key
            return key[:8] + "..." if key else None
        except:
            return None
    
    def validate_required_settings(self) -> List[str]:
        """Validate that all required settings are present."""
        errors = []
        
        try:
            api_key = self.openai_api_key
            if not api_key or len(api_key.strip()) < 10:
                errors.append("OPENAI_API_KEY is missing or too short")
        except Exception as e:
            errors.append(f"OPENAI_API_KEY error: {str(e)}")
        
        if not self.openai_base_url:
            errors.append("OPENAI_BASE_URL is missing")
        
        if not self.openai_model_name:
            errors.append("OPENAI_MODEL_NAME is missing")
        
        # Validate directories
        try:
            os.makedirs(self.chroma_persist_directory, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create ChromaDB directory: {str(e)}")
        
        try:
            os.makedirs("/tmp/uploads", exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create uploads directory: {str(e)}")
        
        return errors
    
    def setup_directories(self) -> bool:
        """Set up required directories."""
        try:
            directories = [
                self.chroma_persist_directory,
                "/tmp/uploads",
                "/tmp/transformers_cache",
                "/tmp/sentence_transformers",
                "/tmp/huggingface"
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            
            return True
        except Exception as e:
            print(f"Failed to setup directories: {e}")
            return False
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return not self.debug and self.secret_key != "change-this-in-production"
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    def get_environment_info(self) -> dict:
        """Get environment information for debugging."""
        return {
            "python_version": sys.version,
            "deployment_mode": self.deployment_mode,
            "streamlit_available": STREAMLIT_AVAILABLE,
            "environment_variables": {
                "OPENAI_API_KEY": "***" if os.getenv("OPENAI_API_KEY") else "Not set",
                "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "Not set"),
                "OPENAI_MODEL_NAME": os.getenv("OPENAI_MODEL_NAME", "Not set"),
                "PORT": os.getenv("PORT", "Not set"),
                "DEBUG": os.getenv("DEBUG", "Not set"),
            },
            "directories": {
                "chroma_persist": self.chroma_persist_directory,
                "chroma_exists": os.path.exists(self.chroma_persist_directory),
                "uploads_exists": os.path.exists("/tmp/uploads"),
            }
        }


# Create global settings instance
def create_settings() -> Settings:
    """Create and validate settings instance."""
    try:
        settings = Settings()
        
        # Setup directories
        settings.setup_directories()
        
        # Validate required settings
        errors = settings.validate_required_settings()
        if errors:
            print("⚠️  Configuration warnings:")
            for error in errors:
                print(f"   - {error}")
        
        return settings
        
    except Exception as e:
        print(f"❌ Failed to create settings: {e}")
        # Create minimal settings for fallback
        return Settings()


# Global settings instance
settings = create_settings()

# Backward compatibility exports
APP_NAME = settings.app_name
APP_VERSION = settings.app_version
DEBUG = settings.debug
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
MAX_FILE_SIZE_MB = settings.max_file_size_mb
CHROMA_PERSIST_DIRECTORY = settings.chroma_persist_directory
EMBEDDING_MODEL = settings.embedding_model

# Helper function for debugging
def print_config_debug():
    """Print configuration debug information."""
    print("🔧 StudyFlow Configuration Debug")
    print("=" * 50)
    
    config_summary = settings.get_config_summary()
    for key, value in config_summary.items():
        if key == "streamlit_status":
            print(f"{key}:")
            for sk, sv in value.items():
                print(f"  {sk}: {sv}")
        else:
            print(f"{key}: {value}")
    
    print("=" * 50)
    
    validation_errors = settings.validate_required_settings()
    if validation_errors:
        print("❌ Validation Errors:")
        for error in validation_errors:
            print(f"   - {error}")
    else:
        print("✅ All required settings validated successfully!")


# Auto-setup for different environments
if __name__ == "__main__":
    print_config_debug()