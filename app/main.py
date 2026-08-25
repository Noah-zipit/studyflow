"""
StudyFlow - AI-powered PDF study assistant with RAG capabilities.
Main FastAPI application.
"""

import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from app.config import settings
from app.utils.logger import configure_logging, get_logger
from app.api import health_router, upload_router, chat_router

# Configure logging
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(
        "Starting StudyFlow application",
        app_name=settings.app_name,
        version=settings.app_version,
        debug=settings.debug
    )
    
    # Ensure required directories exist
    os.makedirs(settings.chroma_persist_directory, exist_ok=True)
    os.makedirs("/tmp/uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Pre-warm embedding model (optional)
    try:
        from app.core.embeddings import EmbeddingProcessor
        embedding_processor = EmbeddingProcessor()
        # This will lazy-load the model on first use
        logger.info("Embedding processor initialized")
    except Exception as e:
        logger.warning(f"Embedding model pre-warming failed: {e}")
    
    logger.info("StudyFlow application startup complete")
    
    yield
    
    # Shutdown
    logger.info("StudyFlow application shutdown")


# Create FastAPI app
app = FastAPI(
    title="StudyFlow",
    description="AI-powered PDF study assistant with RAG capabilities",
    version=settings.app_version,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log HTTP requests with timing."""
    start_time = time.time()
    
    # Log request
    logger.info(
        "HTTP request started",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown"
    )
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        "HTTP request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=round(process_time, 3)
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        url=str(request.url),
        method=request.method,
        error=str(exc),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": time.time()
        }
    )


# Include API routers
app.include_router(health_router)
app.include_router(upload_router)
app.include_router(chat_router)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        # Fallback HTML if static files not found
        return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>StudyFlow - AI Study Assistant</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        h1 { color: #333; }
        .status { color: #28a745; font-weight: bold; }
        .links { margin-top: 30px; }
        .links a { 
            display: inline-block;
            margin: 0 10px;
            padding: 10px 20px;
            background: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }
        .links a:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎓 StudyFlow</h1>
        <p>AI-powered PDF study assistant with RAG capabilities</p>
        <p class="status">✅ Application is running</p>
        <div class="links">
            <a href="/docs">API Documentation</a>
            <a href="/api/health">Health Check</a>
        </div>
    </div>
</body>
</html>
        """)


@app.get("/favicon.ico")
async def favicon():
    """Serve favicon."""
    return FileResponse("static/assets/logo.svg", media_type="image/svg+xml")


# Additional utility endpoints
@app.get("/info")
async def app_info():
    """Get application information."""
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "debug": settings.debug,
        "timestamp": time.time(),
        "endpoints": {
            "health": "/api/health",
            "upload": "/api/upload",
            "chat": "/api/chat",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=settings.debug,
        log_level="info"
    )