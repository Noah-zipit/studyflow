# 🎓 StudyFlow - AI Study Assistant

StudyFlow is an AI-powered PDF study assistant that helps you understand and learn from your documents using advanced Retrieval-Augmented Generation (RAG) technology.

## ✨ Features

- 📄 **PDF Processing**: Upload multiple PDF documents and extract text content
- 🤖 **AI-Powered Chat**: Ask questions about your documents and get intelligent answers
- 🎯 **Smart Summaries**: Generate concise summaries and extract key points
- 🔍 **Source Citations**: Get answers with references to specific pages and sections
- 🌐 **Web Interface**: Clean, modern interface that works on any device
- ⚡ **Fast & Efficient**: Built with FastAPI and optimized for performance

## 🚀 Quick Start

### Option 1: Hugging Face Spaces (Recommended)

1. Deploy directly to Hugging Face Spaces
2. Upload your environment variables
3. Your StudyFlow instance will be live at `https://your-username-studyflow.hf.space`

### Option 2: Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd studyflow
Set up environment

Bash

cp .env.example .env
# Edit .env with your API credentials
Install dependencies

Bash

pip install -r requirements.txt
Run the application

Bash

python -m uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
Open your browser
Navigate to http://localhost:7860

Option 3: Docker
Build and run with Docker
Bash

docker build -t studyflow .
docker run -p 7860:7860 studyflow
🔧 Configuration
Create a .env file with your configuration:

env

# AI API Configuration (Required)
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://tokenin.my.id/v1
OPENAI_MODEL_NAME=myt/gemini-3.5-flash-free

# Optional Configuration
MAX_FILE_SIZE_MB=10
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
📖 Usage
Upload Documents: Drag and drop or select PDF files to upload
Wait for Processing: Documents are automatically processed and indexed
Start Chatting: Ask questions about your documents
Use Quick Actions: Generate summaries or extract key points
View Sources: See which parts of your documents were used for answers
🏗️ Architecture
StudyFlow is built with:

Backend: FastAPI with async/await for high performance
AI Integration: OpenAI-compatible API for LLM interactions
Vector Database: ChromaDB for document embeddings and similarity search
Text Processing: LangChain for PDF processing and text chunking
Embeddings: Sentence Transformers for document vectorization
Frontend: Vanilla JavaScript with modern CSS for simplicity
📁 Project Structure
text

studyflow/
├── app/                     # FastAPI application
│   ├── api/                # API routes
│   ├── core/               # Business logic
│   ├── models/             # Pydantic models
│   └── utils/              # Utilities
├── static/                 # Frontend files
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript
│   └── assets/             # Images and icons
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
└── README.md              # This file
🐳 Deployment
Hugging Face Spaces
Create a new Space on Hugging Face
Choose "Docker" as the SDK
Upload all project files
Set environment variables in the Space settings
The app will automatically build and deploy
Docker Deployment
Bash

# Build the image
docker build -t studyflow .

# Run with environment file
docker run --env-file .env -p 7860:7860 studyflow
🔒 Security
All uploaded files are temporarily stored and processed securely
API keys are encrypted and stored safely
No data is permanently stored without user consent
Rate limiting and input validation protect against abuse
🛠️ Development
Setting up for Development
Install development dependencies

Bash

pip install -r requirements.txt
Run in development mode

Bash

uvicorn app.main:app --reload --host 0.0.0.0 --port 7860
Run tests

Bash

pytest tests/
API Documentation
When running locally, visit:

Interactive API docs: http://localhost:7860/docs
ReDoc documentation: http://localhost:7860/redoc
🧪 Testing
The application includes comprehensive health checks:

/api/health - Basic health status
/api/health/detailed - Component-specific health information
/api/health/ready - Kubernetes readiness probe
/api/health/live - Kubernetes liveness probe
🤝 Contributing
Fork the repository
Create a feature branch
Make your changes
Add tests for new functionality
Submit a pull request
📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

💡 Support
If you encounter any issues or have questions:

Check the API documentation when running locally
Review the application logs for error details
Open an issue on GitHub with detailed information
🙏 Acknowledgments
Built with FastAPI for the backend framework
ChromaDB for vector database functionality
LangChain for document processing
Sentence Transformers for embeddings
Hugging Face for hosting and deployment platform
Made with ❤️ for students and researchers worldwide