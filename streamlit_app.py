"""
StudyFlow - Streamlit Version
AI-powered PDF study assistant with RAG capabilities.
"""

import streamlit as st
import asyncio
import tempfile
import os
import time
import traceback
from typing import List, Dict, Any

# Configure page
st.set_page_config(
    page_title="StudyFlow - AI Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import your existing core modules
try:
    from app.core.pdf_processor import PDFProcessor, PDFProcessingError
    from app.core.embeddings import EmbeddingProcessor
    from app.core.vectorstore import VectorStore, VectorStoreError
    from app.core.llm_client import LLMClient, LLMClientError
    from app.config import Settings
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

# Initialize settings with Streamlit secrets
class StreamlitSettings:
    def __init__(self):
        # Load from Streamlit secrets
        self.app_name = "StudyFlow"
        self.app_version = "1.0.0"
        self.debug = False
        
        # AI API Configuration from Streamlit secrets
        self.openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
        self.openai_base_url = st.secrets.get("OPENAI_BASE_URL", "https://tokenin.my.id/v1")
        self.openai_model_name = st.secrets.get("OPENAI_MODEL_NAME", "myt/gemini-3.5-flash-free")
        
        # File processing settings
        self.max_file_size_mb = 10
        self.max_files_per_upload = 5
        self.allowed_extensions = ["pdf"]
        
        # Text processing
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.max_tokens_per_request = 4000
        
        # Vector database (use Streamlit's temp directory)
        self.chroma_persist_directory = "/tmp/chromadb"
        self.embedding_model = "all-MiniLM-L6-v2"

# Initialize settings
settings = StreamlitSettings()

# Ensure required directories exist
os.makedirs("/tmp/chromadb", exist_ok=True)
os.makedirs("/tmp/uploads", exist_ok=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []
    st.session_state.collection_id = None
    st.session_state.uploaded_documents = []
    st.session_state.processing_status = None

# Cache expensive operations
@st.cache_resource
def get_processors():
    """Initialize and cache all processors."""
    try:
        pdf_processor = PDFProcessor()
        embedding_processor = EmbeddingProcessor()
        vector_store = VectorStore()
        llm_client = LLMClient()
        
        return pdf_processor, embedding_processor, vector_store, llm_client
    except Exception as e:
        st.error(f"Failed to initialize processors: {e}")
        return None, None, None, None

# Get processors
pdf_proc, embed_proc, vector_store, llm_client = get_processors()

def show_header():
    """Display app header."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); color: white; margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 1rem 1rem;">
        <h1 style="margin: 0; font-size: 2.5rem;">🎓 StudyFlow</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">AI-powered PDF study assistant</p>
    </div>
    """, unsafe_allow_html=True)

def handle_file_upload():
    """Handle PDF file uploads in sidebar."""
    st.sidebar.header("📚 Upload Documents")
    
    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help=f"Upload up to {settings.max_files_per_upload} PDF files (max {settings.max_file_size_mb}MB each)",
        key="file_uploader"
    )
    
    if uploaded_files:
        st.sidebar.success(f"📄 {len(uploaded_files)} files selected")
        
        # Validate files
        valid_files = []
        for file in uploaded_files:
            if file.size > settings.max_file_size_mb * 1024 * 1024:
                st.sidebar.error(f"❌ {file.name} is too large (max {settings.max_file_size_mb}MB)")
            else:
                valid_files.append(file)
        
        if valid_files:
            if st.sidebar.button("🚀 Process Documents", type="primary"):
                process_documents(valid_files)
    
    # Show current collection info
    if st.session_state.collection_id:
        st.sidebar.success(f"✅ Documents loaded")
        st.sidebar.info(f"Collection: {st.session_state.collection_id[:8]}...")
        
        if st.session_state.uploaded_documents:
            st.sidebar.write("📋 **Uploaded Documents:**")
            for doc in st.session_state.uploaded_documents:
                st.sidebar.write(f"• {doc['filename']} ({doc['pages']} pages)")
        
        if st.sidebar.button("🗑️ Clear Documents"):
            clear_documents()

def process_documents(uploaded_files):
    """Process uploaded PDF files."""
    if not all([pdf_proc, embed_proc, vector_store, llm_client]):
        st.error("❌ Processors not initialized properly")
        return
    
    try:
        # Create progress indicators
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        status_text.text("🔄 Creating collection...")
        progress_bar.progress(10)
        
        # Create collection
        collection_id = vector_store.create_collection()
        st.session_state.collection_id = collection_id
        
        processed_docs = []
        total_files = len(uploaded_files)
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                status_text.text(f"📄 Processing {uploaded_file.name}...")
                progress = 20 + (i * 60 // total_files)
                progress_bar.progress(progress)
                
                # Save temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Process PDF
                    result = asyncio.run(pdf_proc.process_pdf_async(tmp_path, uploaded_file.name))
                    
                    if result["success"]:
                        # Generate chunks and embeddings
                        status_text.text(f"🧩 Chunking {uploaded_file.name}...")
                        chunks = embed_proc.chunk_text(result["text"], uploaded_file.name)
                        
                        status_text.text(f"🔮 Generating embeddings for {uploaded_file.name}...")
                        chunk_texts = [chunk["content"] for chunk in chunks]
                        embeddings = embed_proc.generate_embeddings(chunk_texts)
                        
                        # Store in vector database
                        status_text.text(f"💾 Storing {uploaded_file.name}...")
                        vector_store.add_documents(collection_id, chunks, embeddings)
                        
                        processed_docs.append({
                            "filename": uploaded_file.name,
                            "pages": result["page_count"],
                            "chunks": len(chunks),
                            "size": uploaded_file.size
                        })
                        
                    else:
                        st.sidebar.error(f"❌ Failed to process {uploaded_file.name}: {result.get('error', 'Unknown error')}")
                
                finally:
                    # Clean up temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        
            except Exception as e:
                st.sidebar.error(f"❌ Error processing {uploaded_file.name}: {e}")
                continue
        
        # Final progress
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # Store processed documents info
        st.session_state.uploaded_documents = processed_docs
        
        if processed_docs:
            st.sidebar.success(f"🎉 Successfully processed {len(processed_docs)} documents!")
            
            # Add welcome message
            welcome_msg = f"📚 I've processed {len(processed_docs)} documents for you! Here's what I found:\n\n"
            for doc in processed_docs:
                welcome_msg += f"• **{doc['filename']}** - {doc['pages']} pages, {doc['chunks']} chunks\n"
            welcome_msg += "\nWhat would you like to know about these documents?"
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": welcome_msg
            })
        
        # Clear progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.sidebar.error(f"❌ Processing failed: {e}")
        st.error(f"Detailed error: {traceback.format_exc()}")

def clear_documents():
    """Clear current document collection."""
    try:
        if st.session_state.collection_id:
            vector_store.delete_collection(st.session_state.collection_id)
        
        st.session_state.collection_id = None
        st.session_state.uploaded_documents = []
        st.session_state.messages = []
        st.sidebar.success("🗑️ Documents cleared!")
        st.rerun()
        
    except Exception as e:
        st.sidebar.error(f"❌ Failed to clear documents: {e}")

def show_chat_interface():
    """Display chat interface."""
    st.header("💬 Chat with Your Documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if "sources" in message:
                with st.expander("📖 Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}:** {source['filename']}")
                        if source.get('page'):
                            st.write(f"📄 Page {source['page']}")
                        st.write(f"🔍 Similarity: {source['similarity_score']:.3f}")
                        st.write(f"_{source['content_preview']}_")
                        if i < len(message["sources"]):
                            st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            if st.session_state.collection_id:
                try:
                    with st.spinner("🤔 Thinking..."):
                        # Generate response
                        response_data = generate_ai_response(prompt)
                        
                        if response_data:
                            st.markdown(response_data["answer"])
                            
                            # Store message with sources
                            message_data = {
                                "role": "assistant", 
                                "content": response_data["answer"]
                            }
                            
                            if response_data.get("sources"):
                                message_data["sources"] = response_data["sources"]
                                
                                # Show sources immediately
                                with st.expander("📖 Sources", expanded=False):
                                    for i, source in enumerate(response_data["sources"], 1):
                                        st.write(f"**Source {i}:** {source['filename']}")
                                        if source.get('page'):
                                            st.write(f"📄 Page {source['page']}")
                                        st.write(f"🔍 Similarity: {source['similarity_score']:.3f}")
                                        st.write(f"_{source['content_preview']}_")
                                        if i < len(response_data["sources"]):
                                            st.divider()
                            
                            st.session_state.messages.append(message_data)
                        else:
                            error_msg = "❌ Sorry, I couldn't generate a response. Please try again."
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                no_docs_msg = "📚 Please upload and process some documents first!"
                st.warning(no_docs_msg)
                st.session_state.messages.append({"role": "assistant", "content": no_docs_msg})

def generate_ai_response(question: str) -> Dict[str, Any]:
    """Generate AI response for user question."""
    try:
        if not all([embed_proc, vector_store, llm_client]):
            return None
        
        # Generate query embedding
        query_embedding = embed_proc.generate_query_embedding(question)
        
        # Search similar documents
        relevant_docs = vector_store.similarity_search(
            collection_id=st.session_state.collection_id,
            query_embedding=query_embedding,
            n_results=5
        )
        
        # Generate LLM response
        llm_response = asyncio.run(llm_client.generate_answer(
            question=question,
            context_docs=relevant_docs,
            max_tokens=1000,
            temperature=0.7
        ))
        
        # Format sources
        sources = []
        for doc in relevant_docs:
            metadata = doc.get("metadata", {})
            sources.append({
                "filename": metadata.get("filename", "Unknown"),
                "page": metadata.get("page"),
                "similarity_score": doc.get("similarity_score", 0.0),
                "content_preview": doc.get("content", "")[:200] + "..." if len(doc.get("content", "")) > 200 else doc.get("content", "")
            })
        
        return {
            "answer": llm_response["answer"],
            "sources": sources,
            "tokens_used": llm_response.get("tokens_used", 0)
        }
        
    except Exception as e:
        st.error(f"AI response generation failed: {e}")
        return None

def show_quick_actions():
    """Display quick action buttons."""
    if not st.session_state.collection_id:
        return
        
    st.header("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Generate Summary", type="secondary"):
            generate_summary()
    
    with col2:
        if st.button("🎯 Extract Key Points", type="secondary"):
            extract_key_points()
    
    with col3:
        if st.button("❓ Ask Example Question", type="secondary"):
            ask_example_question()

def generate_summary():
    """Generate document summary."""
    try:
        with st.spinner("📝 Generating summary..."):
            # Get sample documents for summary
            query_embedding = embed_proc.generate_query_embedding("overview summary main topics")
            sample_docs = vector_store.similarity_search(
                collection_id=st.session_state.collection_id,
                query_embedding=query_embedding,
                n_results=10
            )
            
            if sample_docs:
                # Combine content
                combined_text = "\n\n".join([doc.get("content", "") for doc in sample_docs])
                
                # Generate summary
                summary = asyncio.run(llm_client.summarize_document(combined_text, max_length=500))
                
                summary_msg = f"📝 **Document Summary**\n\n{summary}"
                st.session_state.messages.append({"role": "assistant", "content": summary_msg})
                st.success("✅ Summary generated!")
                st.rerun()
            else:
                st.warning("No documents found to summarize.")
                
    except Exception as e:
        st.error(f"❌ Summary generation failed: {e}")

def extract_key_points():
    """Extract key points from documents."""
    try:
        with st.spinner("🎯 Extracting key points..."):
            # Get sample documents
            query_embedding = embed_proc.generate_query_embedding("important key points main ideas")
            sample_docs = vector_store.similarity_search(
                collection_id=st.session_state.collection_id,
                query_embedding=query_embedding,
                n_results=8
            )
            
            if sample_docs:
                # Combine content
                combined_text = "\n\n".join([doc.get("content", "") for doc in sample_docs])
                
                # Extract key points
                key_points = asyncio.run(llm_client.extract_key_points(combined_text, num_points=5))
                
                points_text = "\n".join([f"{i+1}. {point}" for i, point in enumerate(key_points)])
                key_points_msg = f"🎯 **Key Points**\n\n{points_text}"
                
                st.session_state.messages.append({"role": "assistant", "content": key_points_msg})
                st.success("✅ Key points extracted!")
                st.rerun()
            else:
                st.warning("No documents found to extract key points from.")
                
    except Exception as e:
        st.error(f"❌ Key point extraction failed: {e}")

def ask_example_question():
    """Ask an example question."""
    example_questions = [
        "What are the main topics covered in these documents?",
        "Can you explain the key concepts in simple terms?",
        "What are the most important points I should remember?",
        "How do the different sections relate to each other?",
        "What practical applications are discussed?",
        "Are there any examples or case studies mentioned?"
    ]
    
    import random
    question = random.choice(example_questions)
    
    # Add to chat
    st.session_state.messages.append({"role": "user", "content": question})
    st.rerun()

def show_features():
    """Show features when no documents are loaded."""
    if st.session_state.collection_id:
        return
    
    st.header("✨ StudyFlow Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🤖 AI-Powered Chat
        Ask questions and get intelligent answers from your documents using advanced RAG technology.
        
        ### 📄 PDF Processing
        Upload multiple PDF files and automatically extract and index their content for searching.
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Smart Summaries
        Generate concise summaries and extract key points from your documents automatically.
        
        ### 🔍 Source Citations
        Get answers with references to specific pages and sections in your documents.
        """)

def main():
    """Main application function."""
    # Show header
    show_header()
    
    # Handle file uploads in sidebar
    handle_file_upload()
    
    # Main content area
    if st.session_state.collection_id:
        # Show chat interface if documents are loaded
        show_chat_interface()
        show_quick_actions()
    else:
        # Show welcome message and features
        st.markdown("""
        ## Welcome to StudyFlow! 🎓
        
        Upload your PDF documents using the sidebar, and I'll help you understand and learn from them.
        
        **How to get started:**
        1. 📚 Upload PDF files using the sidebar
        2. 🚀 Click "Process Documents" 
        3. 💬 Start asking questions about your documents
        4. ⚡ Use quick actions for summaries and key points
        """)
        
        show_features()
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit • StudyFlow v1.0.0")

if __name__ == "__main__":
    main()