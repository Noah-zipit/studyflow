import streamlit as st
import openai
import PyPDF2
import io
import requests

# Page config
st.set_page_config(
    page_title="StudyFlow - AI Study Assistant",
    page_icon="🎓",
    layout="wide"
)

# Header
st.title("🎓 StudyFlow - AI Study Assistant")
st.markdown("Upload PDFs and chat with your documents using AI!")

# Check for API key
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ Please add OPENAI_API_KEY to your Streamlit secrets")
    st.info("Go to your app dashboard → Settings → Secrets")
    st.stop()

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url=st.secrets.get("OPENAI_BASE_URL", "https://tokenin.my.id/v1")
)

MODEL_NAME = st.secrets.get("OPENAI_MODEL_NAME", "myt/gemini-3.5-flash-free")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents' not in st.session_state:
    st.session_state.documents = []

# Sidebar for file upload
with st.sidebar:
    st.header("📚 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload your PDF documents"
    )
    
    if uploaded_files:
        if st.button("🚀 Process Documents"):
            st.session_state.documents = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {file.name}...")
                    
                    # Extract text from PDF
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                    text = ""
                    
                    for page_num, page in enumerate(pdf_reader.pages):
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page.extract_text()
                    
                    if text.strip():
                        st.session_state.documents.append({
                            'name': file.name,
                            'text': text,
                            'pages': len(pdf_reader.pages)
                        })
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    else:
                        st.warning(f"No text found in {file.name}")
                        
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
            
            status_text.text("✅ Processing complete!")
            st.success(f"Processed {len(st.session_state.documents)} documents")
            
            # Add welcome message
            if st.session_state.documents:
                welcome = f"📚 I've processed {len(st.session_state.documents)} documents:\n\n"
                for doc in st.session_state.documents:
                    welcome += f"• **{doc['name']}** ({doc['pages']} pages)\n"
                welcome += "\nAsk me anything about these documents!"
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": welcome
                })
    
    # Show loaded documents
    if st.session_state.documents:
        st.success("✅ Documents loaded:")
        for doc in st.session_state.documents:
            st.write(f"📄 {doc['name']} ({doc['pages']} pages)")
        
        if st.button("🗑️ Clear All"):
            st.session_state.documents = []
            st.session_state.messages = []
            st.rerun()

# Main chat interface
st.header("💬 Chat with Your Documents")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        if st.session_state.documents:
            try:
                with st.spinner("🤔 Thinking..."):
                    # Combine document content
                    context = "\n\n".join([
                        f"=== {doc['name']} ===\n{doc['text'][:3000]}..." 
                        for doc in st.session_state.documents
                    ])
                    
                    # Truncate context if too long
                    if len(context) > 8000:
                        context = context[:8000] + "\n\n[Content truncated...]"
                    
                    # Create messages for API
                    messages = [
                        {
                            "role": "system", 
                            "content": "You are a helpful study assistant. Answer questions based on the provided documents. If the answer isn't in the documents, say so."
                        },
                        {
                            "role": "user", 
                            "content": f"Documents:\n{context}\n\nQuestion: {prompt}"
                        }
                    ]
                    
                    # Call OpenAI API
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=500,
                        temperature=0.7
                    )
                    
                    answer = response.choices[0].message.content
                    st.write(answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer
                    })
                    
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_msg
                })
        else:
            no_docs = "📚 Please upload and process some documents first!"
            st.warning(no_docs)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": no_docs
            })

# Quick actions
if st.session_state.documents:
    st.header("⚡ Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Summarize All"):
            summary_prompt = "Please provide a comprehensive summary of all the uploaded documents, highlighting the main topics and key points."
            st.session_state.messages.append({"role": "user", "content": summary_prompt})
            st.rerun()
    
    with col2:
        if st.button("🎯 Key Points"):
            key_points_prompt = "Extract the 5 most important key points from all the uploaded documents."
            st.session_state.messages.append({"role": "user", "content": key_points_prompt})
            st.rerun()
    
    with col3:
        if st.button("❓ Study Questions"):
            questions_prompt = "Generate 3 study questions based on the content of the uploaded documents."
            st.session_state.messages.append({"role": "user", "content": questions_prompt})
            st.rerun()

# Footer
if not st.session_state.documents:
    st.info("👆 Upload PDF files in the sidebar to get started!")
    
    st.markdown("""
    ## ✨ Features
    - 📄 **PDF Upload**: Extract text from your PDF documents
    - 🤖 **AI Chat**: Ask questions about your documents  
    - 📝 **Summaries**: Get quick summaries of your content
    - 🎯 **Key Points**: Extract important information automatically
    """)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit • StudyFlow v1.0")
