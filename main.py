import os
import uuid
import tempfile
import streamlit as st
from pathlib import Path
import shutil
import logging
from qdrant_client import QdrantClient, models
import asyncio

# Import configuration and utilities
import config
from utils import (
    initialize_session_state, monitor_memory_usage, clear_chat_history,
    clear_all_resources, download_document_from_url_async, initialize_llm,
    load_and_index_document, load_query_engine_from_db, get_document_list,
    select_active_document, save_to_uploaded_files
)

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), 
                   format=config.LOG_FORMAT)
logger = logging.getLogger(config.LOGGER_NAME)

# Initialize the application
def main():
    # Set page configuration
    st.set_page_config(
        page_title=config.APP_TITLE_BAR, 
        page_icon=config.APP_ICON, 
        layout=config.APP_LAYOUT
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Get user session ID
    user_session_id = st.session_state.session_id
    
    # Display app title and subtitle
    st.title(config.APP_TITLE)
    st.subheader(config.APP_SUBTITLE)
    
    # Check memory usage
    if monitor_memory_usage():
        st.warning("Memory usage is high. Consider clearing resources to improve performance.")
    
    # Create sidebar and main content layout
    create_sidebar(user_session_id)
    create_main_content()
    
    # Add footer
    st.markdown("---")
    st.markdown(config.APP_FOOTER)

# Folder to store uploaded files
UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def manage_uploaded_files(file_path, document_key):
    """Manage uploaded files to keep only the 5 most recent ones."""
    # Save the new file
    new_file_path = os.path.join(UPLOAD_FOLDER, os.path.basename(file_path))
    shutil.move(file_path, new_file_path)
    
    # List all files in the upload folder
    files = sorted(Path(UPLOAD_FOLDER).iterdir(), key=os.path.getmtime, reverse=True)
    
    # Keep only the 5 most recent files
    for old_file in files[5:]:
        os.remove(old_file)

def create_sidebar(user_session_id):
    """Create sidebar for document upload, management, and settings."""
    with st.sidebar:
        st.subheader("Document Management")
        
        # Upload document section
        with st.expander("Upload Document", expanded=True):
            # File uploader
            uploaded_file = st.file_uploader("Upload a document", 
                                           type=config.ALLOWED_FILE_TYPES,
                                           help="Supported file types: " + ", ".join(config.ALLOWED_FILE_TYPES))
            
            # URL input
            url = st.text_input("Or enter document URL:", 
                               help="Enter a URL to a document (PDF, DOCX, TXT, etc.) or a web page (e.g., Wikipedia, GitHub README)")
            
            # Generate document key
            document_key = str(uuid.uuid4())
            
            # Process uploaded file
            if uploaded_file is not None:
                if st.button("Process Document", key="process_upload"):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Save uploaded file to temp directory
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Save to uploaded_files directory
                        saved_file_path = save_to_uploaded_files(file_path)
                        
                        # Load and index the document
                        if load_and_index_document(saved_file_path, document_key):
                            st.success(f"Document '{uploaded_file.name}' processed successfully!")
            
            # Process URL
            elif url:
                if st.button("Process URL", key="process_url"):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Download the document or scrape the web page
                        file_path = asyncio.run(download_document_from_url_async(url, temp_dir))
                        if file_path:
                            # Save to uploaded_files directory
                            saved_file_path = save_to_uploaded_files(file_path)
                            
                            # Load and index the document
                            if load_and_index_document(saved_file_path, document_key):
                                st.success(f"Document from URL processed successfully!")
        
        # Document list and delete functionality
        with st.expander("Your Documents", expanded=True):
            if not st.session_state.documents:
                st.info("No documents found. Please upload a document to begin.")
            else:
                # Group documents by unique document name to prevent duplicates
                documents_by_name = {}
                for doc in st.session_state.documents:
                    doc_name = doc["name"]
                    if doc_name not in documents_by_name:
                        documents_by_name[doc_name] = doc
                
                # Display the unique documents with delete buttons
                for doc_name, doc in documents_by_name.items():
                    with st.container(border=True):
                        cols = st.columns([3, 1])
                        with cols[0]:
                            st.markdown(f"**{doc_name}**")
                        with cols[1]:
                            # Custom CSS for larger delete button
                            st.markdown(
                                """
                                <style>
                                    .stButton button {
                                        width: 100%;
                                        height: 40px;
                                        font-size: 16px;
                                        padding: 0;
                                    }
                                </style>
                                """,
                                unsafe_allow_html=True
                            )
                            if st.button("Delete", key=f"delete_{doc['key']}"):
                                delete_document(doc["key"])
        
        # Settings section
        with st.expander("Settings"):
            # LLM model selection
            llm_model = st.selectbox("LLM Model", 
                                    options=config.AVAILABLE_LLM_MODELS,
                                    index=config.AVAILABLE_LLM_MODELS.index(config.DEFAULT_LLM_MODEL))
            
            # Embedding model selection
            embed_model_names = [model["display_name"] for model in config.AVAILABLE_EMBEDDING_MODELS]
            default_index = 0
            for i, model in enumerate(config.AVAILABLE_EMBEDDING_MODELS):
                if model["name"] == config.DEFAULT_EMBEDDING_MODEL:
                    default_index = i
                    break
            selected_embed_display = st.selectbox("Embedding Model", 
                                               options=embed_model_names,
                                               index=default_index)
            selected_embed_model = config.DEFAULT_EMBEDDING_MODEL  # fallback
            for model in config.AVAILABLE_EMBEDDING_MODELS:
                if model["display_name"] == selected_embed_display:
                    selected_embed_model = model["name"]
                    break
            
            # Custom prompt template
            custom_prompt = st.text_area("Custom Prompt Template", 
                                       value=config.DEFAULT_PROMPT_TEMPLATE,
                                       height=150)
        
        # Resource management
        with st.expander("Resource Management"):
            if st.button("Clear Chat History"):
                clear_chat_history()
            if st.button("Clear All Resources"):
                clear_all_resources()

def delete_document(document_key):
    """Delete a document from the system."""
    try: 
        qdrant_client = QdrantClient(
            host=config.QDRANT_DB_HOST, 
            port=config.QDRANT_DB_PORT, 
            timeout=config.QDRANT_TIMEOUT
        )
        
        # Delete the document from Qdrant using a filter
        qdrant_client.delete(
            collection_name=config.DOCUMENT_COLLECTION_NAME,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="document_key", match=models.MatchValue(value=document_key))]
            )
        )
        
        # Update session state
        st.session_state.documents = [doc for doc in st.session_state.documents if doc["key"] != document_key]
        
        # If this was the active document, clear the active document
        if st.session_state.document_key == document_key:
            st.session_state.document_key = None
            st.session_state.scanned = False
        
        st.success("Document deleted successfully!")
        st.rerun()
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        st.error(f"Error deleting document: {str(e)}")

def create_main_content():
    """Create the main content area for chat."""
    st.subheader("Chat with your Document")
    
    # Display chat messages
    for message in st.session_state.get("chat_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if not st.session_state.scanned:
        st.info("Please upload and process a document first.")
    else:
        query = st.chat_input("Ask a question about your document...")
        if query:
            # Add user message to chat
            st.session_state.chat_messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            # Create response message container
            with st.chat_message("assistant"):
                response_container = st.empty()
                response_text = ""
                
                try:
                    # Display "Thinking..." message
                    with st.status("Thinking...", expanded=False) as status:
                        # Get query engine for the active document
                        query_engine = load_query_engine_from_db(
                            llm_model_name=config.DEFAULT_LLM_MODEL,
                            embed_model_name=config.DEFAULT_EMBEDDING_MODEL,
                            document_id=st.session_state.document_key
                        )
                        
                        if not query_engine:
                            st.error("Failed to initialize query engine.")
                            return
                        
                        # Stream response
                        response = query_engine.query(query)
                        for text in response.response_gen:
                            response_text += text
                            response_container.markdown(response_text + "▌")
                        
                        # Final response
                        response_container.markdown(response_text)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                        
                        # Update status to indicate completion
                        status.update(label="Response ready!", state="complete")
                    
                except Exception as e:
                    logger.error(f"Query error: {str(e)}")
                    st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()