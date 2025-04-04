import os
import uuid
import tempfile
import streamlit as st
from pathlib import Path
import shutil
import logging
from qdrant_client import QdrantClient, models
import asyncio
import datetime
from typing import Dict
import time

# Import configuration and utilities
import config
from utils import (
    initialize_session_state, monitor_memory_usage, clear_chat_history,
    clear_all_resources, download_document_from_url_async, initialize_llm,
    load_and_index_document, load_query_engine_from_db, get_document_list,
    select_active_document, save_to_uploaded_files, cleanup_old_indices,
    ensure_qdrant_collection
)



# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), 
                   format=config.LOG_FORMAT)
logger = logging.getLogger(config.LOGGER_NAME)

def main():
    # Initialize Qdrant collection at startup
    ensure_qdrant_collection(config.DEFAULT_EMBEDDING_MODEL)
    
    # Clean up old indices at startup
    cleanup_old_indices()
    
    # Set page configuration
    st.set_page_config(
        page_title=config.APP_TITLE_BAR, 
        page_icon=config.APP_ICON, 
        layout=config.APP_LAYOUT
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Display app title and subtitle
    st.title(config.APP_TITLE)
    st.subheader(config.APP_SUBTITLE)
    
    # Check memory usage
    if monitor_memory_usage():
        st.warning("Memory usage is high. Consider clearing resources to improve performance.")
    
    # Create sidebar and main content layout
    create_sidebar()
    create_main_content()
    
    # Add footer
    st.markdown("---")
    st.markdown(config.APP_FOOTER)

def create_sidebar():
    """Create sidebar for document upload, management, and settings."""
    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <h2 style="margin: 0;">📂 Document Management</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload document section
        with st.expander("📤 Upload Document", expanded=True):
            uploaded_file = st.file_uploader("Choose a file", 
                                           type=config.ALLOWED_FILE_TYPES,
                                           help="Supported file types: " + ", ".join(config.ALLOWED_FILE_TYPES))
            
            st.markdown("<div style='margin: 10px 0; text-align: center; color: #666;'>or</div>", 
                       unsafe_allow_html=True)
            url = st.text_input("Enter document URL", 
                               placeholder="https://example.com/document.pdf",
                               help="Enter a URL to a document or a web page")
            
            # Generate document key with date suffix for daily indexing
            today = datetime.date.today().strftime("%Y-%m-%d")
            document_key = f"{str(uuid.uuid4())}_{today}"
            
            if uploaded_file is not None:
                if st.button("Process Document", key="process_upload"):
                    with st.spinner("Processing document..."):
                        with tempfile.TemporaryDirectory() as temp_dir:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            
                            saved_file_path = save_to_uploaded_files(file_path)
                            
                            if load_and_index_document(saved_file_path, document_key):
                                st.toast(f"✅ Document '{uploaded_file.name}' processed successfully!", icon="✅")
            
            elif url:
                if st.button("Process URL", key="process_url"):
                    with st.spinner("Downloading and processing URL..."):
                        with tempfile.TemporaryDirectory() as temp_dir:
                            file_path = asyncio.run(download_document_from_url_async(url, temp_dir))
                            if file_path:
                                saved_file_path = save_to_uploaded_files(file_path)
                                if load_and_index_document(saved_file_path, document_key):
                                    st.toast("✅ Document from URL processed successfully!", icon="✅")
        
        # Document list
        with st.expander("📄 Your Documents", expanded=True):
            documents = get_document_list()
            if not documents:
                st.info("No documents found. Please upload a document to begin.")
            else:
                # Get unique documents by name
                unique_docs = {}
                for doc in documents:
                    base_name = os.path.splitext(doc['name'])[0]  # Remove extensions for comparison
                    if base_name not in unique_docs:
                        unique_docs[base_name] = doc
                
                # Display documents
                for base_name, doc in unique_docs.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if st.button(f"📄 {doc['name']}", key=f"select_{doc['key']}"):
                            select_active_document(doc["key"])
                    with col2:
                        if st.button("🗑️", key=f"delete_{doc['key']}"):
                            with st.spinner(f"Deleting {doc['name']}..."):
                                delete_document(doc)

        # Settings section with icons
        with st.expander("⚙️ Settings", expanded=False):
            # LLM model selection
            llm_model = st.selectbox("LLM Model", 
                                   options=config.AVAILABLE_LLM_MODELS,
                                   index=config.AVAILABLE_LLM_MODELS.index(config.DEFAULT_LLM_MODEL),
                                   help="Select the language model to use for generating responses")
            
            # Embedding model selection
            embed_model_names = [model["display_name"] for model in config.AVAILABLE_EMBEDDING_MODELS]
            default_index = 0
            for i, model in enumerate(config.AVAILABLE_EMBEDDING_MODELS):
                if model["name"] == config.DEFAULT_EMBEDDING_MODEL:
                    default_index = i
                    break
            selected_embed_display = st.selectbox("Embedding Model", 
                                                options=embed_model_names,
                                                index=default_index,
                                                help="Select the model to use for document embeddings")

        # Resource management with icons
        with st.expander("🗑️ Resource Management", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Chat History", help="Clear all chat messages"):
                    with st.spinner("Clearing chat history..."):
                        clear_chat_history()
                        st.toast("Chat history cleared!", icon="✅")
            with col2:
                if st.button("Clear All Resources", help="Clear all documents and chat history"):
                    with st.spinner("Clearing all resources..."):
                        clear_all_resources()
                        st.toast("All resources cleared!", icon="✅")

def delete_document(document_info: Dict[str, str]) -> None:
    """Delete a document from the system."""
    try: 
        qdrant_client = QdrantClient(
            host=config.QDRANT_DB_HOST, 
            port=config.QDRANT_DB_PORT, 
            timeout=config.QDRANT_TIMEOUT
        )
        
        # Delete from Qdrant
        qdrant_client.delete(
            collection_name=document_info["collection"],
            points_selector=models.Filter(
                must=[models.FieldCondition(key="id", match=models.MatchValue(value=document_info["key"]))]
            )
        )
        
        # Update session state
        if "documents" in st.session_state:
            st.session_state.documents = [
                doc for doc in st.session_state.documents 
                if doc["key"] != document_info["key"]
            ]
        
        if st.session_state.get("document_key") == document_info["key"]:
            st.session_state.document_key = None
            st.session_state.scanned = False
        
        st.toast("Document deleted successfully!", icon="✅")
        time.sleep(1)  # Give time for toast to appear
        st.rerun()  # Refresh the UI
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        st.error(f"Error deleting document: {str(e)}")


def create_main_content():
    """Create the main content area for chat."""
    st.subheader("💬 Chat with your Document")
    
    for message in st.session_state.get("chat_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Only show chat input if we have documents in the database
    documents = get_document_list()
    if not documents:
        st.info("ℹ️ Please upload and process a document first.")
    else:
        query = st.chat_input("Ask a question about your document...")
        if query:
            st.session_state.chat_messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.chat_message("assistant"):
                response_container = st.empty()
                response_text = ""
                
                try:
                    start_time = time.time()  # Start timing
                    
                    with st.status("🤔 Thinking...", expanded=False) as status:
                        query_engine = load_query_engine_from_db(
                            llm_model_name=config.DEFAULT_LLM_MODEL,
                            embed_model_name=config.DEFAULT_EMBEDDING_MODEL,
                            document_id=st.session_state.get("document_key")
                        )
                        
                        if not query_engine:
                            st.error("Failed to initialize query engine.")
                            return
                        
                        response = query_engine.query(query)
                        for text in response.response_gen:
                            response_text += text
                            response_container.markdown(response_text + "▌")
                        
                        # Calculate response time
                        end_time = time.time()
                        response_time = end_time - start_time
                        
                        # Add response time to message
                        response_text += f"\n\n⏱️ Response generated in {response_time:.2f} seconds"
                        response_container.markdown(response_text)
                        st.session_state.chat_messages.append({
                            "role": "assistant", 
                            "content": response_text
                        })
                        
                        status.update(
                            label=f"✅ Response ready! ({response_time:.2f}s)", 
                            state="complete"
                        )
                    
                except Exception as e:
                    logger.error(f"Query error: {str(e)}")
                    st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()