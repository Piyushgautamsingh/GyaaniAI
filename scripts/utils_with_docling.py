import os
import uuid
from uuid import uuid4
import tempfile
import logging
import requests
import streamlit as st
import psutil
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from urllib.parse import urlparse
from docling.document_converter import DocumentConverter
from llama_index.core import Settings, PromptTemplate, VectorStoreIndex
from llama_index.core.schema import Document, TextNode
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding  # Updated import
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Import configuration
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), 
                   format=config.LOG_FORMAT)
logger = logging.getLogger(config.LOGGER_NAME)

def initialize_session_state() -> None:
    """Initialize session state variables if they don't exist."""
    if "session_id" not in st.session_state:
        st.session_state.update({
            "session_id": str(uuid.uuid4()),
            "chat_messages": [],
            "scanned": False,        # Flag to track if document has been scanned/indexed
            "document_key": None,    # To store the active document key
            "documents": [],         # List of loaded documents
            "memory_warning": False  # Flag to track if memory warning has been shown
        })

def monitor_memory_usage(threshold_percent: float = config.MEMORY_WARNING_THRESHOLD) -> bool:
    """Monitor system memory usage and return True if above threshold."""
    memory_info = psutil.virtual_memory()
    if memory_info.percent > threshold_percent and not st.session_state.memory_warning:
        st.session_state.memory_warning = True
        return True
    return False

def clear_chat_history() -> None:
    """Clear chat history and reset context."""
    st.session_state.update({"chat_messages": []})
    st.success("Chat history cleared!")

def clear_all_resources() -> None:
    """Clear all resources including cached objects and documents."""
    try:
        # Clear session state
        st.session_state.update({
            "chat_messages": [],
            "scanned": False,
            "document_key": None,
            "documents": [],
            "memory_warning": False
        })
        
        # Clear cache
        initialize_llm.clear()
        load_and_index_document.clear()
        load_query_engine_from_db.clear()
        
        # Delete the entire collection from Qdrant
        qdrant_client = QdrantClient(
            host=config.QDRANT_DB_HOST, 
            port=config.QDRANT_DB_PORT, 
            timeout=config.QDRANT_TIMEOUT
        )
        qdrant_client.delete_collection(collection_name=config.DOCUMENT_COLLECTION_NAME)
        
        st.success("All resources and database data cleared successfully!")
    except Exception as e:
        logger.error(f"Error clearing resources: {str(e)}")
        st.error(f"Error clearing resources: {str(e)}")

def validate_url(url: str) -> bool:
    """Validate if the URL is well-formed and has an allowed file extension."""
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        
        path = result.path.lower()
        return any(path.endswith(f".{ext}") for ext in config.ALLOWED_FILE_TYPES)
    except Exception:
        return False

def download_document_from_url(url: str, temp_dir: str) -> Optional[str]:
    """Download a document from a URL and save it with security checks."""
    if not validate_url(url):
        st.error("Invalid URL or unsupported file type. Please check and try again.")
        return None
    
    try:
        # First make a HEAD request to check file size
        head_response = requests.head(url, timeout=config.REQUEST_TIMEOUT)
        content_length = int(head_response.headers.get("Content-Length", 0))
        
        # Check if file is too large (convert MB to bytes)
        if content_length > config.MAX_URL_SIZE_MB * 1024 * 1024:
            st.error(f"File too large (>{config.MAX_URL_SIZE_MB}MB). Please use a smaller file.")
            return None
        
        # Now get the actual file
        response = requests.get(url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Create safe filename
        filename = os.path.basename(urlparse(url).path)
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
        if not safe_filename:
            safe_filename = "downloaded_document.pdf"
            
        file_path = os.path.join(temp_dir, safe_filename)
        with open(file_path, "wb") as file:
            file.write(response.content)
        return file_path
    except requests.RequestException as e:
        if "404" in str(e):
            st.error("Document not found (404). Please check the URL and try again.")
        elif "timeout" in str(e).lower():
            st.error("Download timeout. Please try again or use a different URL.")
        else:
            st.error(f"Download error: {str(e)}")
        logger.error(f"Download error for URL {url}: {str(e)}")
        return None

@st.cache_resource
def initialize_llm(model_name: str = config.DEFAULT_LLM_MODEL, request_timeout: float = 120.0) -> Ollama:
    """Initialize the LLM model."""
    try:
        logger.info(f"Initializing LLM with model: {model_name}")
        return Ollama(model=model_name, request_timeout=request_timeout)
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

def cleanup_old_documents(max_docs: int = config.MAX_DOCUMENTS_PER_SESSION) -> None:
    """Clean up old documents if we've exceeded the maximum."""
    documents = st.session_state.documents
    if len(documents) > max_docs:
        # Keep only the most recent documents up to max_docs
        documents_to_remove = documents[:-max_docs]
        remaining_documents = documents[-max_docs:]
        
        # Remove from Qdrant
        try:
            qdrant_client = QdrantClient(
                host=config.QDRANT_DB_HOST, 
                port=config.QDRANT_DB_PORT, 
                timeout=config.QDRANT_TIMEOUT
            )
            for doc in documents_to_remove:
                try:
                    qdrant_client.delete(
                        collection_name=config.DOCUMENT_COLLECTION_NAME,
                        points_selector=[doc["key"]]
                    )
                    logger.info(f"Removed document {doc['key']} from Qdrant")
                except Exception as e:
                    logger.error(f"Error removing document {doc['key']}: {str(e)}")
        except Exception as e:
            logger.error(f"Error accessing Qdrant during cleanup: {str(e)}")
        
        # Update session state
        st.session_state.documents = remaining_documents
        logger.info(f"Cleaned up {len(documents_to_remove)} old documents")

def ensure_qdrant_collection(model_name: str) -> Optional[QdrantClient]:
    """Ensure Qdrant collection exists with proper vector dimensions and hybrid search enabled."""
    try:
        qdrant_client = QdrantClient(
            host=config.QDRANT_DB_HOST, 
            port=config.QDRANT_DB_PORT, 
            timeout=config.QDRANT_TIMEOUT
        )
        
        # Get embedding dimensions for the model
        vector_size = config.get_embedding_dimensions(model_name)
        
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        
        if config.DOCUMENT_COLLECTION_NAME not in collection_names:
            # Create new collection with proper dimensions and hybrid search enabled
            qdrant_client.recreate_collection(
                collection_name=config.DOCUMENT_COLLECTION_NAME,
                vectors_config={
                    "text-dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)  # Dense vectors for ANN
                },
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams()  # Sparse vectors for BM25
                }
            )
            logger.info(f"Created collection: {config.DOCUMENT_COLLECTION_NAME} with dimension {vector_size} and hybrid search enabled.")
        else:
            # Verify vector dimensions match
            collection_info = qdrant_client.get_collection(config.DOCUMENT_COLLECTION_NAME)
            existing_size = collection_info.config.params.vectors["text-dense"].size  # Correctly access the size attribute
            
            if existing_size != vector_size:
                logger.warning(
                    f"Vector dimension mismatch: Collection has {existing_size}, model needs {vector_size}. "
                    "Recreating collection."
                )
                qdrant_client.recreate_collection(
                    collection_name=config.DOCUMENT_COLLECTION_NAME,
                    vectors_config={
                        "text-dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)  # Dense vectors for ANN
                    },
                    sparse_vectors_config={
                        "text-sparse": models.SparseVectorParams()  # Sparse vectors for BM25
                    }
                )
                logger.info(f"Recreated collection with new dimension {vector_size} and hybrid search enabled.")
                
        return qdrant_client
    except Exception as e:
        logger.error(f"Error ensuring Qdrant collection: {str(e)}")
        st.error(f"Database connection error: {str(e)}")
        return None

def rerank_results(query: str, documents: List[str], model_name: str = "mixedbread-ai/mxbai-rerank-large-v1") -> List[str]:
    """Rerank documents based on relevance to the query using a reranking model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Tokenize the query and documents
    inputs = tokenizer([query] * len(documents), documents, return_tensors="pt", padding=True, truncation=True)
    
    # Get scores from the model
    with torch.no_grad():
        scores = model(**inputs).logits
    
    # Sort documents by score
    sorted_indices = torch.argsort(scores, descending=True)
    reranked_documents = [documents[i] for i in sorted_indices]
    
    return reranked_documents

@st.cache_resource
def load_and_index_document(file_path: str, document_key: str, embed_model_name: str = config.DEFAULT_EMBEDDING_MODEL):
    """Load and index a document using Docling for parsing."""
    try:
        logger.info(f"Loading document: {file_path}")
        progress_placeholder = st.empty()
        progress_placeholder.info("Converting document...")
        
        # Convert the document using Docling
        converter = DocumentConverter()
        result = converter.convert(file_path)

        if not result or not result.document:
            progress_placeholder.error("Conversion returned no result.")
            return None

        # Extract text from the document
        progress_placeholder.info("Extracting text...")
        document_text = result.document.export_to_markdown()
        # print(document_text)

        if not document_text.strip():
            progress_placeholder.error("Failed to extract text from the document.")
            return None

        # Initialize Qdrant with proper collection
        progress_placeholder.info("Initializing database...")
        qdrant_client = ensure_qdrant_collection(embed_model_name)
        if not qdrant_client:
            progress_placeholder.error("Failed to initialize database.")
            return None

        # Create vector store
        vector_store = QdrantVectorStore(
            client=qdrant_client, 
            collection_name=config.DOCUMENT_COLLECTION_NAME,
            enable_hybrid=True  # Enable hybrid search
        )

        # Configure embeddings using OllamaEmbedding
        progress_placeholder.info("Generating embeddings...")
        embed_model = OllamaEmbedding(model_name=embed_model_name)  # Updated to use OllamaEmbedding

        # Chunk the document text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust chunk size as needed
            chunk_overlap=200,  # Adjust overlap as needed
        )
        chunks = text_splitter.split_text(document_text)

        # Generate embeddings for each chunk and create TextNode objects
        progress_placeholder.info("Indexing document...")
        text_nodes = []
        for chunk in chunks:
            embeddings = embed_model.get_text_embedding(chunk)
            text_node = TextNode(
                text=chunk,
                embedding=embeddings,
                metadata={
                    "source": os.path.basename(file_path),
                    "timestamp": str(uuid.uuid4()),
                    "document_key": document_key
                },
                id_=str(uuid4())  # Use a valid UUID for the ID
            )
            text_nodes.append(text_node)

        # Add the document chunks to the Qdrant collection
        vector_store.add(nodes=text_nodes)

        # Add to document list and clean up old ones
        doc_info = {
            "key": document_key,
            "name": os.path.basename(file_path),
            "timestamp": str(uuid.uuid4())
        }
        st.session_state.documents.append(doc_info)
        cleanup_old_documents()

        # Mark as scanned and store the document key
        st.session_state.document_key = document_key
        st.session_state.scanned = True
        
        progress_placeholder.success("Document indexed successfully!")
        progress_placeholder.empty()
        return True

    except Exception as e:
        logger.error(f"Indexing error: {str(e)}")
        st.error(f"Indexing error: {str(e)}")
        return None

@st.cache_resource
def load_query_engine_from_db(llm_model_name: str = config.DEFAULT_LLM_MODEL, 
                             embed_model_name: str = config.DEFAULT_EMBEDDING_MODEL,
                             custom_prompt: Optional[str] = None,
                             document_id: Optional[str] = None):
    """Load the query engine from the persistent Qdrant with hybrid search enabled."""
    try:
        logger.info(f"Loading query engine from DB for document: {document_id if document_id else 'all'}")
        
        # Initialize Qdrant
        qdrant_client = QdrantClient(
            host=config.QDRANT_DB_HOST, 
            port=config.QDRANT_DB_PORT, 
            timeout=config.QDRANT_TIMEOUT
        )
        
        # Create vector store with hybrid search enabled
        vector_store = QdrantVectorStore(
            client=qdrant_client, 
            collection_name=config.DOCUMENT_COLLECTION_NAME,
            enable_hybrid=True  # Enable hybrid search
        )

        # Configure embeddings and LLM
        Settings.embed_model = OllamaEmbedding(model_name=embed_model_name)
        Settings.llm = initialize_llm(model_name=llm_model_name)

        # Load the index from the vector store with optional filter
        if document_id:
            # Filter by document ID
            vector_store_query_kwargs = {
                "filter": {"document_key": document_id}
            }
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, 
                vector_store_query_kwargs=vector_store_query_kwargs
            )
        else:
            # No filter, use all documents
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # Set up prompt template
        prompt_template = PromptTemplate(custom_prompt or config.DEFAULT_PROMPT_TEMPLATE)

        # Create query engine with hybrid search (BM25 + ANN) and reranking
        query_engine = index.as_query_engine(
            streaming=True,
            similarity_top_k=5,  # Adjust as needed
            vector_store_query_mode="hybrid"  # Enable hybrid search
        )
        
        # Add reranking step
        def rerank_query(query: str, documents: List[str]) -> List[str]:
            return rerank_results(query, documents)
        
        query_engine.update_prompts({"response_synthesizer:text_qa_template": prompt_template})
        query_engine.rerank = rerank_query

        return query_engine
    except Exception as e:
        logger.error(f"Error loading query engine from DB: {str(e)}")
        st.error(f"Error loading query engine: {str(e)}")
        return None

def get_document_list() -> List[Dict[str, str]]:
    """Get list of documents indexed in Qdrant."""
    try:
        qdrant_client = QdrantClient(
            host=config.QDRANT_DB_HOST, 
            port=config.QDRANT_DB_PORT, 
            timeout=config.QDRANT_TIMEOUT
        )

        # Check if the collection exists
        collections = qdrant_client.get_collections()
        collection_names = [collection.name for collection in collections.collections]
        if config.DOCUMENT_COLLECTION_NAME not in collection_names:
            logger.info(f"Collection {config.DOCUMENT_COLLECTION_NAME} does not exist.")
            return []

        result = qdrant_client.scroll(
            collection_name=config.DOCUMENT_COLLECTION_NAME,
            with_payload=True,
            limit=config.MAX_DOCUMENTS_PER_SESSION
        )
        
        documents = []
        for item in result[0]:  # Scroll returns tuple with items and next_page_offset
            payload = item.payload or {}
            documents.append({
                "key": item.id,
                "name": payload.get("source", "Unknown document"),
                "timestamp": payload.get("timestamp", "")
            })
        
        return documents
    except Exception as e:
        logger.error(f"Error getting document list: {str(e)}")
        return []

def select_active_document(document_key: str) -> None:
    """Select a document as the active one for querying."""
    st.session_state.document_key = document_key
    st.session_state.scanned = True
    st.success(f"Selected document as active")