import os
import uuid
from uuid import uuid4
import tempfile
import logging
import requests
import streamlit as st
import psutil
from typing import Optional, List, Dict, Any, Tuple, Generator
from pathlib import Path
from urllib.parse import urlparse
import pdfplumber
from bs4 import BeautifulSoup
from llama_index.core import Settings, PromptTemplate, VectorStoreIndex
from llama_index.core.schema import Document, TextNode
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding   
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import asyncio
import shutil
import aiohttp
from functools import lru_cache

# Import configuration
import config

from docx import Document as DocxDocument

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


def extract_text_from_file(file_path: str) -> Optional[str]:
    """Extract text from a file based on its type."""
    try:
        if file_path.endswith(".pdf"):
            return extract_text_from_pdf(file_path)
        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        elif file_path.endswith(".html") or file_path.endswith(".htm"):
            return extract_text_from_html(file_path)
        elif file_path.endswith(".docx"):
            doc = DocxDocument(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_path.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        else:
            logger.error(f"Unsupported file type: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error extracting text from file {file_path}: {str(e)}")
        return None

def validate_url(url: str) -> bool:
    """Validate if the URL is well-formed and has an allowed file extension or is a web page."""
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False

        # Check if the URL is a web page (e.g., Wikipedia, GitHub)
        if any(domain in url.lower() for domain in config.WEB_PAGE_DOMAINS):
            return True

        # Check if the URL has an allowed file extension
        path = result.path.lower()
        return any(path.endswith(f".{ext}") for ext in config.ALLOWED_FILE_TYPES)
    except Exception:
        return False

# Ensure the uploaded_files directory exists
UPLOAD_FOLDER = "./uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_to_uploaded_files(file_path: str) -> str:
    """Save a file to the uploaded_files directory and return the new path."""
    filename = os.path.basename(file_path)
    new_file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # Handle duplicate filenames
    counter = 1
    while os.path.exists(new_file_path):
        name, ext = os.path.splitext(filename)
        new_file_path = os.path.join(UPLOAD_FOLDER, f"{name}_{counter}{ext}")
        counter += 1
    
    shutil.move(file_path, new_file_path)
    return new_file_path

def scrape_web_page(url: str) -> Optional[str]:
    """Scrape content from a web page and return the text."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract relevant content (e.g., paragraphs, headings)
        text = ""
        for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
            text += element.get_text() + "\n"
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error scraping web page {url}: {str(e)}")
        return None

async def download_document_from_url_async(url: str, temp_dir: str) -> Optional[str]:
    """Download a document from a URL asynchronously or scrape a web page."""
    if not validate_url(url):
        st.error("Invalid URL or unsupported file type. Please check and try again.")
        return None

    try:
        # Check if the URL should be treated as a web page
        is_web_page = (
            any(domain in url.lower() for domain in config.WEB_PAGE_DOMAINS) or
            any(url.lower().endswith(ext) for ext in config.WEB_PAGE_EXTENSIONS)
        )

        if is_web_page:
            # Scrape the web page
            text = scrape_web_page(url)
            if not text:
                st.error("Failed to scrape content from the web page.")
                return None
            
            # Save the scraped content as a text file
            filename = "web_page_content.txt"
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text)
            
            return file_path

        # Handle file downloads (e.g., PDF, DOCX, TXT, HTML, etc.)
        async with aiohttp.ClientSession() as session:
            # First make a HEAD request to check file size
            async with session.head(url, timeout=config.REQUEST_TIMEOUT) as head_response:
                content_length = int(head_response.headers.get("Content-Length", 0))

                # Check if file is too large
                if content_length > config.MAX_URL_SIZE_MB * 1024 * 1024:
                    st.error(f"File too large (>{config.MAX_URL_SIZE_MB}MB). Please use a smaller file.")
                    return None

            # Now get the actual file
            async with session.get(url, timeout=config.REQUEST_TIMEOUT) as response:
                response.raise_for_status()

                # Create safe filename
                filename = os.path.basename(urlparse(url).path)
                safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
                if not safe_filename:
                    safe_filename = "downloaded_document"

                # Add file extension if missing
                if not any(safe_filename.lower().endswith(ext) for ext in config.ALLOWED_FILE_TYPES):
                    safe_filename += ".txt"  # Default to .txt if no extension is found

                file_path = os.path.join(temp_dir, safe_filename)
                with open(file_path, "wb") as file:
                    file.write(await response.read())
                return file_path
    except Exception as e:
        logger.error(f"Download error for URL {url}: {str(e)}")
        return None

@st.cache_resource
def initialize_llm(model_name: str = config.DEFAULT_LLM_MODEL, request_timeout: float = 120.0) -> Ollama:
    """Initialize the LLM model with fallback."""
    try:
        logger.info(f"Initializing LLM with model: {model_name}")
        return Ollama(model=model_name, request_timeout=request_timeout)
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}. Trying fallback model.")
        try:
            return Ollama(model=config.FALLBACK_LLM_MODEL, request_timeout=request_timeout)
        except Exception as e:
            logger.error(f"Fallback LLM initialization failed: {str(e)}")
            st.error("Failed to initialize LLM. Please check your configuration.")
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
    """Ensure Qdrant collection exists with proper vector dimensions and payload indexing."""
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
            # Create new collection with proper dimensions and payload indexing
            qdrant_client.recreate_collection(
                collection_name=config.DOCUMENT_COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                payload_schema={
                    "source": models.PayloadSchemaType.KEYWORD,
                    "timestamp": models.PayloadSchemaType.INTEGER,
                    "document_key": models.PayloadSchemaType.KEYWORD
                }
            )
            logger.info(f"Created collection: {config.DOCUMENT_COLLECTION_NAME} with dimension {vector_size}.")
        else:
            # Verify vector dimensions match
            collection_info = qdrant_client.get_collection(config.DOCUMENT_COLLECTION_NAME)
            existing_size = collection_info.config.params.vectors.size

            if existing_size != vector_size:
                logger.warning(
                    f"Vector dimension mismatch: Collection has {existing_size}, model needs {vector_size}. "
                    "Recreating collection."
                )
                qdrant_client.recreate_collection(
                    collection_name=config.DOCUMENT_COLLECTION_NAME,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    payload_schema={
                        "source": models.PayloadSchemaType.KEYWORD,
                        "timestamp": models.PayloadSchemaType.INTEGER,
                        "document_key": models.PayloadSchemaType.KEYWORD
                    }
                )
                logger.info(f"Recreated collection with new dimension {vector_size}.")

        return qdrant_client
    except Exception as e:
        logger.error(f"Error ensuring Qdrant collection: {str(e)}")
        st.error(f"Database connection error: {str(e)}")
        return None

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pdfplumber and return as a single string."""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Ensure the page has text
                    text += page_text + "\n"  # Add a newline between pages
        return text.strip()  # Remove any leading/trailing whitespace
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return None

def extract_text_from_html(file_path: str) -> Optional[str]:
    """Extract text from an HTML file using BeautifulSoup."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            text = soup.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {str(e)}")
        return None

def get_embedding_with_prefix(texts: List[str], prefix: str = "Represent this document for retrieval: ", embed_model_name: str = config.DEFAULT_EMBEDDING_MODEL) -> List[List[float]]:
    """Generate embeddings for a batch of texts with a prefix for better retrieval."""
    prefixed_texts = [prefix + text for text in texts]
    embed_model = OllamaEmbedding(model_name=embed_model_name)
    return embed_model.get_text_embedding_batch(prefixed_texts)  # Use batch processing

@st.cache_resource
def load_and_index_document(_file_path: str, document_key: str, embed_model_name: str = config.DEFAULT_EMBEDDING_MODEL):
    """Load and index a document using the appropriate text extraction method."""
    try:
        logger.info(f"Loading document: {_file_path}")
        progress_placeholder = st.empty()
        progress_placeholder.info("Extracting text...")

        # Extract text based on file type
        document_text = extract_text_from_file(_file_path)
        if not document_text or not document_text.strip():
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
            collection_name=config.DOCUMENT_COLLECTION_NAME
        )

        # Chunk the document text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust chunk size as needed
            chunk_overlap=200,  # Adjust overlap as needed
        )
        chunks = text_splitter.split_text(document_text)  # Split the single string into chunks

        # Generate embeddings for all chunks in a batch
        progress_placeholder.info("Generating embeddings...")
        embeddings = get_embedding_with_prefix(chunks, embed_model_name=embed_model_name)

        # Create TextNode objects with embeddings
        text_nodes = [
            TextNode(
                text=chunk,
                embedding=embedding,
                metadata={
                    "source": os.path.basename(_file_path),
                    "timestamp": str(uuid.uuid4()),
                    "document_key": document_key
                },
                id_=str(uuid4())
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        # Add the document chunks to the Qdrant collection
        vector_store.add(nodes=text_nodes)

        # Add to document list and clean up old ones
        doc_info = {
            "key": document_key,
            "name": os.path.basename(_file_path),
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

@lru_cache(maxsize=100)  # Cache up to 100 queries
def cached_llm_response(query: str, llm_model_name: str = config.DEFAULT_LLM_MODEL) -> str:
    """Cache LLM responses for common queries."""
    llm = initialize_llm(model_name=llm_model_name)
    return llm.complete(query).text

def summarize_answer(query: str, documents: List[str]) -> str:
    """Summarize the retrieved documents to provide a concise answer."""
    combined_text = "\n".join(documents)
    prompt = f"Summarize the following information to answer the query: {query}\n\n{combined_text}"
    return cached_llm_response(prompt)

@st.cache_resource
def load_query_engine_from_db(llm_model_name: str = config.DEFAULT_LLM_MODEL, 
                             embed_model_name: str = config.DEFAULT_EMBEDDING_MODEL,
                             custom_prompt: Optional[str] = None,
                             document_id: Optional[str] = None):
    """Load the query engine from the persistent Qdrant with vector search and reranking."""
    try:
        logger.info(f"Loading query engine from DB for document: {document_id if document_id else 'all'}")

        # Initialize Qdrant (reuse client if possible)
        qdrant_client = QdrantClient(
            host=config.QDRANT_DB_HOST, 
            port=config.QDRANT_DB_PORT, 
            timeout=config.QDRANT_TIMEOUT
        )

        # Check if collection exists
        collections = qdrant_client.get_collections()
        if config.DOCUMENT_COLLECTION_NAME not in [c.name for c in collections.collections]:
            st.error("No documents found in the database. Please upload documents first.")
            return None

        # Create vector store
        vector_store = QdrantVectorStore(
            client=qdrant_client, 
            collection_name=config.DOCUMENT_COLLECTION_NAME
        )

        # Configure embeddings and LLM (preload models if possible)
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

        # Create query engine with vector search
        query_engine = index.as_query_engine(
            streaming=True,
            similarity_top_k=5,  # Reduced from 10 to 5 for faster processing
            vector_store_query_mode="default"  # Use default vector search
        )

        # Add reranking step using mxbai-rerank-large-v1 via Ollama
        def rerank_query(query: str, documents: List[str]) -> List[str]:
            """Rerank documents using mxbai-rerank-large-v1 running via Ollama."""
            try:
                # Prepare the input for the reranking model
                rerank_input = {
                    "query": query,
                    "documents": documents
                }

                # Call the Ollama API for reranking
                rerank_response = requests.post(
                    config.RERANKING_API_ENDPOINT,  # Use config value
                    json=rerank_input,
                    timeout=10  # Reduced timeout from 30 to 10 seconds
                )
                rerank_response.raise_for_status()

                # Parse the reranked results
                reranked_documents = rerank_response.json().get("results", [])
                return reranked_documents
            except Exception as e:
                logger.error(f"Error during reranking: {str(e)}")
                return documents  # Fallback to original documents if reranking fails

        # Update the query engine to include summarization and "Thinking..." message
        def query_with_summary(query: str) -> str:
            """Query the engine and summarize the results with a 'Thinking...' message."""
            with st.status("Thinking...", expanded=False) as status:
                # Embed the question for similarity search
                query_embedding = get_embedding_with_prefix(
                    query, 
                    prefix="Represent this question for retrieval: ", 
                    embed_model_name=embed_model_name
                )
                results = query_engine.query(query)
                summarized_answer = summarize_answer(query, results, llm_model_name=llm_model_name)
                status.update(label="Response ready!", state="complete")
                return summarized_answer

        query_engine.query_with_summary = query_with_summary

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