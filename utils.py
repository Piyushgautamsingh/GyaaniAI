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
import datetime

# Import configuration
import config

from docx import Document as DocxDocument

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), 
                   format=config.LOG_FORMAT)
logger = logging.getLogger(config.LOGGER_NAME)

def cleanup_old_indices():
    """Clean up collections older than 3 days."""
    try:
        qdrant_client = QdrantClient(
            host=config.QDRANT_DB_HOST, 
            port=config.QDRANT_DB_PORT, 
            timeout=config.QDRANT_TIMEOUT
        )
        
        # Get all collections
        collections = qdrant_client.get_collections()
        today = datetime.date.today()
        cutoff_date = today - datetime.timedelta(days=config.DATA_RETENTION_DAYS)
        
        # Find collections older than cutoff date
        for collection in collections.collections:
            try:
                # Extract date from collection name (format: documents_YYYY-MM-DD)
                if collection.name.startswith("documents_"):
                    collection_date_str = collection.name.split("_")[1]
                    collection_date = datetime.datetime.strptime(collection_date_str, "%Y-%m-%d").date()
                    if collection_date < cutoff_date:
                        qdrant_client.delete_collection(collection.name)
                        logger.info(f"Deleted old collection: {collection.name}")
            except (ValueError, IndexError):
                continue
                
    except Exception as e:
        logger.error(f"Error cleaning up old collections: {str(e)}")

async def scrape_web_page_with_children(url: str, max_depth: int = 2) -> str:
    """Scrape content from a web page and its children up to max_depth levels."""
    try:
        visited_urls = set()  # Track visited URLs to prevent cycles
        return await _scrape_recursive(url, max_depth, 0, visited_urls)
    except Exception as e:
        logger.error(f"Error scraping web page {url}: {str(e)}")
        return None

async def _scrape_recursive(url: str, max_depth: int, current_depth: int, visited_urls: set) -> str:
    """Recursive helper function for web scraping with cycle prevention."""
    if current_depth > max_depth or url in visited_urls:
        return ""
    
    visited_urls.add(url)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=config.REQUEST_TIMEOUT) as response:
                response.raise_for_status()
                
                # Check if this is actually HTML content
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('text/html'):
                    return ""
                    
                html = await response.text()
                
                soup = BeautifulSoup(html, "html.parser")
                
                # Extract main content
                text = ""
                for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
                    text += element.get_text() + "\n"
                
                # If we haven't reached max depth, look for child links
                if current_depth < max_depth:
                    base_url = "{0.scheme}://{0.netloc}".format(urlparse(url))
                    links = set()
                    
                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        if not href or href.startswith('#'):
                            continue
                            
                        # Handle relative URLs
                        if href.startswith("/"):
                            href = base_url + href
                        elif not href.startswith("http"):
                            href = urljoin(base_url, href)
                        
                        # Validate the URL before adding
                        try:
                            parsed_href = urlparse(href)
                            if (parsed_href.netloc == urlparse(url).netloc 
                                and validate_url(href)
                                and href not in visited_urls):
                                links.add(href)
                        except:
                            continue
                    
                    # Scrape child pages with limited concurrency
                    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
                    async def limited_scrape(link):
                        async with semaphore:
                            return await _scrape_recursive(
                                link, max_depth, current_depth + 1, visited_urls
                            )
                    
                    child_texts = await asyncio.gather(
                        *[limited_scrape(link) for link in list(links)[:config.MAX_CHILD_PAGES]],
                        return_exceptions=True  # Prevent one failure from stopping all
                    )
                    
                    # Filter out any exceptions that occurred
                    valid_texts = [
                        t for t in child_texts 
                        if isinstance(t, str) and t.strip()
                    ]
                    text += "\n".join(valid_texts)
                
                return text.strip()
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return ""

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

        # Delete all collections from Qdrant
        qdrant_client = QdrantClient(
            host=config.QDRANT_DB_HOST, 
            port=config.QDRANT_DB_PORT, 
            timeout=config.QDRANT_TIMEOUT
        )
        collections = qdrant_client.get_collections()
        for collection in collections.collections:
            if collection.name.startswith("documents_"):
                try:
                    qdrant_client.delete_collection(collection.name)
                    logger.info(f"Deleted collection: {collection.name}")
                except Exception as e:
                    logger.error(f"Error deleting collection {collection.name}: {str(e)}")

        st.toast("All resources cleared successfully!", icon="✅")
        time.sleep(1)  # Give time for toast to appear
        st.rerun()  # Use st.rerun() instead of experimental_rerun()
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
    """Validate if the URL is well-formed and has a valid scheme."""
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        # Check for valid schemes
        if result.scheme not in ['http', 'https', 'ftp']:
            return False
        # Basic domain validation
        if '.' not in result.netloc:
            return False
        return True
    except Exception as e:
        logger.error(f"URL validation error: {str(e)}")
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
    """Download a document from a URL asynchronously or scrape a web page with improved error handling and recursion limits."""
    try:
        if not validate_url(url):
            st.error("Invalid URL format. Please include http:// or https:// and a valid domain.")
            return None

        # Set conservative recursion limit for this operation
        import sys
        original_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100)  # Set safe recursion limit
        
        try:
            parsed = urlparse(url)
            logger.info(f"Processing URL: {url}")
            
            # First try to determine if this is a downloadable file
            is_downloadable = any(
                parsed.path.lower().endswith(ext) 
                for ext in config.ALLOWED_FILE_TYPES
            )

            if not is_downloadable:
                # If not obviously a file, check content type via HEAD request
                async with aiohttp.ClientSession() as session:
                    async with session.head(url, timeout=config.REQUEST_TIMEOUT) as response:
                        response.raise_for_status()
                        content_type = response.headers.get('Content-Type', '').lower()
                        is_downloadable = any(
                            ct in content_type 
                            for ct in ['pdf', 'msword', 'wordprocessing', 'octet-stream']
                        )

            if is_downloadable:
                # Handle file download
                async with aiohttp.ClientSession() as session:
                    # GET request to download the file
                    async with session.get(url, timeout=config.REQUEST_TIMEOUT) as response:
                        response.raise_for_status()
                        
                        # Get filename from Content-Disposition or URL path
                        content_disposition = response.headers.get('Content-Disposition', '')
                        if 'filename=' in content_disposition:
                            filename = content_disposition.split('filename=')[1].strip('"\'')
                        else:
                            filename = os.path.basename(parsed.path) or "downloaded_file"
                        
                        # Clean filename
                        filename = "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
                        if not filename:
                            filename = "downloaded_file"
                        
                        # Add extension if missing
                        if not os.path.splitext(filename)[1]:
                            content_type = response.headers.get('Content-Type', '').lower()
                            if 'pdf' in content_type:
                                filename += '.pdf'
                            elif 'msword' in content_type:
                                filename += '.doc'
                            elif 'wordprocessing' in content_type:
                                filename += '.docx'
                            else:
                                filename += '.txt'

                        file_path = os.path.join(temp_dir, filename)
                        with open(file_path, "wb") as f:
                            async for chunk in response.content.iter_chunked(1024*1024):  # 1MB chunks
                                f.write(chunk)
                        
                        logger.info(f"Successfully downloaded file from {url}")
                        return file_path
            else:
                # Treat as web page to scrape
                text = await scrape_web_page_with_children(url)
                if not text:
                    st.error("Failed to scrape content from the web page.")
                    return None
                
                # Save the scraped content as a text file
                filename = f"webpage_{parsed.netloc}.txt"
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(text)
                
                logger.info(f"Successfully scraped content from {url}")
                return file_path
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout while processing URL: {url}")
            st.error("The request timed out. Please try again or check the URL.")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"Client error while processing URL {url}: {str(e)}")
            st.error(f"Error accessing the URL: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing URL {url}: {str(e)}")
            st.error(f"An unexpected error occurred: {str(e)}")
            return None
        finally:
            # Restore original recursion limit
            sys.setrecursionlimit(original_recursion_limit)

    except Exception as e:
        logger.error(f"Error in URL processing pipeline: {str(e)}")
        st.error(f"Error processing URL: {str(e)}")
        return None

    except Exception as e:
        logger.error(f"Download error for URL {url}: {str(e)}")
        st.error(f"Error downloading from URL: {str(e)}")
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
        collection_name = config.get_collection_name()
        qdrant_client = QdrantClient(
            host=config.QDRANT_DB_HOST, 
            port=config.QDRANT_DB_PORT, 
            timeout=config.QDRANT_TIMEOUT
        )

        # Get embedding dimensions for the model
        vector_size = config.get_embedding_dimensions(model_name)

        # Check if collection exists
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            existing_size = collection_info.config.params.vectors.size
            
            if existing_size != vector_size:
                logger.warning(
                    f"Vector dimension mismatch: Collection has {existing_size}, model needs {vector_size}. "
                    "Recreating collection."
                )
                qdrant_client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                logger.info(f"Recreated collection with new dimension {vector_size}.")
                
        except Exception as e:
            # Collection doesn't exist, create it
            logger.info(f"Creating new collection: {collection_name}")
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

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
        file_obj = Path(_file_path)
        source_url = getattr(file_obj, 'source_url', '') if hasattr(file_obj, 'source_url') else ''
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
    """Get list of documents from all collections within retention period."""
    try:
        qdrant_client = QdrantClient(
            host=config.QDRANT_DB_HOST, 
            port=config.QDRANT_DB_PORT, 
            timeout=config.QDRANT_TIMEOUT
        )
        
        today = datetime.date.today()
        cutoff_date = today - datetime.timedelta(days=config.DATA_RETENTION_DAYS)
        documents = []
        
        # Check all collections within retention period
        collections = qdrant_client.get_collections()
        for collection in collections.collections:
            try:
                if collection.name.startswith("documents_"):
                    collection_date_str = collection.name.split("_")[1]
                    collection_date = datetime.datetime.strptime(collection_date_str, "%Y-%m-%d").date()
                    if collection_date >= cutoff_date:
                        result = qdrant_client.scroll(
                            collection_name=collection.name,
                            with_payload=True,
                            limit=config.MAX_DOCUMENTS_PER_SESSION
                        )
                        for item in result[0]:
                            payload = item.payload or {}
                            documents.append({
                                "key": item.id,
                                "name": payload.get("source", "Unknown document"),
                                "timestamp": payload.get("timestamp", ""),
                                "collection": collection.name
                            })
            except (ValueError, IndexError):
                continue
        
        return documents
    except Exception as e:
        logger.error(f"Error getting document list: {str(e)}")
        return []
def select_active_document(document_key: str) -> None:
    """Select a document as the active one for querying."""
    st.session_state.document_key = document_key
    st.session_state.scanned = True
    st.success(f"Selected document as active")