# GyaaniAI 🧠

GyaaniAI is an AI-powered document assistant designed to help users interact with and extract knowledge from various document formats. It leverages advanced language models and vector databases to provide accurate, context-aware answers to user queries.

## Features

- **Document Processing**: Supports multiple file types including PDF, HTML, DOCX, TXT, and Markdown.
- **Web Page Scraping**: Extracts content from web pages for analysis.
- **AI-Powered Chat**: Interact with your documents using a conversational interface.
- **Vector Database**: Utilizes Qdrant for efficient document indexing and retrieval.
- **Customizable Prompts**: Tailor the AI's responses with custom prompt templates.
- **Memory Management**: Monitors system memory usage and provides warnings when usage is high.

## Installation

### Prerequisites

- Docker
- Python 3.8 or higher

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/gyaaniai.git
   cd gyaaniai
   ```

2. **Set Up Environment Variables**:
   Create a `.env` file in the root directory with the following content:
   ```bash
   QDRANT_DB_HOST=localhost
   QDRANT_DB_PORT=6333
   QDRANT_TIMEOUT=10
   GYAANI_AI_TEMP_DIR=/tmp/gyaani_ai
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Docker Compose**:
   ```bash
   docker-compose up -d
   ```

5. **Run the Application**:
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Upload Documents**:
   - Use the sidebar to upload documents or enter a URL to a document or web page.
   - Supported file types: PDF, HTML, DOCX, TXT, and Markdown.

2. **Chat with Your Document**:
   - Once a document is processed, you can start asking questions in the chat interface.
   - The AI will provide answers based on the content of the document.

3. **Manage Documents**:
   - View, select, and delete documents using the document management section in the sidebar.

4. **Customize Settings**:
   - Adjust the LLM model, embedding model, and prompt template in the settings section.

## Configuration

The application can be configured using the `config.py` file. Key settings include:

- **Allowed File Types**: Modify `ALLOWED_FILE_TYPES` to add or remove supported file types.
- **Database Settings**: Adjust `QDRANT_DB_HOST`, `QDRANT_DB_PORT`, and `QDRANT_TIMEOUT` for database connectivity.
- **Model Settings**: Change `DEFAULT_LLM_MODEL` and `DEFAULT_EMBEDDING_MODEL` to use different AI models.

## License

GyaaniAI is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.


## Acknowledgments

- **Qdrant**: For providing an efficient vector database.
- **Ollama**: For enabling local model management and inference.
- **Streamlit**: For the interactive web interface.

## Contact

For any questions or issues, please open an issue on GitHub or contact the maintainers directly.

---

**GyaaniAI - Where Knowledge Meets AI**  
© Piyush 2025