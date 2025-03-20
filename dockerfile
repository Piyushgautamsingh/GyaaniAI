FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Install Docling and its dependencies
RUN pip install docling

# Install additional models for hybrid search (if needed)
# For example, if hybrid search requires a specific model, download it here
# RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('model-name')"

RUN mkdir -p /app/uploaded_files
RUN mkdir -p /tmp/gyaani_ai

EXPOSE 8501

ENV GYAANI_AI_TEMP_DIR=/tmp/gyaani_ai
ENV QDRANT_DB_HOST=qdrant
ENV QDRANT_DB_PORT=6333

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]