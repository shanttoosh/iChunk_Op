# Installation Guide

##  Prerequisites

Before installing the Chunking Optimizer API, ensure you have the following:

- **Python**: Version 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 8GB RAM (16GB recommended for large files)
- **Disk Space**: 5GB free space for dependencies and cache
- **Optional**: CUDA-enabled GPU for faster embedding generation

##  System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 5GB
- **Network**: Internet connection for downloading models

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 20GB SSD
- **GPU**: CUDA-compatible (optional)

##  Installation Steps

### Step 1: Clone Repository

```bash
# Clone the repository
git clone <repository-url>
cd chunking-optimizer

# Verify files
ls -la
# Should see: app.py, backend.py, main.py, requirements.txt
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# This will install 72+ packages including:
# - FastAPI, Streamlit
# - pandas, numpy, scikit-learn
# - sentence-transformers, transformers
# - chromadb, faiss-cpu
# - spaCy, nltk, langchain
```

**Installation Time**: 5-10 minutes depending on your internet speed

### Step 4: Download NLP Models

```bash
# Download spaCy model for text processing
python -m spacy download en_core_web_sm

# Download NLTK data (optional, for advanced text processing)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 5: Configure Database Connectors (Optional)

If you plan to use database import features:

```bash
# For MySQL
pip install mysql-connector-python

# For PostgreSQL
pip install psycopg2-binary

# Already included in requirements.txt
```

### Step 6: Verify Installation

```bash
# Test Python imports
python -c "import fastapi, pandas, chromadb, faiss; print(' All core packages installed')"

# Test spaCy
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print(' spaCy model loaded')"
```

##  Starting the API Server

### Option 1: Direct Start

```bash
# Start the API server
python main.py

# Output should show:
# INFO:     Started server process
# INFO:     Uvicorn running on http://127.0.0.1:8001
```

### Option 2: Custom Configuration

```bash
# Start with custom host and port
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Production mode (no auto-reload)
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
```

### Option 3: Start Streamlit UI (Optional)

```bash
# In a separate terminal
streamlit run app.py

# Output should show:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
```

##  Verify Installation

### Test 1: Health Check

```bash
curl http://127.0.0.1:8001/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "large_file_support": true,
  "performance_optimized": true
}
```

### Test 2: System Info

```bash
curl http://127.0.0.1:8001/system_info
```

**Expected Response:**
```json
{
  "memory_usage": "45.2%",
  "available_memory": "8.76 GB",
  "total_memory": "16.00 GB",
  "large_file_support": true,
  "max_recommended_file_size": "3GB+",
  "embedding_batch_size": 256,
  "parallel_workers": 6
}
```

### Test 3: Capabilities

```bash
curl http://127.0.0.1:8001/capabilities
```

**Expected Response:**
```json
{
  "openai_compatible_endpoints": [
    "/v1/embeddings",
    "/v1/chat/completions",
    "/v1/retrieve"
  ],
  "large_file_support": true,
  "max_file_size_recommendation": "3GB+",
  "supported_embedding_models": [
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L6-v2",
    "text-embedding-ada-002"
  ],
  "batch_processing": true,
  "memory_optimized": true
}
```

##  Troubleshooting

### Issue 1: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install specific package
pip install <package-name>
```

### Issue 2: FAISS Installation Issues

**Problem**: FAISS fails to install on Windows

**Solution**:
```bash
# Use conda for FAISS on Windows
conda install -c conda-forge faiss-cpu

# Or use pre-built wheels
pip install faiss-cpu --no-cache-dir
```

### Issue 3: spaCy Model Not Found

**Problem**: `OSError: [E050] Can't find model 'en_core_web_sm'`

**Solution**:
```bash
# Download model again
python -m spacy download en_core_web_sm

# Or install directly
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl
```

### Issue 4: Port Already in Use

**Problem**: `OSError: [Errno 98] Address already in use`

**Solution**:
```bash
# Find process using port 8001
# On Linux/macOS:
lsof -i :8001

# On Windows:
netstat -ano | findstr :8001

# Kill the process or use different port
uvicorn main:app --port 8002
```

### Issue 5: Memory Issues

**Problem**: `MemoryError` or system slowdown

**Solution**:
```bash
# Reduce batch size in configuration
export EMBEDDING_BATCH_SIZE=128
export BATCH_SIZE=1000

# Or modify backend.py:
# BATCH_SIZE = 1000
# EMBEDDING_BATCH_SIZE = 128
```

##  Docker Installation (Alternative)

If you prefer containerized deployment:

```dockerfile
# Dockerfile (create this file)
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

```bash
# Build Docker image
docker build -t chunking-optimizer:v2.0 .

# Run container
docker run -p 8001:8001 -v $(pwd)/data:/app/data chunking-optimizer:v2.0
```

##  Configuration Files

### Environment Variables (Optional)

Create `.env` file:

```bash
# API Configuration
API_HOST=127.0.0.1
API_PORT=8001

# Performance Settings
BATCH_SIZE=2000
EMBEDDING_BATCH_SIZE=256
PARALLEL_WORKERS=6

# Storage Paths
CACHE_DIR=./processing_cache
FAISS_STORE=./faiss_store
CHROMA_STORE=./chromadb_store

# Database (Optional)
MYSQL_HOST=localhost
MYSQL_PORT=3306
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# OpenAI (Optional)
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

##  Next Steps

Once installation is complete:

1.  **Test the API**: Follow the [Quick Start Guide](./quickstart.md)
2.  **Read API Docs**: Check [API Reference](../03-API-REFERENCE/core-endpoints.md)
3.  **Try Examples**: Explore [Code Examples](../05-EXAMPLES/python-examples.md)
4.  **Process Data**: Start with [Fast Mode Workflow](../04-WORKFLOWS/fast-mode-workflow.md)

##  Support

If you encounter issues during installation:

1. Check the [Troubleshooting Guide](../08-APPENDIX/troubleshooting.md)
2. Review the [FAQ](../08-APPENDIX/faq.md)
3. Consult system requirements above
4. Contact support team

---

**Installation Complete! ** You're ready to start processing and chunking data.

