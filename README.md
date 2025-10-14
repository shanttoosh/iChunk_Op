# Chunking Optimizer API v2.0

**Enterprise-Grade Data Processing & Vector Search System**

A powerful FastAPI-based system for processing large datasets (3GB+), intelligent text chunking, embedding generation, and semantic search with metadata filtering.

---

## Features

### Core Capabilities
- **4 Processing Modes**: Fast, Config-1, Deep Config (9-step pipeline), and Campaign Mode
- **Large File Support**: Handle CSV files up to 3GB+ with streaming I/O
- **Multiple Chunking Strategies**: Fixed, Recursive, Semantic, Document-based, Record-based, Company-based, Source-based
- **Vector Storage**: FAISS and ChromaDB support
- **Metadata Filtering**: Advanced filtered search with statistical aggregations
- **Database Integration**: MySQL and PostgreSQL import
- **OpenAI Compatible**: Drop-in replacement for OpenAI embeddings API
- **Export Options**: CSV, NumPy (.npy), JSON formats
- **Campaign Features**: Smart company retrieval, field detection, contextual display

### Performance
- **Fast Mode**: ~60 seconds for 100K rows
- **Parallel Processing**: 6 workers for embedding generation
- **Batch Processing**: 2K row batches for memory efficiency
- **Turbo Mode**: Optimized processing with parallel execution

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd copy-streamlitchunking

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (for advanced text processing)
python -m spacy download en_core_web_sm
```

### 2. Start the Server

```bash
# Start FastAPI server
python main.py

# Server will run on http://127.0.0.1:8001
```

### 3. Test the API

```bash
# Health check
curl http://127.0.0.1:8001/health

# System info
curl http://127.0.0.1:8001/system_info

# View capabilities
curl http://127.0.0.1:8001/capabilities
```

### 4. Process Your First File

```bash
# Fast Mode (simplest)
curl -X POST http://127.0.0.1:8001/run_fast \
  -F "file=@your_data.csv" \
  -F "use_turbo=true"

# Search
curl -X POST http://127.0.0.1:8001/retrieve \
  -F "query=your search query" \
  -F "k=5"

# Export results
curl http://127.0.0.1:8001/export/chunks --output results.csv
```

---

## Project Structure

```
copy-streamlitchunking/
│
├── main.py                      # FastAPI server (30+ endpoints)
├── backend.py                   # Processing engine (core logic)
├── backend_campaign.py          # Campaign mode processing engine
├── app.py                       # Streamlit UI (4 processing modes)
├── requirements.txt             # Python dependencies
├── current_state.pkl            # State persistence file
│
├── API-DOCUMENTATION/           # Complete API documentation
│   ├── 00-START-HERE.md        # Navigation guide
│   ├── API_USAGE_GUIDE.md      # Complete API reference
│   ├── API_VISUAL_GUIDE.md     # Visual diagrams & workflows
│   ├── PROJECT_SUBMISSION_SUMMARY.md
│   ├── DOCUMENTATION_INDEX.md
│   ├── installation.md
│   ├── quickstart.md
│   ├── core-endpoints.md       # All 24+ endpoints documented
│   ├── data-flow.md            # Architecture diagrams
│   ├── python-examples.md      # Integration examples
│   └── README.md
│
└── TEST_DATASETS_README.md     # Test data guide
```

---

## Architecture

```
+--------------------------------------------------------------------+
|                         CLIENT LAYER                               |
|   Web UI (Streamlit)  |  Python SDK  |  cURL  |  REST Clients     |
+-----------------------------+--------------------------------------+
                              | HTTP/REST
                              v
+--------------------------------------------------------------------+
|                      FASTAPI SERVER                                |
|                    (Port 8001 - main.py)                          |
|  +-------------+  +-------------+  +--------------+  +----------+ |
|  |  Fast Mode  |  |  Config-1   |  | Deep Config  |  |Campaign  | |
|  |  Endpoints  |  |  Endpoints  |  |  (8 Steps)   |  |  Mode    | |
|  +-------------+  +-------------+  +--------------+  +----------+ |
+-----------------------------+--------------------------------------+
                              |
                              v
+--------------------------------------------------------------------+
|                     PROCESSING ENGINE                              |
|                      (backend.py)                                  |
|  +------------+  +-----------+  +------------+  +---------------+ |
|  |Preprocess  |  | Chunking  |  | Embedding  |  |   Storage     | |
|  |  Pipeline  |  |Strategies |  |Generation  |  | & Retrieval   | |
|  +------------+  +-----------+  +------------+  +---------------+ |
+-----------------------------+--------------------------------------+
                              |
         +--------------------+--------------------+
         |                    |                    |
         v                    v                    v
  +--------------+     +-------------+      +----------------+
  | DATA SOURCES |     |  VECTOR DBs |      | EXPORT FORMATS |
  | • CSV (3GB+) |     |  • FAISS    |      |  • CSV         |
  | • MySQL      |     |  • ChromaDB |      |  • NumPy (.npy)|
  | • PostgreSQL |     |  • Metadata |      |  • JSON        |
  +--------------+     +-------------+      +----------------+
```

---

## Processing Modes

### Fast Mode (Automatic)
**Best for:** Quick prototyping, demos, exploration

```bash
curl -X POST http://127.0.0.1:8001/run_fast \
  -F "file=@data.csv" \
  -F "use_turbo=true"
```

- 1 API call
- Automatic preprocessing
- Semantic clustering (KMeans)
- ~60s for 100K rows

### Config-1 Mode (Configurable)
**Best for:** Production apps, custom requirements

```bash
curl -X POST http://127.0.0.1:8001/run_config1 \
  -F "file=@data.csv" \
  -F "chunk_method=document" \
  -F "model_choice=all-MiniLM-L6-v2" \
  -F "storage_choice=faiss"
```

- Custom chunking strategies
- Model selection
- Storage choice
- ~90s for 100K rows

### Deep Config Mode (Advanced)
**Best for:** Enterprise, data quality, metadata filtering

9-step pipeline:
1. Preprocess → 2. Type Convert → 3. Null Handle → 4. Duplicates → 
5. Stopwords → 6. Normalize → 7. Chunk → 8. Embed → 9. Store

- Maximum control
- Metadata extraction
- Filtered search
- ~120s for 100K rows

### Campaign Mode (Specialized)
**Best for:** Media campaigns, contact data, lead management

```bash
curl -X POST http://127.0.0.1:8001/campaign/run \
  -F "file=@campaign_data.csv" \
  -F "chunk_method=company_based" \
  -F "model_choice=paraphrase-MiniLM-L6-v2"
```

- Smart company retrieval (2-stage matching)
- Field detection and mapping
- 5 specialized chunking methods
- Complete record preservation
- Contextual column display
- ~75s for 100K rows

---

## Key Features

### 1. Intelligent Chunking
- **Fixed**: Uniform size with overlap
- **Recursive**: Key-value format preservation
- **Semantic**: KMeans clustering for similar content
- **Document**: Group by entity/category
- **Record-based**: Group contact records by count
- **Company-based**: Group by company name
- **Source-based**: Group by lead source

### 2. Metadata System
- Smart column selection (cardinality filtering)
- Statistical aggregations (mean, median, std)
- Categorical mode values
- Fast indexed search

### 3. Advanced Preprocessing
- Header normalization
- HTML tag removal
- Type conversion (7 types)
- Null handling (7 strategies)
- Duplicate removal (4 strategies)
- Stopword removal (spaCy/basic)
- Text normalization (lemmatize/stem)

### 4. Vector Search
- FAISS: Fast similarity search with L2 distance
- ChromaDB: Persistent storage with multiple metrics
- Metadata filtering
- Top-K retrieval with similarity scores

---

## API Endpoints

### Processing (20 endpoints)
- `POST /run_fast` - Fast mode processing
- `POST /run_config1` - Configurable processing
- `POST /deep_config/*` - 9 deep config steps
- `POST /campaign/run` - Campaign mode processing
- `POST /campaign/retrieve` - Campaign retrieval
- `POST /campaign/smart_retrieval` - Smart company retrieval

### Retrieval (3 endpoints)
- `POST /retrieve` - Basic search
- `POST /retrieve_with_metadata` - Filtered search
- `POST /v1/retrieve` - OpenAI-compatible

### Database (3 endpoints)
- `POST /db/test_connection` - Test database
- `POST /db/list_tables` - List tables
- `POST /db/import_one` - Import & process

### Export (9 endpoints)
- `GET /export/chunks` - Export chunks as CSV
- `GET /export/embeddings` - Export as NumPy
- `GET /export/embeddings_text` - Export as JSON
- `GET /campaign/export/chunks` - Export campaign chunks
- `GET /campaign/export/embeddings` - Export campaign embeddings
- `GET /campaign/export/preprocessed` - Export campaign preprocessed data
- And more...

### System (4 endpoints)
- `GET /` - API info
- `GET /health` - Health check
- `GET /system_info` - Resource usage
- `GET /capabilities` - Feature list

**Total: 40+ endpoints**

---

## Configuration

### Environment Variables (Optional)

```bash
# OpenAI API (optional)
export OPENAI_API_KEY="your-api-key"

# Custom OpenAI base URL (optional)
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Database connections (optional)
export MYSQL_HOST="localhost"
export MYSQL_USER="user"
export MYSQL_PASSWORD="password"
export POSTGRES_HOST="localhost"
```

### Models Available

**Local Models (No API key needed):**
- `paraphrase-MiniLM-L6-v2` (384 dims, fast)
- `all-MiniLM-L6-v2` (384 dims, balanced)
- `paraphrase-mpnet-base-v2` (768 dims, best quality)

**OpenAI Models (API key required):**
- `text-embedding-ada-002` (1536 dims)

---

## Python Integration

```python
import requests

# Process file
with open('data.csv', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:8001/run_fast',
        files={'file': f},
        data={'use_turbo': 'true'}
    )

# Search
response = requests.post(
    'http://127.0.0.1:8001/retrieve',
    data={
        'query': 'your search query',
        'k': 5
    }
)

results = response.json()
for item in results['results']:
    print(f"Rank {item['rank']}: {item['content']}")
    print(f"Similarity: {item['similarity']:.3f}\n")
```

---

## Performance Benchmarks

| Dataset Size | Fast Mode | Config-1 | Deep Config | Campaign Mode |
|-------------|-----------|----------|-------------|---------------|
| 10K rows    | ~6s       | ~9s      | ~12s        | ~7s           |
| 100K rows   | ~60s      | ~90s     | ~120s       | ~75s          |
| 1M rows     | ~10min    | ~15min   | ~20min      | ~12min        |

**Memory Usage:**
- 1GB file: ~2-3GB RAM
- 3GB file: ~4-6GB RAM

---

## Testing

### Test Datasets Included
See `TEST_DATASETS_README.md` for sample datasets covering:
- Basic CSV processing
- Special characters & encoding
- Type conversion scenarios
- Null handling cases
- Duplicate detection
- Large file handling

### Run Tests

```bash
# Test health endpoint
curl http://127.0.0.1:8001/health

# Test with sample data
curl -X POST http://127.0.0.1:8001/run_fast \
  -F "file=@test_data.csv"
```

---

## Documentation

**Start Here:**
1. `API-DOCUMENTATION/00-START-HERE.md` - Navigation guide
2. `API-DOCUMENTATION/API_USAGE_GUIDE.md` - Complete reference
3. `API-DOCUMENTATION/API_VISUAL_GUIDE.md` - Visual diagrams

**For Developers:**
- `API-DOCUMENTATION/installation.md` - Setup guide
- `API-DOCUMENTATION/quickstart.md` - 5-minute tutorial
- `API-DOCUMENTATION/python-examples.md` - Code examples

**For Reference:**
- `API-DOCUMENTATION/core-endpoints.md` - All endpoints
- `API-DOCUMENTATION/data-flow.md` - Architecture
- `API-DOCUMENTATION/DOCUMENTATION_INDEX.md` - Complete index

---

## Troubleshooting

### Common Issues

**1. Import errors**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**2. Port already in use**
```bash
# Change port in main.py
uvicorn.run(app, host="127.0.0.1", port=8002)
```

**3. Out of memory**
- Reduce batch size in backend.py
- Use turbo mode for optimization
- Process smaller files

**4. Slow processing**
- Enable `use_turbo=true`
- Use Fast Mode for initial testing
- Check system resources with `/system_info`

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## License

[Specify your license here]

---

## Support

For questions, issues, or feature requests:
- Check documentation in `API-DOCUMENTATION/`
- Review `TEST_DATASETS_README.md` for examples
- Open an issue on GitHub

---

## Version History

**v3.0** (Current)
- 4 processing modes (including Campaign Mode)
- 40+ API endpoints
- Smart company retrieval
- Campaign-specific features
- Metadata filtering
- OpenAI compatibility
- Large file support (3GB+)
- Complete documentation

---

## Acknowledgments

Built with:
- FastAPI - Web framework
- Sentence Transformers - Embeddings
- FAISS - Vector search
- ChromaDB - Vector database
- Pandas - Data processing
- Streamlit - Web UI

---

**Ready for Production | Enterprise-Grade | Fully Documented**

