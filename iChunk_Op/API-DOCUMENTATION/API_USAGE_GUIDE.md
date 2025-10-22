# Chunking Optimizer API - Complete Usage Guide

**Version:** 2.0  
**Purpose:** Company Project Submission Documentation  
**Date:** January 2024

---

##  Executive Summary

The **Chunking Optimizer API** is an enterprise-grade system for processing, chunking, and vectorizing large datasets (up to 3GB+) with semantic search capabilities. It provides three processing modes (Fast, Config-1, Deep Config) tailored for different use cases, from rapid prototyping to advanced customization with metadata filtering.

### Key Capabilities
-  Process CSV files and databases (MySQL, PostgreSQL)
-  Handle files up to 3GB+ with streaming I/O
-  Multiple chunking strategies (Fixed, Recursive, Semantic, Document-based)
-  Semantic search with metadata filtering
-  OpenAI-compatible API endpoints
-  Performance optimizations (turbo mode, parallel processing)
-  Export capabilities (CSV, NumPy, JSON)

---

##  System Architecture

```
+--------------------------------------------------------------------+
|                         CLIENT LAYER                               |
|   Web UI (Streamlit)  |  Python SDK  |  cURL  |  REST Clients     |
+-----------------------------+-------------------------------------- +
                              | HTTP/REST
                              v
+--------------------------------------------------------------------+
|                      FASTAPI SERVER                                |
|                    (Port 8001 - main.py)                          |
|  +-------------+  +-------------+  +--------------+  +----------+ |
|  |  Fast Mode  |  |  Config-1   |  | Deep Config  |  |Universal | |
|  |  Endpoints  |  |  Endpoints  |  |  (8 Steps)   |  |Endpoint  | |
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

##  Processing Modes

### Mode 1: Fast Mode (Automatic)
**Best for:** Quick prototyping, automatic settings, simple workflows

**Features:**
- Automatic preprocessing (header normalization, text cleaning)
- Semantic clustering (KMeans, n=20)
- Default model: `paraphrase-MiniLM-L6-v2` (384-dim)
- FAISS storage with L2 distance
- Single API call

**Performance:** ~60 seconds for 100K rows

**Use Case:** Initial exploration, rapid development, demos

---

### Mode 2: Config-1 Mode (Configurable)
**Best for:** Custom chunking requirements, flexible embedding models

**Features:**
- Configurable chunking methods (fixed/recursive/semantic/document)
- Custom chunk size and overlap
- Multiple embedding models
- Storage choice (FAISS or ChromaDB)
- Retrieval metric selection (cosine/euclidean/dot)
- Optional default preprocessing

**Performance:** ~90 seconds for 100K rows

**Use Case:** Production applications with specific requirements

---

### Mode 3: Deep Config Mode (Advanced)
**Best for:** Maximum control, advanced preprocessing, metadata extraction

**Features:**
- 8-step preprocessing pipeline:
  1. Load & preprocess
  2. Type conversion
  3. Null handling (7 strategies)
  4. Duplicate removal
  5. Stopword removal
  6. Text normalization (lemmatization/stemming)
  7. Chunking with metadata extraction
  8. Embedding generation
  9. Storage with metadata indexing
- Statistical metadata (min/mean/max for numeric columns)
- Categorical metadata (mode for categorical columns)
- Metadata filtering in retrieval

**Performance:** ~120 seconds for 100K rows (all steps)

**Use Case:** Enterprise applications, data quality requirements, filtered search

---

##  Quick Reference Table

| Feature | Fast Mode | Config-1 | Deep Config |
|---------|-----------|----------|-------------|
| **Setup Complexity** | Low | Medium | High |
| **API Calls** | 1 | 1 | 9 |
| **Preprocessing** | Automatic | Optional | Customizable (8 steps) |
| **Chunking** | Semantic (auto) | 4 methods | 4 methods + metadata |
| **Model Selection** | Fixed | Flexible | Flexible |
| **Storage** | FAISS | FAISS/Chroma | FAISS/Chroma |
| **Metadata** | No | No | Yes (advanced) |
| **Filtered Retrieval** | No | No | Yes |
| **Performance** | Fastest | Medium | Slower (most features) |
| **Best For** | Prototyping | Production | Enterprise |

---

##  Workflow Examples

### Workflow 1: CSV File → Search (Fast Mode)

```bash
# 1. Process CSV file
curl -X POST http://127.0.0.1:8001/run_fast \
  -F "file=@data.csv" \
  -F "use_turbo=true"

# Response: {"mode": "fast", "summary": {"rows": 10000, "chunks": 150, ...}}

# 2. Search for similar content
curl -X POST http://127.0.0.1:8001/retrieve \
  -F "query=ergonomic office furniture" \
  -F "k=5"

# 3. Export results
curl http://127.0.0.1:8001/export/chunks --output chunks.csv
```

**Time:** 2-3 minutes for 100K rows

---

### Workflow 2: Database → Custom Chunking → Search (Config-1)

```bash
# 1. Test database connection
curl -X POST http://127.0.0.1:8001/db/test_connection \
  -F "db_type=mysql" \
  -F "host=localhost" \
  -F "port=3306" \
  -F "username=root" \
  -F "password=mypassword" \
  -F "database=mydb"

# 2. List available tables
curl -X POST http://127.0.0.1:8001/db/list_tables \
  -F "db_type=mysql" \
  -F "host=localhost" \
  -F "port=3306" \
  -F "username=root" \
  -F "password=mypassword" \
  -F "database=mydb"

# 3. Import and process table
curl -X POST http://127.0.0.1:8001/db/import_one \
  -F "db_type=mysql" \
  -F "host=localhost" \
  -F "port=3306" \
  -F "username=root" \
  -F "password=mypassword" \
  -F "database=mydb" \
  -F "table_name=products" \
  -F "processing_mode=config1" \
  -F "use_turbo=true"

# Alternative: Upload CSV with custom settings
curl -X POST http://127.0.0.1:8001/run_config1 \
  -F "file=@data.csv" \
  -F "chunk_method=document" \
  -F "document_key_column=category" \
  -F "token_limit=1500" \
  -F "model_choice=all-MiniLM-L6-v2" \
  -F "storage_choice=chroma" \
  -F "retrieval_metric=cosine" \
  -F "use_turbo=true"

# 4. Search
curl -X POST http://127.0.0.1:8001/retrieve \
  -F "query=modern electronics" \
  -F "k=10"
```

**Time:** 3-5 minutes for 100K rows

---

### Workflow 3: Advanced Processing with Metadata (Deep Config)

```bash
# Step 1: Preprocess
curl -X POST http://127.0.0.1:8001/deep_config/preprocess \
  -F "file=@data.csv"

# Step 2: Convert data types
curl -X POST http://127.0.0.1:8001/deep_config/type_convert \
  -F 'type_conversions={"id":"integer","price":"float","date_added":"datetime"}'

# Step 3: Handle nulls
curl -X POST http://127.0.0.1:8001/deep_config/null_handle \
  -F 'null_strategies={"price":"mean","description":"unknown"}'

# Step 4: Remove duplicates
curl -X POST http://127.0.0.1:8001/deep_config/duplicates \
  -F "strategy=keep_first"

# Step 5: Remove stopwords
curl -X POST http://127.0.0.1:8001/deep_config/stopwords \
  -F "remove_stopwords=true"

# Step 6: Text normalization
curl -X POST http://127.0.0.1:8001/deep_config/normalize \
  -F "text_processing=lemmatize"

# Step 7: Chunk with metadata
curl -X POST http://127.0.0.1:8001/deep_config/chunk \
  -F "chunk_method=document" \
  -F "key_column=category" \
  -F "token_limit=1500" \
  -F "store_metadata=true" \
  -F 'selected_numeric_columns=["price","rating"]' \
  -F 'selected_categorical_columns=["category","brand"]'

# Step 8: Generate embeddings
curl -X POST http://127.0.0.1:8001/deep_config/embed \
  -F "model_name=paraphrase-MiniLM-L6-v2" \
  -F "batch_size=128" \
  -F "use_parallel=true"

# Step 9: Store vectors
curl -X POST http://127.0.0.1:8001/deep_config/store \
  -F "storage_type=faiss" \
  -F "collection_name=my_vectors"

# Step 10: Retrieve with metadata filtering
curl -X POST http://127.0.0.1:8001/retrieve_with_metadata \
  -F "query=premium products" \
  -F "k=5" \
  -F 'metadata_filter={"category_mode":"Electronics","price_mean":500}'
```

**Time:** 4-6 minutes for 100K rows (all steps)

---

##  API Endpoints Summary

### Core Processing (3 modes)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/run_fast` | POST | Fast mode - automatic processing |
| `/run_config1` | POST | Config-1 mode - customizable |
| `/deep_config/preprocess` | POST | Deep Config Step 1 - Preprocess |
| `/deep_config/type_convert` | POST | Deep Config Step 2 - Type conversion |
| `/deep_config/null_handle` | POST | Deep Config Step 3 - Null handling |
| `/deep_config/duplicates` | POST | Deep Config Step 4 - Duplicates |
| `/deep_config/stopwords` | POST | Deep Config Step 5 - Stopwords |
| `/deep_config/normalize` | POST | Deep Config Step 6 - Normalization |
| `/deep_config/chunk` | POST | Deep Config Step 7 - Chunking |
| `/deep_config/embed` | POST | Deep Config Step 8 - Embedding |
| `/deep_config/store` | POST | Deep Config Step 9 - Storage |

### Retrieval (2 endpoints)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/retrieve` | POST | Basic semantic search |
| `/retrieve_with_metadata` | POST | Filtered semantic search |

### Database (3 endpoints)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/db/test_connection` | POST | Test database connectivity |
| `/db/list_tables` | POST | List database tables |
| `/db/import_one` | POST | Import and process table |

### Export (4 endpoints)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/export/chunks` | GET | Export chunks as CSV |
| `/export/embeddings` | GET | Export embeddings as .npy |
| `/export/embeddings_text` | GET | Export embeddings as JSON |
| `/export/preprocessed` | GET | Export preprocessed data CSV |

### System (4 endpoints)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Root - API information |
| `/health` | GET | Health check |
| `/system_info` | GET | System resource information |
| `/capabilities` | GET | API feature list |

### Universal Endpoint (1 endpoint - consolidates all)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/process` | POST | Universal endpoint - all operations |

**Total:** 24 dedicated endpoints + 1 universal endpoint

---

##  Configuration Parameters

### Common Parameters (All Modes)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | - | CSV file upload |
| `use_turbo` | Boolean | false | Enable parallel processing |
| `batch_size` | Integer | 256 | Embedding batch size |
| `process_large_files` | Boolean | true | Enable large file optimization |

### Fast Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| *(None)* | - | - | Uses all automatic settings |

### Config-1 Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_method` | String | "recursive" | fixed/recursive/semantic/document |
| `chunk_size` | Integer | 400 | Characters per chunk |
| `overlap` | Integer | 50 | Overlap between chunks |
| `n_clusters` | Integer | 10 | Clusters (semantic mode) |
| `document_key_column` | String | - | Key column (document mode) |
| `token_limit` | Integer | 2000 | Token limit (document mode) |
| `retrieval_metric` | String | "cosine" | cosine/euclidean/dot |
| `model_choice` | String | "paraphrase-MiniLM-L6-v2" | Embedding model |
| `storage_choice` | String | "faiss" | faiss/chroma |
| `apply_default_preprocessing` | Boolean | true | Apply text cleaning |

### Deep Config Parameters

| Step | Parameters | Description |
|------|------------|-------------|
| **Type Convert** | `type_conversions` (JSON) | Column type mappings |
| **Null Handle** | `null_strategies` (JSON) | Null handling per column |
| **Duplicates** | `strategy` (String) | keep_first/keep_last/remove_all |
| **Stopwords** | `remove_stopwords` (Boolean) | Enable stopword removal |
| **Normalize** | `text_processing` (String) | none/lemmatize/stem |
| **Chunk** | `chunk_method`, `store_metadata`, column selections | Chunking + metadata |
| **Embed** | `model_name`, `batch_size`, `use_parallel` | Embedding settings |
| **Store** | `storage_type`, `collection_name` | Storage configuration |

---

##  Performance Metrics

### Processing Time (by Dataset Size)

| Dataset Size | Fast Mode | Config-1 | Deep Config (All Steps) |
|--------------|-----------|----------|-------------------------|
| 1K rows | ~1s | ~2s | ~5s |
| 10K rows | ~5s | ~8s | ~15s |
| 100K rows | ~60s | ~90s | ~120s |
| 1M rows | ~10min | ~15min | ~20min |
| 3GB file | ~30min | ~45min | ~60min |

*With turbo mode enabled, default settings, 16GB RAM, 8-core CPU*

### Throughput

- **Fast Mode:** ~1,667 rows/second
- **Config-1:** ~1,111 rows/second
- **Deep Config:** ~833 rows/second

### Memory Usage

| File Size | Memory Peak | Recommended RAM |
|-----------|-------------|-----------------|
| 100MB | ~500MB | 4GB |
| 500MB | ~2GB | 8GB |
| 1GB | ~4GB | 8GB |
| 3GB | ~10GB | 16GB |

---

##  Security Considerations

### Current Implementation
-  **No Authentication**: API is completely open
-  **No Rate Limiting**: Unlimited requests
-  **No Request Size Limits**: Beyond 3GB recommendation
-  **No Input Validation**: Minimal sanitization

### Recommendations for Production
1. **Add Authentication**: API keys, JWT tokens, OAuth
2. **Implement Rate Limiting**: Prevent abuse
3. **Add Request Validation**: Sanitize inputs
4. **Enable HTTPS**: SSL/TLS encryption
5. **Add Logging**: Audit trail
6. **Implement CORS**: Cross-origin restrictions
7. **Add Monitoring**: Performance tracking

---

##  Error Handling

### Standard HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Missing parameters, invalid format |
| 404 | Not Found | Invalid endpoint |
| 500 | Internal Error | Processing failure, system error |

### Error Response Format

```json
{
  "error": "Detailed error message describing what went wrong"
}
```

### Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "No model available" | Retrieval before processing | Run processing first |
| "No data available" | Deep Config step before preprocess | Run preprocess first |
| "Column not found" | Invalid column name | Check column names |
| "File too large" | Memory exhaustion | Enable turbo, reduce batch size |
| "Connection refused" | Database unreachable | Check credentials, network |

---

##  Best Practices

### 1. Choose the Right Mode

```
Simple Use Case          → Fast Mode
Specific Requirements    → Config-1 Mode
Enterprise/Complex       → Deep Config Mode
```

### 2. Optimize Performance

-  Enable turbo mode for large files
-  Increase batch size if you have memory
-  Use semantic chunking for better grouping
-  Cache results for repeated queries

### 3. Data Preparation

-  Clean your data before upload
-  Use consistent encoding (UTF-8)
-  Remove unnecessary columns
-  Check for corrupt rows

### 4. Metadata Strategy

-  Select 3-5 key columns for metadata
-  Use low-cardinality categorical columns
-  Include relevant numeric columns
-  Test filters before production

### 5. Retrieval Optimization

-  Use metadata filters to reduce search space
-  Adjust k based on requirements (5-10 typical)
-  Cache frequently used queries
-  Monitor similarity scores

---

##  Complete Example: E-Commerce Product Search

```python
import requests

BASE_URL = "http://127.0.0.1:8001"

# ===== STEP 1: Process Product Catalog =====
print("Processing product catalog...")

url = f"{BASE_URL}/run_config1"
with open("products.csv", "rb") as f:
    files = {"file": f}
    data = {
        "chunk_method": "document",
        "document_key_column": "category",
        "token_limit": 1500,
        "model_choice": "all-MiniLM-L6-v2",
        "storage_choice": "faiss",
        "retrieval_metric": "cosine",
        "use_turbo": "true"
    }
    response = requests.post(url, files=files, data=data)
    result = response.json()

print(f" Processed {result['summary']['chunks']} product chunks")

# ===== STEP 2: Search Products =====
print("\nSearching for products...")

search_queries = [
    "ergonomic office chair",
    "wireless gaming mouse",
    "4K monitor for programming"
]

for query in search_queries:
    url = f"{BASE_URL}/retrieve"
    data = {"query": query, "k": 3}
    response = requests.post(url, data=data)
    results = response.json()
    
    print(f"\nQuery: '{query}'")
    for r in results['results']:
        print(f"  [{r['rank']}] {r['similarity']:.3f} - {r['content'][:80]}...")

# ===== STEP 3: Export for Analysis =====
print("\nExporting results...")

url = f"{BASE_URL}/export/chunks"
response = requests.get(url)
with open("product_chunks.csv", "wb") as f:
    f.write(response.content)

print(" Export complete: product_chunks.csv")
```

---

##  Learning Path

### Beginner (Day 1)
1.  Install dependencies
2.  Start API server
3.  Test health endpoint
4.  Process first CSV with Fast Mode
5.  Perform basic retrieval

### Intermediate (Day 2-3)
1.  Try Config-1 Mode with different chunking methods
2.  Test database import
3.  Experiment with metadata filtering
4.  Export and analyze results

### Advanced (Day 4-7)
1.  Use Deep Config Mode with all steps
2.  Optimize metadata column selection
3.  Build production integration
4.  Implement caching strategy
5.  Monitor performance

---

##  Support & Resources

### Documentation
- **Installation Guide**: `API-DOCUMENTATION/01-GETTING-STARTED/installation.md`
- **Quick Start**: `API-DOCUMENTATION/01-GETTING-STARTED/quickstart.md`
- **API Reference**: `API-DOCUMENTATION/03-API-REFERENCE/core-endpoints.md`
- **Python Examples**: `API-DOCUMENTATION/05-EXAMPLES/python-examples.md`
- **Architecture**: `API-DOCUMENTATION/02-ARCHITECTURE/data-flow.md`

### Test Data
- Sample datasets in project root: `TEST_DATASETS_README.md`
- Three comprehensive test files included
- Covers all preprocessing scenarios

### System Info
- Check: `curl http://127.0.0.1:8001/system_info`
- Health: `curl http://127.0.0.1:8001/health`
- Capabilities: `curl http://127.0.0.1:8001/capabilities`

---

##  Project Submission Checklist

- [ ] API server running successfully
- [ ] All endpoints tested and documented
- [ ] Sample data processed without errors
- [ ] Retrieval returns relevant results
- [ ] Export functions working
- [ ] Performance metrics collected
- [ ] Documentation complete and clear
- [ ] Code comments and structure reviewed
- [ ] Error handling tested
- [ ] System requirements documented

---

##  License & Credits

**Version:** 2.0  
**Last Updated:** January 2024  
**Maintained By:** [Your Name/Organization]

---

**End of API Usage Guide**

For detailed technical documentation, see the `API-DOCUMENTATION/` directory.

