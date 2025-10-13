# Core API Endpoints Reference

##  Table of Contents

1. [Processing Endpoints](#processing-endpoints)
2. [Retrieval Endpoints](#retrieval-endpoints)
3. [Export Endpoints](#export-endpoints)
4. [System Endpoints](#system-endpoints)
5. [Database Endpoints](#database-endpoints)
6. [Universal Endpoint](#universal-endpoint)

---

## Processing Endpoints

### 1. Fast Mode Processing

Process data with automatic settings for quick results.

**Endpoint**: `POST /run_fast`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes* | - | CSV file to process |
| `db_type` | String | No | "sqlite" | Database type (mysql/postgresql) |
| `host` | String | No | - | Database host |
| `port` | Integer | No | - | Database port |
| `username` | String | No | - | Database username |
| `password` | String | No | - | Database password |
| `database` | String | No | - | Database name |
| `table_name` | String | No | - | Table to import |
| `process_large_files` | Boolean | No | true | Enable large file optimization |
| `use_turbo` | Boolean | No | false | Enable turbo mode (faster) |
| `batch_size` | Integer | No | 256 | Embedding batch size |

*Either `file` or database parameters required

**Request Example (cURL)**:

```bash
curl -X POST http://127.0.0.1:8001/run_fast \
  -F "file=@data.csv" \
  -F "use_turbo=true" \
  -F "batch_size=256"
```

**Request Example (Python)**:

```python
import requests

url = "http://127.0.0.1:8001/run_fast"
files = {"file": open("data.csv", "rb")}
data = {
    "use_turbo": "true",
    "batch_size": 256
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Response (200 OK)**:

```json
{
  "mode": "fast",
  "summary": {
    "rows": 10000,
    "chunks": 150,
    "stored": "faiss",
    "embedding_model": "paraphrase-MiniLM-L6-v2",
    "retrieval_ready": true,
    "turbo_mode": true,
    "file_info": {
      "filename": "data.csv",
      "file_size": 2048576,
      "upload_time": "2024-01-15T10:30:00.123456",
      "large_file_processed": false
    }
  }
}
```

**Error Responses**:

```json
// 400 Bad Request
{
  "error": "Either file upload or database parameters required"
}

// 500 Internal Server Error
{
  "error": "Processing failed: [detailed error message]"
}
```

---

### 2. Config-1 Mode Processing

Process data with custom chunking and embedding configurations.

**Endpoint**: `POST /run_config1`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes* | - | CSV file to process |
| `chunk_method` | String | No | "recursive" | Chunking method: fixed/recursive/semantic/document |
| `chunk_size` | Integer | No | 400 | Chunk size in characters |
| `overlap` | Integer | No | 50 | Overlap between chunks |
| `n_clusters` | Integer | No | 10 | Number of clusters (semantic mode) |
| `document_key_column` | String | No | - | Key column for document chunking |
| `token_limit` | Integer | No | 2000 | Token limit for document chunking |
| `retrieval_metric` | String | No | "cosine" | Distance metric: cosine/euclidean/dot |
| `model_choice` | String | No | "paraphrase-MiniLM-L6-v2" | Embedding model |
| `storage_choice` | String | No | "faiss" | Vector store: faiss/chroma |
| `apply_default_preprocessing` | Boolean | No | true | Apply default text preprocessing |
| `use_turbo` | Boolean | No | false | Enable turbo mode |
| `batch_size` | Integer | No | 256 | Embedding batch size |

*Plus database parameters same as Fast Mode

**Request Example (cURL)**:

```bash
curl -X POST http://127.0.0.1:8001/run_config1 \
  -F "file=@data.csv" \
  -F "chunk_method=document" \
  -F "document_key_column=category" \
  -F "token_limit=1500" \
  -F "model_choice=all-MiniLM-L6-v2" \
  -F "storage_choice=chroma" \
  -F "retrieval_metric=cosine" \
  -F "use_turbo=true"
```

**Request Example (Python)**:

```python
import requests

url = "http://127.0.0.1:8001/run_config1"
files = {"file": open("data.csv", "rb")}
data = {
    "chunk_method": "document",
    "document_key_column": "category",
    "token_limit": 1500,
    "model_choice": "all-MiniLM-L6-v2",
    "storage_choice": "chroma",
    "retrieval_metric": "cosine",
    "use_turbo": "true"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Response (200 OK)**:

```json
{
  "mode": "config1",
  "summary": {
    "rows": 10000,
    "chunks": 245,
    "stored": "chroma",
    "embedding_model": "all-MiniLM-L6-v2",
    "retrieval_ready": true,
    "turbo_mode": true
  }
}
```

---

### 3. Deep Config Mode - Step 1: Preprocess

Load and preprocess data with automatic header normalization and text cleaning.

**Endpoint**: `POST /deep_config/preprocess`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes* | - | CSV file to process |
| `db_type` | String | No | "sqlite" | Database type |
| `host` | String | No | - | Database host |
| `port` | Integer | No | - | Database port |
| `username` | String | No | - | Database username |
| `password` | String | No | - | Database password |
| `database` | String | No | - | Database name |
| `table_name` | String | No | - | Table to import |

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/deep_config/preprocess \
  -F "file=@data.csv"
```

**Response (200 OK)**:

```json
{
  "status": "success",
  "rows": 10000,
  "columns": 15,
  "column_names": ["id", "name", "description", "category", "price"],
  "data_types": {
    "id": "int64",
    "name": "object",
    "description": "object",
    "category": "object",
    "price": "float64"
  },
  "file_info": {
    "source": "csv",
    "filename": "data.csv"
  }
}
```

---

### 4. Deep Config Mode - Step 2: Type Conversion

Convert column data types.

**Endpoint**: `POST /deep_config/type_convert`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `type_conversions` | JSON String | No | "{}" | Column type conversions |

**Type Conversion Options**:
- `"integer"` / `"int"` - Convert to integer
- `"float"` - Convert to float
- `"string"` / `"text"` - Convert to string
- `"boolean"` / `"bool"` - Convert to boolean
- `"datetime"` - Convert to datetime
- `"category"` - Convert to categorical

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/deep_config/type_convert \
  -F 'type_conversions={"id":"integer","price":"float","is_active":"boolean","date_added":"datetime","category":"category"}'
```

**Response (200 OK)**:

```json
{
  "status": "success",
  "rows": 10000,
  "columns": 15,
  "data_types": {
    "id": "Int64",
    "price": "float64",
    "is_active": "boolean",
    "date_added": "datetime64[ns]",
    "category": "category"
  },
  "conversions_applied": {
    "id": "integer",
    "price": "float",
    "is_active": "boolean",
    "date_added": "datetime",
    "category": "category"
  }
}
```

---

### 5. Deep Config Mode - Step 3: Null Handling

Handle missing values with various strategies.

**Endpoint**: `POST /deep_config/null_handle`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `null_strategies` | JSON String | No | "{}" | Null handling strategies per column |

**Null Strategy Options**:
- `"drop"` - Remove rows with nulls
- `"mean"` - Fill with column mean (numeric only)
- `"median"` - Fill with column median (numeric only)
- `"mode"` - Fill with most frequent value
- `"zero"` - Fill with zero (numeric) or "Unknown" (text)
- `"unknown"` - Fill with "Unknown" string
- `"ffill"` - Forward fill from previous row
- `"bfill"` - Backward fill from next row
- `"custom_value:X"` - Fill with custom value X

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/deep_config/null_handle \
  -F 'null_strategies={"price":"mean","description":"unknown","date_added":"bfill","rating":"median","custom_field":"custom_value:N/A"}'
```

**Response (200 OK)**:

```json
{
  "status": "success",
  "rows": 10000,
  "columns": 15,
  "null_count": 0,
  "strategies_applied": {
    "price": "mean",
    "description": "unknown",
    "date_added": "bfill",
    "rating": "median",
    "custom_field": "custom_value:N/A"
  }
}
```

---

### 6. Deep Config Mode - Step 4: Duplicate Handling

Analyze and remove duplicate rows.

**Endpoint**: `POST /deep_config/duplicates`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `strategy` | String | No | "keep_first" | Duplicate handling strategy |

**Strategy Options**:
- `"keep_first"` - Keep first occurrence, remove rest
- `"keep_last"` - Keep last occurrence, remove rest
- `"remove_all"` - Remove all duplicate rows
- `"keep_all"` - Don't remove any duplicates

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/deep_config/duplicates \
  -F "strategy=keep_first"
```

**Response (200 OK)**:

```json
{
  "status": "success",
  "rows_before": 10000,
  "rows_after": 9850,
  "duplicates_removed": 150,
  "strategy_applied": "keep_first",
  "has_duplicates": true,
  "duplicate_groups_count": 75
}
```

---

### 7. Deep Config Mode - Step 5: Stopword Removal

Remove common stopwords from text columns.

**Endpoint**: `POST /deep_config/stopwords`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `remove_stopwords` | Boolean | No | false | Enable stopword removal |

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/deep_config/stopwords \
  -F "remove_stopwords=true"
```

**Response (200 OK)**:

```json
{
  "status": "success",
  "rows": 9850,
  "columns": 15,
  "stopwords_removed": true
}
```

---

### 8. Deep Config Mode - Step 6: Text Normalization

Apply advanced text processing (lemmatization or stemming).

**Endpoint**: `POST /deep_config/normalize`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text_processing` | String | No | "none" | Text processing method |

**Processing Options**:
- `"none"` - No additional processing
- `"lemmatize"` - Apply lemmatization (spaCy)
- `"stem"` - Apply stemming (NLTK Porter Stemmer)

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/deep_config/normalize \
  -F "text_processing=lemmatize"
```

**Response (200 OK)**:

```json
{
  "status": "success",
  "rows": 9850,
  "columns": 15,
  "text_processing": "lemmatize"
}
```

---

### 9. Deep Config Mode - Step 7: Chunking

Chunk data with optional metadata extraction.

**Endpoint**: `POST /deep_config/chunk`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `chunk_method` | String | No | "fixed" | Chunking method |
| `chunk_size` | Integer | No | 400 | Chunk size (fixed/recursive) |
| `overlap` | Integer | No | 50 | Overlap size (fixed/recursive) |
| `key_column` | String | No | - | Key column (document method) |
| `token_limit` | Integer | No | 2000 | Token limit (document method) |
| `n_clusters` | Integer | No | 10 | Number of clusters (semantic method) |
| `store_metadata` | Boolean | No | false | Extract and store metadata |
| `selected_numeric_columns` | JSON Array | No | "[]" | Numeric columns for metadata |
| `selected_categorical_columns` | JSON Array | No | "[]" | Categorical columns for metadata |

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/deep_config/chunk \
  -F "chunk_method=document" \
  -F "key_column=category" \
  -F "token_limit=1500" \
  -F "store_metadata=true" \
  -F 'selected_numeric_columns=["price","rating"]' \
  -F 'selected_categorical_columns=["category","brand"]'
```

**Response (200 OK)**:

```json
{
  "status": "success",
  "total_chunks": 245,
  "chunks": ["chunk text 1...", "chunk text 2...", "..."],
  "chunk_method": "document",
  "chunk_size": null,
  "overlap": null,
  "metadata": [
    {
      "chunk_id": "0",
      "key_column": "category",
      "key_value": "Electronics",
      "chunk_index": 0,
      "price_min": 99.99,
      "price_mean": 549.50,
      "price_max": 1299.99,
      "rating_min": 3.5,
      "rating_mean": 4.2,
      "rating_max": 4.9,
      "category_mode": "Electronics",
      "brand_mode": "Samsung"
    }
  ],
  "metadata_enabled": true
}
```

---

### 10. Deep Config Mode - Step 8: Embedding Generation

Generate vector embeddings for chunks.

**Endpoint**: `POST /deep_config/embed`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_name` | String | No | "paraphrase-MiniLM-L6-v2" | Embedding model |
| `batch_size` | Integer | No | 64 | Batch size for embedding |
| `use_parallel` | Boolean | No | true | Use parallel processing |

**Model Options**:
- `"paraphrase-MiniLM-L6-v2"` - Fast, 384 dimensions
- `"all-MiniLM-L6-v2"` - Balanced, 384 dimensions
- `"text-embedding-ada-002"` - OpenAI (requires API key), 1536 dimensions

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/deep_config/embed \
  -F "model_name=all-MiniLM-L6-v2" \
  -F "batch_size=128" \
  -F "use_parallel=true"
```

**Response (200 OK)**:

```json
{
  "status": "success",
  "total_chunks": 245,
  "vector_dimension": 384,
  "embeddings": [[0.123, -0.456, ...], ...],
  "chunk_texts": ["chunk 1", "chunk 2", ...],
  "model_name": "all-MiniLM-L6-v2"
}
```

---

### 11. Deep Config Mode - Step 9: Vector Storage

Store embeddings in vector database.

**Endpoint**: `POST /deep_config/store`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `storage_type` | String | No | "chroma" | Storage type: chroma/faiss |
| `collection_name` | String | No | "deep_config_collection" | Collection/index name |

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/deep_config/store \
  -F "storage_type=faiss" \
  -F "collection_name=my_vectors"
```

**Response (200 OK)**:

```json
{
  "status": "success",
  "storage_type": "faiss",
  "collection_name": "faiss_index",
  "total_vectors": 245
}
```

---

## Retrieval Endpoints

### 12. Basic Retrieval

Semantic search for similar chunks.

**Endpoint**: `POST /retrieve`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | String | Yes | - | Search query |
| `k` | Integer | No | 5 | Number of results to return |

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/retrieve \
  -F "query=ergonomic office furniture" \
  -F "k=5"
```

**Response (200 OK)**:

```json
{
  "query": "ergonomic office furniture",
  "k": 5,
  "results": [
    {
      "rank": 1,
      "content": "id: 42 | name: ErgoChair Pro | description: Ergonomic office chair...",
      "similarity": 0.89,
      "distance": 0.11
    },
    {
      "rank": 2,
      "content": "id: 87 | name: Standing Desk | description: Adjustable height desk...",
      "similarity": 0.85,
      "distance": 0.15
    }
  ]
}
```

---

### 13. Metadata-Filtered Retrieval

Semantic search with metadata filtering.

**Endpoint**: `POST /retrieve_with_metadata`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | String | Yes | - | Search query |
| `k` | Integer | No | 5 | Number of results to return |
| `metadata_filter` | JSON String | No | "{}" | Metadata filter conditions |

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/retrieve_with_metadata \
  -F "query=premium laptops" \
  -F "k=3" \
  -F 'metadata_filter={"category_mode":"Electronics","price_min":1000}'
```

**Response (200 OK)**:

```json
{
  "status": "success",
  "query": "premium laptops",
  "results": [
    {
      "rank": 1,
      "content": "id: 15 | name: MacBook Pro | description: Professional laptop...",
      "similarity": 0.92,
      "distance": 0.08,
      "metadata": {
        "chunk_index": 5,
        "category_mode": "Electronics",
        "price_min": 1299.99,
        "price_mean": 1450.00,
        "price_max": 1699.99
      }
    }
  ],
  "total_results": 3,
  "metadata_filter_applied": true,
  "store_type": "faiss"
}
```

---

## Export Endpoints

### 14. Export Chunks

Download processed chunks as CSV.

**Endpoint**: `GET /export/chunks`

**Response**: CSV file download

**Request Example**:

```bash
curl http://127.0.0.1:8001/export/chunks --output chunks.csv
```

---

### 15. Export Embeddings (Binary)

Download embeddings as NumPy array.

**Endpoint**: `GET /export/embeddings`

**Response**: .npy file download

**Request Example**:

```bash
curl http://127.0.0.1:8001/export/embeddings --output embeddings.npy

# Load in Python:
# import numpy as np
# embeddings = np.load('embeddings.npy')
```

---

### 16. Export Embeddings (JSON)

Download embeddings as JSON.

**Endpoint**: `GET /export/embeddings_text`

**Response**: JSON file download

**Request Example**:

```bash
curl http://127.0.0.1:8001/export/embeddings_text --output embeddings.json
```

---

### 17. Export Preprocessed Data

Download preprocessed data as CSV.

**Endpoint**: `GET /export/preprocessed`

**Response**: CSV file download

**Request Example**:

```bash
curl http://127.0.0.1:8001/export/preprocessed --output preprocessed_data.csv
```

---

## System Endpoints

### 18. Health Check

Check API health status.

**Endpoint**: `GET /health`

**Response (200 OK)**:

```json
{
  "status": "healthy",
  "large_file_support": true,
  "performance_optimized": true
}
```

---

### 19. System Information

Get system resource information.

**Endpoint**: `GET /system_info`

**Response (200 OK)**:

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

---

### 20. API Capabilities

Get list of API features and supported models.

**Endpoint**: `GET /capabilities`

**Response (200 OK)**:

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
  "memory_optimized": true,
  "database_large_table_support": true,
  "performance_features": {
    "turbo_mode": true,
    "parallel_processing": true,
    "optimized_batch_size": 256,
    "caching_system": true
  }
}
```

---

## Database Endpoints

### 21. Test Database Connection

Test connection to MySQL or PostgreSQL database.

**Endpoint**: `POST /db/test_connection`

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `db_type` | String | Yes | Database type: mysql/postgresql |
| `host` | String | Yes | Database host |
| `port` | Integer | Yes | Database port |
| `username` | String | Yes | Database username |
| `password` | String | Yes | Database password |
| `database` | String | Yes | Database name |

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/db/test_connection \
  -F "db_type=mysql" \
  -F "host=localhost" \
  -F "port=3306" \
  -F "username=root" \
  -F "password=password" \
  -F "database=mydb"
```

**Response (200 OK)**:

```json
{
  "status": "success"
}
```

**Error Response**:

```json
{
  "status": "error",
  "message": "Access denied for user 'root'@'localhost'"
}
```

---

### 22. List Database Tables

List all tables in database.

**Endpoint**: `POST /db/list_tables`

**Parameters**: Same as `/db/test_connection`

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/db/list_tables \
  -F "db_type=mysql" \
  -F "host=localhost" \
  -F "port=3306" \
  -F "username=root" \
  -F "password=password" \
  -F "database=mydb"
```

**Response (200 OK)**:

```json
{
  "tables": [
    "products",
    "customers",
    "orders",
    "categories"
  ]
}
```

---

### 23. Import and Process Database Table

Import table from database and process it.

**Endpoint**: `POST /db/import_one`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `db_type` | String | Yes | - | Database type |
| `host` | String | Yes | - | Database host |
| `port` | Integer | Yes | - | Database port |
| `username` | String | Yes | - | Database username |
| `password` | String | Yes | - | Database password |
| `database` | String | Yes | - | Database name |
| `table_name` | String | Yes | - | Table to import |
| `processing_mode` | String | No | "fast" | Processing mode: fast/config1/deep |
| `use_turbo` | Boolean | No | false | Enable turbo mode |
| `batch_size` | Integer | No | 256 | Batch size |

**Request Example**:

```bash
curl -X POST http://127.0.0.1:8001/db/import_one \
  -F "db_type=mysql" \
  -F "host=localhost" \
  -F "port=3306" \
  -F "username=root" \
  -F "password=password" \
  -F "database=mydb" \
  -F "table_name=products" \
  -F "processing_mode=fast" \
  -F "use_turbo=true"
```

**Response (200 OK)**:

```json
{
  "mode": "fast",
  "summary": {
    "rows": 50000,
    "chunks": 750,
    "stored": "faiss",
    "embedding_model": "paraphrase-MiniLM-L6-v2",
    "retrieval_ready": true
  }
}
```

---

## Universal Endpoint

### 24. Universal Process Endpoint

Single endpoint that handles all operations.

**Endpoint**: `POST /api/v1/process`

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `operation` | String | Yes | Operation type (see below) |
| ... | ... | Varies | Additional parameters based on operation |

**Operation Types**:
- `"fast"` - Fast mode processing
- `"config1"` - Config-1 mode processing
- `"deep_config"` - Deep config mode processing
- `"deep_config_step"` - Deep config step-by-step (specify `step`)
- `"retrieve"` - Semantic search
- `"export"` - Export data (specify `export_type`)
- `"system"` - System info (specify `system_action`)
- `"db_test"` - Test database connection
- `"db_list"` - List database tables
- `"db_import"` - Import database table

**Example Requests**:

```bash
# Fast Mode
curl -X POST http://127.0.0.1:8001/api/v1/process \
  -F "operation=fast" \
  -F "file=@data.csv" \
  -F "use_turbo=true"

# Retrieval
curl -X POST http://127.0.0.1:8001/api/v1/process \
  -F "operation=retrieve" \
  -F "query=search term" \
  -F "k=5"

# Export
curl -X POST http://127.0.0.1:8001/api/v1/process \
  -F "operation=export" \
  -F "export_type=chunks"

# System Info
curl -X POST http://127.0.0.1:8001/api/v1/process \
  -F "operation=system" \
  -F "system_action=info"
```

---

## Error Handling

All endpoints return standard HTTP status codes:

| Status Code | Meaning | Example |
|-------------|---------|---------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid parameters or missing required fields |
| 404 | Not Found | Endpoint doesn't exist |
| 500 | Internal Server Error | Processing error or system failure |

**Standard Error Response Format**:

```json
{
  "error": "Detailed error message describing what went wrong"
}
```

---

## Rate Limits

Currently, there are no rate limits enforced. For production deployment, consider implementing:

- Request rate limiting
- File size limits (currently 3GB recommended)
- Concurrent request limits
- API key authentication

---

**Next**: See [Python Examples](../../05-EXAMPLES/python-examples.md) for code samples.

