# Quick Start Guide

##  Getting Started in 5 Minutes

This guide will walk you through your first data processing workflow using the Chunking Optimizer API.

## Prerequisites

-  API server running on `http://127.0.0.1:8001`
-  Sample CSV file ready (or use test datasets)
-  cURL or Postman installed

##  Workflow Overview

```
                  
   Upload        Process       Retrieve      Export     
   CSV File            & Chunk             Similar             Results    
                  
```

##  Example 1: Fast Mode (Simplest)

### Step 1: Prepare Your Data

Create a sample CSV file `sample.csv`:

```csv
id,name,description,category
1,Product A,High quality laptop with 16GB RAM,Electronics
2,Product B,Comfortable office chair with lumbar support,Furniture
3,Product C,Wireless mouse with ergonomic design,Electronics
4,Product D,Standing desk adjustable height,Furniture
5,Product E,Noise cancelling headphones,Electronics
```

### Step 2: Process the File

```bash
curl -X POST http://127.0.0.1:8001/run_fast \
  -F "file=@sample.csv" \
  -F "use_turbo=true" \
  -F "batch_size=256"
```

**Response:**
```json
{
  "mode": "fast",
  "summary": {
    "rows": 5,
    "chunks": 3,
    "stored": "faiss",
    "embedding_model": "paraphrase-MiniLM-L6-v2",
    "retrieval_ready": true,
    "turbo_mode": true,
    "file_info": {
      "filename": "sample.csv",
      "file_size": 285,
      "upload_time": "2024-01-15T10:30:00"
    }
  }
}
```

### Step 3: Search for Similar Items

```bash
curl -X POST http://127.0.0.1:8001/retrieve \
  -F "query=ergonomic computer accessories" \
  -F "k=3"
```

**Response:**
```json
{
  "query": "ergonomic computer accessories",
  "k": 3,
  "results": [
    {
      "rank": 1,
      "content": "id: 3 | name: Product C | description: Wireless mouse with ergonomic design | category: Electronics",
      "similarity": 0.87,
      "distance": 0.13
    },
    {
      "rank": 2,
      "content": "id: 5 | name: Product E | description: Noise cancelling headphones | category: Electronics",
      "similarity": 0.76,
      "distance": 0.24
    },
    {
      "rank": 3,
      "content": "id: 1 | name: Product A | description: High quality laptop with 16GB RAM | category: Electronics",
      "similarity": 0.68,
      "distance": 0.32
    }
  ]
}
```

### Step 4: Export Results

```bash
# Export chunks
curl http://127.0.0.1:8001/export/chunks --output chunks.csv

# Export embeddings
curl http://127.0.0.1:8001/export/embeddings --output embeddings.npy

# Export preprocessed data
curl http://127.0.0.1:8001/export/preprocessed --output preprocessed.csv
```

##  Example 2: Config-1 Mode (Customizable)

### Step 1: Process with Custom Settings

```bash
curl -X POST http://127.0.0.1:8001/run_config1 \
  -F "file=@sample.csv" \
  -F "chunk_method=document" \
  -F "document_key_column=category" \
  -F "token_limit=1000" \
  -F "model_choice=all-MiniLM-L6-v2" \
  -F "storage_choice=chroma" \
  -F "retrieval_metric=cosine" \
  -F "use_turbo=true"
```

**Response:**
```json
{
  "mode": "config1",
  "summary": {
    "rows": 5,
    "chunks": 2,
    "stored": "chroma",
    "embedding_model": "all-MiniLM-L6-v2",
    "retrieval_ready": true,
    "turbo_mode": true
  }
}
```

### Step 2: Retrieve with Metadata Filtering

```bash
curl -X POST http://127.0.0.1:8001/retrieve_with_metadata \
  -F "query=comfortable workspace" \
  -F "k=5" \
  -F 'metadata_filter={"key_value":"Furniture"}'
```

**Response:**
```json
{
  "status": "success",
  "query": "comfortable workspace",
  "results": [
    {
      "rank": 1,
      "content": "HEADERS: id | name | description | category\nid:2 | name:Product B | description:Comfortable office chair...",
      "similarity": 0.92,
      "distance": 0.08,
      "metadata": {
        "key_column": "category",
        "key_value": "Furniture",
        "chunk_index": 0
      }
    },
    {
      "rank": 2,
      "content": "HEADERS: id | name | description | category\nid:4 | name:Product D | description:Standing desk...",
      "similarity": 0.85,
      "distance": 0.15,
      "metadata": {
        "key_column": "category",
        "key_value": "Furniture",
        "chunk_index": 1
      }
    }
  ],
  "total_results": 2,
  "metadata_filter_applied": true
}
```

##  Example 3: Deep Config Mode (Step-by-Step)

### Step 1: Preprocess Data

```bash
curl -X POST http://127.0.0.1:8001/deep_config/preprocess \
  -F "file=@sample.csv"
```

**Response:**
```json
{
  "status": "success",
  "rows": 5,
  "columns": 4,
  "column_names": ["id", "name", "description", "category"],
  "data_types": {
    "id": "int64",
    "name": "object",
    "description": "object",
    "category": "object"
  }
}
```

### Step 2: Handle Data Types

```bash
curl -X POST http://127.0.0.1:8001/deep_config/type_convert \
  -F 'type_conversions={"id":"integer","category":"category"}'
```

### Step 3: Handle Nulls

```bash
curl -X POST http://127.0.0.1:8001/deep_config/null_handle \
  -F 'null_strategies={"description":"unknown","name":"mode"}'
```

### Step 4: Remove Duplicates

```bash
curl -X POST http://127.0.0.1:8001/deep_config/duplicates \
  -F "strategy=keep_first"
```

### Step 5: Chunk Data

```bash
curl -X POST http://127.0.0.1:8001/deep_config/chunk \
  -F "chunk_method=semantic" \
  -F "n_clusters=3" \
  -F "store_metadata=true" \
  -F 'selected_numeric_columns=["id"]' \
  -F 'selected_categorical_columns=["category"]'
```

**Response:**
```json
{
  "status": "success",
  "total_chunks": 3,
  "chunks": [
    "id: 1 | name: product a | description: high quality laptop...",
    "id: 3 | name: product c | description: wireless mouse...",
    "id: 2 | name: product b | description: comfortable office chair..."
  ],
  "metadata": [
    {
      "chunk_id": "sem_cluster_0000",
      "method": "semantic_cluster",
      "id_min": 1,
      "id_mean": 1.0,
      "id_max": 1,
      "category_mode": "Electronics"
    }
  ],
  "metadata_enabled": true
}
```

### Step 6: Generate Embeddings

```bash
curl -X POST http://127.0.0.1:8001/deep_config/embed \
  -F "model_name=paraphrase-MiniLM-L6-v2" \
  -F "batch_size=64" \
  -F "use_parallel=true"
```

**Response:**
```json
{
  "status": "success",
  "total_chunks": 3,
  "vector_dimension": 384,
  "model_name": "paraphrase-MiniLM-L6-v2"
}
```

### Step 7: Store Vectors

```bash
curl -X POST http://127.0.0.1:8001/deep_config/store \
  -F "storage_type=faiss" \
  -F "collection_name=my_collection"
```

**Response:**
```json
{
  "status": "success",
  "storage_type": "faiss",
  "collection_name": "faiss_index",
  "total_vectors": 3
}
```

### Step 8: Retrieve with Metadata

```bash
curl -X POST http://127.0.0.1:8001/retrieve_with_metadata \
  -F "query=electronic devices" \
  -F "k=3" \
  -F 'metadata_filter={"category_mode":"Electronics"}'
```

##  Example 4: Database Import

### Step 1: Test Connection

```bash
curl -X POST http://127.0.0.1:8001/db/test_connection \
  -F "db_type=mysql" \
  -F "host=localhost" \
  -F "port=3306" \
  -F "username=root" \
  -F "password=password" \
  -F "database=mydb"
```

**Response:**
```json
{
  "status": "success"
}
```

### Step 2: List Tables

```bash
curl -X POST http://127.0.0.1:8001/db/list_tables \
  -F "db_type=mysql" \
  -F "host=localhost" \
  -F "port=3306" \
  -F "username=root" \
  -F "password=password" \
  -F "database=mydb"
```

**Response:**
```json
{
  "tables": ["products", "customers", "orders"]
}
```

### Step 3: Import and Process

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

##  Example 5: Universal Endpoint

The API provides a single universal endpoint that can handle all operations:

```bash
# Fast Mode
curl -X POST http://127.0.0.1:8001/api/v1/process \
  -F "operation=fast" \
  -F "file=@sample.csv" \
  -F "use_turbo=true"

# Retrieval
curl -X POST http://127.0.0.1:8001/api/v1/process \
  -F "operation=retrieve" \
  -F "query=ergonomic accessories" \
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

##  Performance Tips

### 1. Enable Turbo Mode
```bash
-F "use_turbo=true"  # 30-50% faster processing
```

### 2. Adjust Batch Size
```bash
-F "batch_size=512"  # Larger = faster (more memory)
-F "batch_size=128"  # Smaller = slower (less memory)
```

### 3. Choose Right Chunking Method
- **Semantic**: Best for similar content grouping (slowest)
- **Document**: Best for key-based grouping (medium)
- **Recursive**: Best for structured data (fast)
- **Fixed**: Best for uniform chunks (fastest)

### 4. Select Appropriate Model
- **paraphrase-MiniLM-L6-v2**: Fastest, good quality
- **all-MiniLM-L6-v2**: Fast, balanced
- **text-embedding-ada-002**: Best quality (requires API key)

##  Common Patterns

### Pattern 1: CSV → Chunks → Search

```bash
# 1. Process
curl -X POST http://127.0.0.1:8001/run_fast -F "file=@data.csv" -F "use_turbo=true"

# 2. Search
curl -X POST http://127.0.0.1:8001/retrieve -F "query=your query" -F "k=5"

# 3. Export
curl http://127.0.0.1:8001/export/chunks --output results.csv
```

### Pattern 2: Database → Custom Chunks → Filtered Search

```bash
# 1. Import from DB
curl -X POST http://127.0.0.1:8001/db/import_one \
  -F "db_type=mysql" -F "host=localhost" -F "table_name=products" \
  -F "processing_mode=config1"

# 2. Search with metadata
curl -X POST http://127.0.0.1:8001/retrieve_with_metadata \
  -F "query=premium products" -F 'metadata_filter={"category":"Electronics"}'
```

### Pattern 3: Step-by-Step Processing

```bash
# 1. Preprocess
curl -X POST http://127.0.0.1:8001/deep_config/preprocess -F "file=@data.csv"

# 2. Clean nulls
curl -X POST http://127.0.0.1:8001/deep_config/null_handle -F 'null_strategies={"col":"mean"}'

# 3. Chunk
curl -X POST http://127.0.0.1:8001/deep_config/chunk -F "chunk_method=semantic"

# 4. Embed
curl -X POST http://127.0.0.1:8001/deep_config/embed -F "model_name=paraphrase-MiniLM-L6-v2"

# 5. Store
curl -X POST http://127.0.0.1:8001/deep_config/store -F "storage_type=faiss"

# 6. Retrieve
curl -X POST http://127.0.0.1:8001/retrieve -F "query=search term" -F "k=5"
```

##  Verification Checklist

After completing the quick start:

- [ ] API server is running
- [ ] Health check returns `{"status": "healthy"}`
- [ ] Successfully processed a CSV file
- [ ] Retrieved similar chunks
- [ ] Exported results to files
- [ ] Tested metadata filtering (optional)
- [ ] Tested database import (optional)

##  Next Steps

Now that you've completed the quick start:

1.  Explore [Detailed API Reference](../03-API-REFERENCE/core-endpoints.md)
2.  Learn about [Metadata Filtering](../06-ADVANCED/metadata-filtering.md)
3.  Optimize for [Large Files](../06-ADVANCED/large-files.md)
4.  Check [More Examples](../05-EXAMPLES/python-examples.md)

##  Troubleshooting

**Issue**: File upload fails
- Check file size < 3GB
- Verify CSV format is correct
- Try with smaller test file first

**Issue**: No results returned
- Verify processing completed successfully
- Check if chunks were created
- Try different query terms

**Issue**: Slow performance
- Enable turbo mode
- Increase batch size
- Use faster embedding model
- Check system resources

---

**Congratulations! ** You've successfully completed the quick start guide.

