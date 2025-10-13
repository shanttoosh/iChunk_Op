# Python Client Examples

##  Installation

```python
pip install requests pandas numpy
```

##  Basic Usage

### Example 1: Simple Fast Mode Processing

```python
import requests
import pandas as pd

# API endpoint
BASE_URL = "http://127.0.0.1:8001"

# Load your data
df = pd.read_csv("data.csv")

# Process with Fast Mode
url = f"{BASE_URL}/run_fast"
files = {"file": open("data.csv", "rb")}
data = {
    "use_turbo": "true",
    "batch_size": 256
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Processed {result['summary']['rows']} rows")
print(f"Created {result['summary']['chunks']} chunks")
print(f"Ready for retrieval: {result['summary']['retrieval_ready']}")
```

**Output:**
```
Processed 10000 rows
Created 150 chunks
Ready for retrieval: True
```

---

### Example 2: Search and Retrieve

```python
import requests

BASE_URL = "http://127.0.0.1:8001"

# Search for similar content
query = "ergonomic office furniture"
url = f"{BASE_URL}/retrieve"
data = {
    "query": query,
    "k": 5
}

response = requests.post(url, data=data)
results = response.json()

# Print results
print(f"Query: {results['query']}")
print(f"Found {len(results['results'])} results\n")

for result in results['results']:
    print(f"Rank {result['rank']}: {result['similarity']:.3f}")
    print(f"Content: {result['content'][:100]}...")
    print()
```

**Output:**
```
Query: ergonomic office furniture
Found 5 results

Rank 1: 0.892
Content: id: 42 | name: ErgoChair Pro | description: Ergonomic office chair with lumbar support...

Rank 2: 0.857
Content: id: 87 | name: Standing Desk | description: Adjustable height standing desk for comfort...
```

---

### Example 3: Complete Workflow with Error Handling

```python
import requests
import time

class ChunkingAPI:
    """Python client for Chunking Optimizer API"""
    
    def __init__(self, base_url="http://127.0.0.1:8001"):
        self.base_url = base_url
        
    def health_check(self):
        """Check if API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Health check failed: {e}")
            return None
    
    def process_fast(self, file_path, use_turbo=True, batch_size=256):
        """Process file in Fast Mode"""
        url = f"{self.base_url}/run_fast"
        
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {
                "use_turbo": str(use_turbo).lower(),
                "batch_size": batch_size
            }
            
            try:
                response = requests.post(url, files=files, data=data)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"Processing failed: {e}")
                return None
    
    def retrieve(self, query, k=5, metadata_filter=None):
        """Retrieve similar chunks"""
        if metadata_filter:
            url = f"{self.base_url}/retrieve_with_metadata"
            data = {
                "query": query,
                "k": k,
                "metadata_filter": str(metadata_filter)
            }
        else:
            url = f"{self.base_url}/retrieve"
            data = {
                "query": query,
                "k": k
            }
        
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Retrieval failed: {e}")
            return None
    
    def export_chunks(self, output_path="chunks.csv"):
        """Export chunks to CSV"""
        url = f"{self.base_url}/export/chunks"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            print(f"Chunks exported to {output_path}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Export failed: {e}")
            return False

# Usage
api = ChunkingAPI()

# 1. Check health
health = api.health_check()
if health:
    print(f"API Status: {health['status']}")

# 2. Process file
print("\nProcessing file...")
start_time = time.time()
result = api.process_fast("data.csv", use_turbo=True)
if result:
    elapsed = time.time() - start_time
    print(f"Processing completed in {elapsed:.2f}s")
    print(f"Rows: {result['summary']['rows']}")
    print(f"Chunks: {result['summary']['chunks']}")

# 3. Search
print("\nSearching...")
results = api.retrieve("modern office equipment", k=3)
if results:
    for r in results['results']:
        print(f"  [{r['rank']}] Similarity: {r['similarity']:.3f}")

# 4. Export
print("\nExporting...")
api.export_chunks("output_chunks.csv")
```

---

### Example 4: Config-1 Mode with Custom Settings

```python
import requests

BASE_URL = "http://127.0.0.1:8001"

# Configure custom chunking
url = f"{BASE_URL}/run_config1"

with open("data.csv", "rb") as f:
    files = {"file": f}
    data = {
        "chunk_method": "document",
        "document_key_column": "category",
        "token_limit": 1500,
        "model_choice": "all-MiniLM-L6-v2",
        "storage_choice": "chroma",
        "retrieval_metric": "cosine",
        "apply_default_preprocessing": "true",
        "use_turbo": "true",
        "batch_size": 256
    }
    
    response = requests.post(url, files=files, data=data)
    result = response.json()
    
    print(f"Mode: {result['mode']}")
    print(f"Chunks: {result['summary']['chunks']}")
    print(f"Storage: {result['summary']['stored']}")
    print(f"Model: {result['summary']['embedding_model']}")
```

---

### Example 5: Deep Config Mode (Step-by-Step)

```python
import requests
import json

BASE_URL = "http://127.0.0.1:8001"

class DeepConfigProcessor:
    """Step-by-step processing with Deep Config Mode"""
    
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
    
    def step1_preprocess(self, file_path):
        """Step 1: Load and preprocess"""
        url = f"{self.base_url}/deep_config/preprocess"
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
            return response.json()
    
    def step2_type_convert(self, conversions):
        """Step 2: Convert data types"""
        url = f"{self.base_url}/deep_config/type_convert"
        data = {"type_conversions": json.dumps(conversions)}
        response = requests.post(url, data=data)
        return response.json()
    
    def step3_handle_nulls(self, strategies):
        """Step 3: Handle null values"""
        url = f"{self.base_url}/deep_config/null_handle"
        data = {"null_strategies": json.dumps(strategies)}
        response = requests.post(url, data=data)
        return response.json()
    
    def step4_remove_duplicates(self, strategy="keep_first"):
        """Step 4: Remove duplicates"""
        url = f"{self.base_url}/deep_config/duplicates"
        data = {"strategy": strategy}
        response = requests.post(url, data=data)
        return response.json()
    
    def step5_remove_stopwords(self, remove=True):
        """Step 5: Remove stopwords"""
        url = f"{self.base_url}/deep_config/stopwords"
        data = {"remove_stopwords": str(remove).lower()}
        response = requests.post(url, data=data)
        return response.json()
    
    def step6_chunk(self, method="semantic", n_clusters=10, 
                    store_metadata=True, numeric_cols=None, categorical_cols=None):
        """Step 6: Chunk data with metadata"""
        url = f"{self.base_url}/deep_config/chunk"
        data = {
            "chunk_method": method,
            "n_clusters": n_clusters,
            "store_metadata": str(store_metadata).lower(),
            "selected_numeric_columns": json.dumps(numeric_cols or []),
            "selected_categorical_columns": json.dumps(categorical_cols or [])
        }
        response = requests.post(url, data=data)
        return response.json()
    
    def step7_embed(self, model_name="paraphrase-MiniLM-L6-v2", 
                    batch_size=64, use_parallel=True):
        """Step 7: Generate embeddings"""
        url = f"{self.base_url}/deep_config/embed"
        data = {
            "model_name": model_name,
            "batch_size": batch_size,
            "use_parallel": str(use_parallel).lower()
        }
        response = requests.post(url, data=data)
        return response.json()
    
    def step8_store(self, storage_type="faiss", collection_name="my_collection"):
        """Step 8: Store vectors"""
        url = f"{self.base_url}/deep_config/store"
        data = {
            "storage_type": storage_type,
            "collection_name": collection_name
        }
        response = requests.post(url, data=data)
        return response.json()

# Usage
processor = DeepConfigProcessor()

# Step 1: Preprocess
print("Step 1: Preprocessing...")
result = processor.step1_preprocess("data.csv")
print(f"  Loaded {result['rows']} rows, {result['columns']} columns")

# Step 2: Convert types
print("\nStep 2: Type conversion...")
conversions = {
    "id": "integer",
    "price": "float",
    "is_active": "boolean",
    "date_added": "datetime"
}
result = processor.step2_type_convert(conversions)
print(f"  Types converted: {len(result.get('conversions_applied', {}))}")

# Step 3: Handle nulls
print("\nStep 3: Handling nulls...")
null_strategies = {
    "price": "mean",
    "description": "unknown",
    "rating": "median"
}
result = processor.step3_handle_nulls(null_strategies)
print(f"  Remaining nulls: {result['null_count']}")

# Step 4: Remove duplicates
print("\nStep 4: Removing duplicates...")
result = processor.step4_remove_duplicates("keep_first")
print(f"  Removed {result['duplicates_removed']} duplicate rows")

# Step 5: Remove stopwords
print("\nStep 5: Removing stopwords...")
result = processor.step5_remove_stopwords(True)
print(f"  Stopwords removed: {result['stopwords_removed']}")

# Step 6: Chunk with metadata
print("\nStep 6: Chunking with metadata...")
result = processor.step6_chunk(
    method="semantic",
    n_clusters=20,
    store_metadata=True,
    numeric_cols=["price", "rating"],
    categorical_cols=["category", "brand"]
)
print(f"  Created {result['total_chunks']} chunks with metadata")

# Step 7: Generate embeddings
print("\nStep 7: Generating embeddings...")
result = processor.step7_embed(
    model_name="paraphrase-MiniLM-L6-v2",
    batch_size=128,
    use_parallel=True
)
print(f"  Generated {result['total_chunks']} embeddings")
print(f"  Dimension: {result['vector_dimension']}")

# Step 8: Store vectors
print("\nStep 8: Storing vectors...")
result = processor.step8_store("faiss", "my_vectors")
print(f"  Stored {result['total_vectors']} vectors in {result['storage_type']}")

print("\n Processing complete!")
```

---

### Example 6: Database Import and Processing

```python
import requests

BASE_URL = "http://127.0.0.1:8001"

class DatabaseImporter:
    """Import and process data from databases"""
    
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
    
    def test_connection(self, db_config):
        """Test database connection"""
        url = f"{self.base_url}/db/test_connection"
        response = requests.post(url, data=db_config)
        return response.json()
    
    def list_tables(self, db_config):
        """List all tables in database"""
        url = f"{self.base_url}/db/list_tables"
        response = requests.post(url, data=db_config)
        return response.json()
    
    def import_and_process(self, db_config, table_name, mode="fast"):
        """Import table and process"""
        url = f"{self.base_url}/db/import_one"
        data = {
            **db_config,
            "table_name": table_name,
            "processing_mode": mode,
            "use_turbo": "true",
            "batch_size": 256
        }
        response = requests.post(url, data=data)
        return response.json()

# Database configuration
db_config = {
    "db_type": "mysql",
    "host": "localhost",
    "port": 3306,
    "username": "root",
    "password": "password",
    "database": "mydb"
}

importer = DatabaseImporter()

# Test connection
print("Testing connection...")
result = importer.test_connection(db_config)
if result["status"] == "success":
    print(" Connected successfully")
    
    # List tables
    print("\nListing tables...")
    tables = importer.list_tables(db_config)
    print(f"Found {len(tables['tables'])} tables:")
    for table in tables["tables"]:
        print(f"  - {table}")
    
    # Import and process a table
    print("\nImporting and processing 'products' table...")
    result = importer.import_and_process(db_config, "products", mode="fast")
    print(f"Mode: {result['mode']}")
    print(f"Rows: {result['summary']['rows']}")
    print(f"Chunks: {result['summary']['chunks']}")
else:
    print(f" Connection failed: {result['message']}")
```

---

### Example 7: Metadata Filtering

```python
import requests
import json

BASE_URL = "http://127.0.0.1:8001"

def retrieve_with_filters(query, filters=None, k=5):
    """Retrieve with metadata filtering"""
    url = f"{BASE_URL}/retrieve_with_metadata"
    
    data = {
        "query": query,
        "k": k,
        "metadata_filter": json.dumps(filters or {})
    }
    
    response = requests.post(url, data=data)
    return response.json()

# Example 1: Filter by category
results = retrieve_with_filters(
    query="premium products",
    filters={"category_mode": "Electronics"},
    k=5
)

print("Filter: Electronics only")
for r in results["results"]:
    print(f"  Rank {r['rank']}: {r['similarity']:.3f}")
    print(f"  Category: {r['metadata']['category_mode']}")

# Example 2: Filter by price range
results = retrieve_with_filters(
    query="affordable options",
    filters={"price_min": 0, "price_max": 500},
    k=5
)

print("\nFilter: Price < $500")
for r in results["results"]:
    print(f"  Rank {r['rank']}: {r['similarity']:.3f}")
    print(f"  Price: ${r['metadata']['price_mean']:.2f}")

# Example 3: Multiple filters
results = retrieve_with_filters(
    query="high quality items",
    filters={
        "category_mode": "Electronics",
        "rating_mean": 4.0  # Minimum rating
    },
    k=3
)

print("\nFilter: Electronics with rating >= 4.0")
for r in results["results"]:
    print(f"  Rank {r['rank']}: {r['similarity']:.3f}")
    print(f"  Category: {r['metadata']['category_mode']}")
    print(f"  Rating: {r['metadata']['rating_mean']:.1f}")
```

---

### Example 8: Export and Analysis

```python
import requests
import pandas as pd
import numpy as np
import json

BASE_URL = "http://127.0.0.1:8001"

class DataExporter:
    """Export processed data for analysis"""
    
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
    
    def export_chunks(self, output_path="chunks.csv"):
        """Export chunks as CSV"""
        url = f"{self.base_url}/export/chunks"
        response = requests.get(url)
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        return pd.read_csv(output_path)
    
    def export_embeddings_npy(self, output_path="embeddings.npy"):
        """Export embeddings as NumPy array"""
        url = f"{self.base_url}/export/embeddings"
        response = requests.get(url)
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        return np.load(output_path)
    
    def export_embeddings_json(self, output_path="embeddings.json"):
        """Export embeddings as JSON"""
        url = f"{self.base_url}/export/embeddings_text"
        response = requests.get(url)
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        with open(output_path, "r") as f:
            return json.load(f)
    
    def export_preprocessed(self, output_path="preprocessed.csv"):
        """Export preprocessed data"""
        url = f"{self.base_url}/export/preprocessed"
        response = requests.get(url)
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        return pd.read_csv(output_path)

# Usage
exporter = DataExporter()

# Export chunks
print("Exporting chunks...")
chunks_df = exporter.export_chunks("output_chunks.csv")
print(f"Exported {len(chunks_df)} chunks")
print(chunks_df.head())

# Export embeddings
print("\nExporting embeddings...")
embeddings = exporter.export_embeddings_npy("output_embeddings.npy")
print(f"Shape: {embeddings.shape}")
print(f"Dimension: {embeddings.shape[1]}")

# Export preprocessed data
print("\nExporting preprocessed data...")
preprocessed_df = exporter.export_preprocessed("output_preprocessed.csv")
print(f"Exported {len(preprocessed_df)} rows")
print(preprocessed_df.head())

# Analysis
print("\n=== Analysis ===")
print(f"Total chunks: {len(chunks_df)}")
print(f"Average chunk length: {chunks_df['chunk_text'].str.len().mean():.0f} chars")
print(f"Embedding dimension: {embeddings.shape[1]}")
print(f"Total vectors: {embeddings.shape[0]}")
```

---

### Example 9: Batch Processing Multiple Files

```python
import requests
import os
from pathlib import Path
import time

BASE_URL = "http://127.0.0.1:8001"

def process_directory(directory_path, output_dir="results"):
    """Process all CSV files in a directory"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = list(Path(directory_path).glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    results = []
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing {csv_file.name}...")
        
        try:
            # Process file
            start_time = time.time()
            url = f"{BASE_URL}/run_fast"
            
            with open(csv_file, "rb") as f:
                files = {"file": f}
                data = {"use_turbo": "true", "batch_size": 256}
                response = requests.post(url, files=files, data=data)
                result = response.json()
            
            elapsed = time.time() - start_time
            
            # Export chunks
            export_url = f"{BASE_URL}/export/chunks"
            export_response = requests.get(export_url)
            
            output_file = os.path.join(output_dir, f"{csv_file.stem}_chunks.csv")
            with open(output_file, "wb") as f:
                f.write(export_response.content)
            
            # Store results
            results.append({
                "file": csv_file.name,
                "rows": result["summary"]["rows"],
                "chunks": result["summary"]["chunks"],
                "time": elapsed,
                "output": output_file
            })
            
            print(f"   Success: {result['summary']['chunks']} chunks in {elapsed:.2f}s")
            
        except Exception as e:
            print(f"   Failed: {e}")
            results.append({
                "file": csv_file.name,
                "error": str(e)
            })
    
    # Summary
    print("\n=== Summary ===")
    successful = [r for r in results if "error" not in r]
    print(f"Processed: {len(successful)}/{len(csv_files)} files")
    print(f"Total chunks: {sum(r['chunks'] for r in successful)}")
    print(f"Total time: {sum(r['time'] for r in successful):.2f}s")
    
    return results

# Usage
results = process_directory("./data/csv_files", output_dir="./data/results")
```

---

### Example 10: Integration with Pandas

```python
import requests
import pandas as pd
import io

BASE_URL = "http://127.0.0.1:8001"

# Create sample data
df = pd.DataFrame({
    "id": range(1, 101),
    "product": [f"Product {i}" for i in range(1, 101)],
    "description": [f"Description for product {i}" for i in range(1, 101)],
    "category": ["Electronics", "Furniture", "Clothing"] * 33 + ["Electronics"],
    "price": [100 + i * 10 for i in range(100)]
})

# Save to CSV in memory
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
csv_buffer.seek(0)

# Process the data
url = f"{BASE_URL}/run_fast"
files = {"file": ("data.csv", csv_buffer.getvalue().encode())}
data = {"use_turbo": "true"}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Processed {result['summary']['rows']} rows")
print(f"Created {result['summary']['chunks']} chunks")

# Retrieve and analyze
query_url = f"{BASE_URL}/retrieve"
queries = ["electronic devices", "office furniture", "clothing items"]

for query in queries:
    data = {"query": query, "k": 3}
    response = requests.post(query_url, data=data)
    results = response.json()
    
    print(f"\nQuery: '{query}'")
    print(f"Top result: {results['results'][0]['content'][:80]}...")
    print(f"Similarity: {results['results'][0]['similarity']:.3f}")
```

---

##  Utility Functions

```python
import requests
import time
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    """Decorator to retry failed requests"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(delay * (attempt + 1))
                    else:
                        raise
        return wrapper
    return decorator

@retry_on_failure(max_retries=3)
def robust_process(file_path):
    """Process file with automatic retry"""
    url = "http://127.0.0.1:8001/run_fast"
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files, data={"use_turbo": "true"})
        response.raise_for_status()
        return response.json()
```

---

**Next**: See [cURL Examples](./curl-examples.md) for command-line usage.

