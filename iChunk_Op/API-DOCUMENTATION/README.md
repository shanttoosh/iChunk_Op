# Chunking Optimizer API v2.0 - Documentation

##  Overview

The **Chunking Optimizer API** is a high-performance, enterprise-grade system for processing, chunking, and vectorizing large datasets (up to 3GB+) with semantic search capabilities. It provides three processing modes tailored for different use cases, from quick prototyping to advanced customization.

##  Key Features

-  **Fast Processing**: Process 3GB+ files with optimized batch operations
-  **Three Modes**: Fast, Config-1, and Deep Config for different complexity needs
-  **Semantic Search**: Vector-based retrieval with metadata filtering
-  **Database Support**: Direct import from MySQL and PostgreSQL
-  **Flexible Chunking**: Fixed, recursive, semantic, and document-based methods
-  **OpenAI Compatible**: Standard API endpoints for easy integration
-  **Metadata Indexing**: Advanced filtering on embedded metadata

##  System Architecture

```

                        CLIENT APPLICATIONS                       
  (Web UI, Python Scripts, cURL, Postman, JavaScript Apps)      

                             
                             

                     FASTAPI REST API (Port 8001)                
               
   Fast Mode       Config-1        Deep Config           
   Endpoints       Endpoints       Endpoints             
               

                             
                             

                      BACKEND PROCESSING ENGINE                   
                
  Preprocessing    Chunking        Embedding             
     Pipeline      Strategies      Generation            
                

                             
                
                                         
  
   VECTOR DATABASES            DATA SOURCES           
  • FAISS (w/ Metadata)       • CSV Files (3GB+)      
  • ChromaDB                  • MySQL Databases       
  • Metadata Indexing         • PostgreSQL            
  
```

##  Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd chunking-optimizer

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (for NLP features)
python -m spacy download en_core_web_sm

# Start API server
python main.py
```

The API will be available at `http://127.0.0.1:8001`

### Your First API Call

```bash
# Test the API
curl http://127.0.0.1:8001/health

# Process a CSV file (Fast Mode)
curl -X POST http://127.0.0.1:8001/run_fast \
  -F "file=@data.csv" \
  -F "use_turbo=true"

# Retrieve similar chunks
curl -X POST http://127.0.0.1:8001/retrieve \
  -F "query=your search query" \
  -F "k=5"
```

##  Documentation Structure

This documentation is organized into the following sections:

### 1. **Getting Started**
- Installation guide
- Quick start tutorial
- Authentication setup (if applicable)

### 2. **Architecture**
- System overview and design
- Data flow diagrams
- Component descriptions

### 3. **API Reference**
- Complete endpoint documentation
- Request/response schemas
- Error codes and handling

### 4. **Workflows**
- Step-by-step guides for each mode
- Common use cases
- Best practices

### 5. **Examples**
- cURL examples
- Python client code
- JavaScript integration
- Postman collection

### 6. **Advanced Topics**
- Metadata filtering strategies
- Large file optimization
- Performance tuning
- OpenAI compatibility

### 7. **Reference**
- Configuration options
- Data type specifications
- Error code reference

### 8. **Appendix**
- Changelog
- FAQ
- Troubleshooting guide

##  Processing Modes

### Fast Mode
**Best for**: Quick prototyping, automatic settings
- Automatic preprocessing
- Semantic clustering
- Single API call
- ~60 seconds for 100K rows

### Config-1 Mode
**Best for**: Custom chunking requirements
- Configurable chunking methods
- Custom embedding models
- Flexible storage options
- ~90 seconds for 100K rows

### Deep Config Mode
**Best for**: Maximum control and optimization
- 8-step preprocessing pipeline
- Advanced metadata extraction
- Type conversion and null handling
- ~120 seconds for 100K rows (with all steps)

##  Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/run_fast` | POST | Fast mode processing |
| `/run_config1` | POST | Config-1 mode processing |
| `/deep_config/preprocess` | POST | Step 1: Preprocess data |
| `/deep_config/chunk` | POST | Step 6: Chunk data |
| `/deep_config/embed` | POST | Step 7: Generate embeddings |
| `/deep_config/store` | POST | Step 8: Store vectors |
| `/retrieve` | POST | Semantic search |
| `/retrieve_with_metadata` | POST | Filtered semantic search |
| `/export/chunks` | GET | Export chunks as CSV |
| `/export/embeddings` | GET | Export embeddings |

##  Performance Metrics

| Dataset Size | Fast Mode | Config-1 | Deep Config |
|--------------|-----------|----------|-------------|
| 10K rows | ~5s | ~8s | ~15s |
| 100K rows | ~60s | ~90s | ~120s |
| 1M rows | ~10min | ~15min | ~20min |
| 3GB file | ~30min | ~45min | ~60min |

*With turbo mode enabled and default settings*

##  Support

- **Documentation**: See detailed guides in respective sections
- **Examples**: Check `/05-EXAMPLES` for code samples
- **Issues**: Report bugs and feature requests to project repository
- **Contact**: [Your Contact Information]

##  License

[Your License Information]

##  Version History

- **v2.0** - Current version with metadata support and performance optimizations
- **v1.0** - Initial release

---

**Next Steps**: 
1. Read [Installation Guide](./01-GETTING-STARTED/installation.md)
2. Follow [Quick Start Tutorial](./01-GETTING-STARTED/quickstart.md)
3. Explore [API Reference](./03-API-REFERENCE/core-endpoints.md)


