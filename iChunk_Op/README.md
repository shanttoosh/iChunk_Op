# iChunk Optimizer - Backend Documentation

## Overview

The backend is a FastAPI-based Python application that provides the core processing engine for the iChunk Optimizer RAG system. It handles data preprocessing, chunking, embedding generation, vector storage, and semantic search functionality.

## Architecture

- **Framework**: FastAPI with async support
- **Port**: 8001
- **State Management**: Global variables with pickle persistence
- **Vector Databases**: FAISS and ChromaDB support
- **Embedding Models**: Local (sentence-transformers) and OpenAI models
- **Database Support**: MySQL, PostgreSQL, SQLite

## File Structure

```
iChunk_Op/
├── main.py                 # FastAPI application entry point
├── backend.py              # Core processing logic and functions
├── backend_campaign.py     # Campaign mode specific processing
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This documentation
```

## Core Files

### main.py
The main FastAPI application that defines all API endpoints and orchestrates the processing pipeline.

**Key Functions:**
- `clear_all_backend_state()`: Clears all global state variables and files
- `clear_deep_config_state()`: Clears Deep Config specific state
- `reset_session()`: API endpoint for resetting backend state

**API Endpoints:**
- `/run_fast` - Fast mode processing
- `/run_config1` - Config-1 mode processing
- `/deep_config/preprocess` - Deep config preprocessing
- `/deep_config/chunk` - Deep config chunking
- `/deep_config/embed` - Deep config embedding
- `/deep_config/store` - Deep config storage
- `/campaign/run` - Campaign mode processing
- `/retrieve` - Basic semantic search
- `/retrieve_with_metadata` - Metadata-filtered search
- `/export/chunks` - Download chunks
- `/export/embeddings` - Download embeddings
- `/api/reset-session` - Reset all backend state

### backend.py
Contains the core processing logic for all modes except Campaign mode.

**Global Variables:**
- `current_df`: Current DataFrame being processed
- `current_chunks`: Generated text chunks
- `current_embeddings`: Vector embeddings
- `current_model`: Embedding model instance
- `current_store_info`: Vector store information
- `current_metadata`: Chunk metadata
- `current_file_info`: File processing information

**Key Functions:**

**Preprocessing:**
- `preprocess_default()`: Basic data cleaning and normalization
- `apply_null_strategies_enhanced()`: Advanced null value handling
- `remove_duplicates()`: Duplicate row removal
- `remove_stopwords()`: Stopword removal
- `normalize_text()`: Text normalization and lemmatization

**Chunking:**
- `chunk_recursive()`: Recursive text splitting
- `chunk_recursive_keyvalue()`: Key-value format chunking
- `chunk_semantic_cluster()`: Semantic clustering chunking
- `chunk_document_based()`: Document-based chunking
- `RecursiveCharacterTextSplitter`: Custom text splitter class

**Embedding:**
- `generate_embeddings()`: Generate vector embeddings
- `load_embedding_model()`: Load embedding models
- `generate_embeddings_openai()`: OpenAI embedding generation

**Storage:**
- `store_faiss()`: FAISS vector storage
- `store_chroma()`: ChromaDB vector storage
- `store_faiss_enhanced()`: Enhanced FAISS with metadata
- `store_chroma_enhanced()`: Enhanced ChromaDB with metadata

**Retrieval:**
- `retrieve_similar()`: Basic semantic search
- `query_faiss_with_metadata()`: FAISS search with metadata filtering
- `query_chroma_with_metadata()`: ChromaDB search with metadata filtering

**State Management:**
- `save_state()`: Save current state to disk
- `load_state()`: Load state from disk
- `debug_storage_state()`: Debug storage information

**Pipeline Functions:**
- `run_fast_pipeline()`: Fast mode processing pipeline
- `run_config1_pipeline()`: Config-1 mode processing pipeline
- `run_deep_config_pipeline()`: Deep config processing pipeline

### backend_campaign.py
Specialized processing logic for Campaign mode, optimized for marketing campaign data.

**Global Variables:**
- `campaign_state`: Campaign-specific state dictionary

**Key Functions:**
- `run_campaign_pipeline()`: Campaign mode processing pipeline
- `campaign_process_file_direct()`: Direct file processing for campaigns
- `campaign_chunk_contacts()`: Contact-based chunking
- `campaign_generate_embeddings()`: Campaign-specific embedding generation
- `campaign_store_vectors()`: Campaign vector storage
- `campaign_retrieve()`: Campaign-specific retrieval

## Processing Modes

### Fast Mode
Quick processing with optimized settings for rapid data processing.

**Pipeline:**
1. Data loading and basic cleaning
2. Semantic clustering chunking
3. Default embedding model
4. FAISS storage

**Configuration:**
- Chunking: Semantic clustering
- Embedding: paraphrase-MiniLM-L6-v2
- Storage: FAISS
- Preprocessing: Basic cleaning only

### Config-1 Mode
Balanced configuration with moderate customization options.

**Pipeline:**
1. Configurable preprocessing
2. Recursive text splitting
3. Model selection (local or OpenAI)
4. Storage selection (FAISS or ChromaDB)

**Configuration Options:**
- Chunking: Recursive with custom size/overlap
- Embedding: Local or OpenAI models
- Storage: FAISS or ChromaDB
- Preprocessing: Configurable cleaning options

### Deep Config Mode
Advanced configuration with full control over the processing pipeline.

**9-Step Pipeline:**
1. **Preprocessing**: Data loading and basic cleaning
2. **Type Conversion**: Automatic and manual data type conversion
3. **Null Handling**: Advanced null value strategies
4. **Duplicate Removal**: Duplicate row detection and removal
5. **Stopword Removal**: Stopword filtering and removal
6. **Text Normalization**: Text normalization and lemmatization
7. **Chunking**: Multiple chunking strategies
8. **Embedding**: Advanced embedding generation
9. **Storage**: Vector database storage with metadata

**Chunking Strategies:**
- Recursive: Split by separators (paragraphs, sentences, words)
- Semantic: Cluster-based chunking for related content
- Document-based: Group by document or record boundaries
- Key-Value: Structured data chunking with metadata

**Embedding Options:**
- Local Models: sentence-transformers models
- OpenAI Models: text-embedding-ada-002, text-embedding-3-small
- Batch Processing: Optimized for large datasets
- Parallel Processing: Multi-threaded generation

### Campaign Mode
Specialized for marketing campaign data processing.

**Pipeline:**
1. Campaign-specific preprocessing
2. Contact-based chunking
3. Campaign-optimized embedding
4. ChromaDB storage with campaign metadata

**Features:**
- Contact record processing
- Campaign metadata extraction
- Marketing-specific chunking
- Campaign-aware retrieval

## Database Support

### Supported Databases
- **MySQL**: Full support with connection pooling
- **PostgreSQL**: Full support with connection pooling
- **SQLite**: Local database support

### Database Functions
- `connect_mysql()`: MySQL connection
- `connect_postgresql()`: PostgreSQL connection
- `get_table_list()`: List available tables
- `import_table_to_dataframe()`: Import table data
- `get_table_size()`: Get table size information
- `is_large_table()`: Check if table is large

## Vector Databases

### FAISS
Fast similarity search with CPU/GPU support.

**Features:**
- High-performance similarity search
- CPU and GPU support
- Index persistence
- Metadata support

**Functions:**
- `store_faiss()`: Basic FAISS storage
- `store_faiss_enhanced()`: Enhanced FAISS with metadata
- `query_faiss_with_metadata()`: Metadata-filtered search

### ChromaDB
Persistent vector database with metadata support.

**Features:**
- Persistent storage
- Rich metadata support
- Collection management
- Distance metrics

**Functions:**
- `store_chroma()`: Basic ChromaDB storage
- `store_chroma_enhanced()`: Enhanced ChromaDB with metadata
- `query_chroma_with_metadata()`: Metadata-filtered search

## Embedding Models

### Local Models
- **paraphrase-MiniLM-L6-v2**: Default model for general use
- **all-MiniLM-L6-v2**: Alternative model for different domains

### OpenAI Models
- **text-embedding-ada-002**: OpenAI's standard embedding model
- **text-embedding-3-small**: Smaller, faster OpenAI model

## State Management

### Global State Variables
- `current_df`: Current DataFrame
- `current_chunks`: Generated chunks
- `current_embeddings`: Vector embeddings
- `current_model`: Embedding model
- `current_store_info`: Vector store info
- `current_metadata`: Chunk metadata
- `current_file_info`: File information

### State Persistence
- **File**: `current_state.pkl`
- **Function**: `save_state()` and `load_state()`
- **Automatic**: State saved after each processing step
- **Recovery**: State loaded on application startup

### State Clearing
- **Function**: `clear_all_backend_state()`
- **Files Removed**: All state files and cache directories
- **Variables**: All global variables reset to None

## Error Handling

### Exception Management
- Comprehensive try-catch blocks
- Detailed error logging
- Graceful degradation
- User-friendly error messages

### Logging
- Structured logging with timestamps
- Different log levels (INFO, WARNING, ERROR)
- Processing step tracking
- Performance metrics

## Performance Optimization

### Large File Handling
- **Threshold**: 100MB automatic batching
- **Memory Management**: Efficient memory usage
- **Progress Tracking**: Real-time progress updates
- **Parallel Processing**: Multi-threaded operations

### Caching
- **Processing Cache**: `processing_cache/` directory
- **Model Cache**: Embedding model caching
- **Result Cache**: Processing result caching

### Batch Processing
- **Embedding Batches**: Configurable batch sizes
- **Chunking Batches**: Large dataset chunking
- **Storage Batches**: Efficient vector storage

## API Documentation

### Request/Response Formats
- **Input**: FormData for file uploads, JSON for configuration
- **Output**: JSON responses with processing results
- **Error**: JSON error responses with details

### Authentication
- Currently no authentication required
- API key support for OpenAI models
- Database credentials for database connections

### Rate Limiting
- No built-in rate limiting
- Relies on FastAPI's built-in request handling
- Can be extended with middleware

## Dependencies

### Core Dependencies
- **fastapi**: Web framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities
- **sentence-transformers**: Local embedding models
- **faiss-cpu**: FAISS vector search
- **chromadb**: ChromaDB vector database

### Database Dependencies
- **pymysql**: MySQL connector
- **psycopg2**: PostgreSQL connector
- **sqlite3**: SQLite support (built-in)

### Optional Dependencies
- **openai**: OpenAI API client
- **faiss-gpu**: GPU-accelerated FAISS

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps
1. Navigate to the backend directory
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python main.py`

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for OpenAI models
- `OPENAI_BASE_URL`: OpenAI API base URL (optional)

## Configuration

### Processing Parameters
- **Chunk Size**: 200-2000 characters (default: 400)
- **Overlap**: 0-200 characters (default: 50)
- **Batch Size**: 32-512 embeddings per batch
- **Parallel Workers**: 2-8 threads for processing

### Storage Configuration
- **FAISS**: Index type and parameters
- **ChromaDB**: Collection settings and distance metrics
- **Cache**: Cache directory and size limits

## Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce batch size or use large file processing
2. **Model Loading**: Ensure internet connection for model downloads
3. **Database Connection**: Check credentials and network connectivity
4. **State Corruption**: Use reset session endpoint

### Debug Tools
- `debug_storage_state()`: Check storage status
- Logging: Enable detailed logging for debugging
- State Inspection: Check global variables

## Development

### Adding New Features
1. Define new functions in appropriate files
2. Add API endpoints in main.py
3. Update state management if needed
4. Add error handling and logging
5. Test with different data types

### Testing
- Unit tests for individual functions
- Integration tests for full pipelines
- Performance tests for large datasets
- Error handling tests

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Document functions with docstrings
- Use meaningful variable names