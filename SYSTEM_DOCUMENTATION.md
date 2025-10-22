# iChunk Optimizer - Complete System Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Design](#architecture-design)
3. [System Components](#system-components)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Processing Workflows](#processing-workflows)
6. [State Management](#state-management)
7. [API Architecture](#api-architecture)
8. [Database Integration](#database-integration)
9. [Vector Storage Systems](#vector-storage-systems)
10. [Embedding Pipeline](#embedding-pipeline)
11. [Search and Retrieval](#search-and-retrieval)
12. [User Interface Design](#user-interface-design)
13. [Error Handling and Logging](#error-handling-and-logging)
14. [Performance Optimization](#performance-optimization)
15. [Security Architecture](#security-architecture)
16. [Deployment Architecture](#deployment-architecture)
17. [Debugging Tools](#debugging-tools)
18. [Monitoring and Observability](#monitoring-and-observability)
19. [Troubleshooting Guide](#troubleshooting-guide)
20. [Development Workflow](#development-workflow)

## System Overview

iChunk Optimizer is a comprehensive Retrieval-Augmented Generation (RAG) system designed to process, embed, and search through various data formats. The system implements a multi-modal architecture supporting different processing strategies optimized for various use cases.

### Core Capabilities
- Multi-format data ingestion (CSV, database tables)
- Advanced data preprocessing pipelines
- Multiple chunking strategies
- Vector embedding generation
- Vector database storage (FAISS, ChromaDB)
- Semantic search and retrieval
- LLM-enhanced answer generation
- Real-time processing status tracking

### System Characteristics
- **Scalability**: Handles datasets from KB to GB scale
- **Flexibility**: Multiple processing modes for different requirements
- **Performance**: Optimized for both speed and accuracy
- **Extensibility**: Modular architecture for easy feature addition
- **Reliability**: Comprehensive error handling and state management

## Architecture Design

### High-Level Architecture

The system follows a client-server architecture with clear separation of concerns:

```
┌─────────────────┐    HTTP/WebSocket    ┌─────────────────┐
│   React Frontend │ ◄─────────────────► │  FastAPI Backend │
│                 │                      │                 │
│ - UI Components │                      │ - Processing    │
│ - State Mgmt    │                      │ - API Endpoints │
│ - Services      │                      │ - Business Logic │
└─────────────────┘                      └─────────────────┘
         │                                        │
         │                                        │
         ▼                                        ▼
┌─────────────────┐                      ┌─────────────────┐
│   Browser APIs  │                      │  Vector Stores  │
│ - File Upload   │                      │ - FAISS         │
│ - Local Storage │                      │ - ChromaDB      │
└─────────────────┘                      └─────────────────┘
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │   External APIs │
                                         │ - OpenAI        │
                                         │ - Database APIs │
                                         └─────────────────┘
```

### Component Architecture

#### Frontend Architecture
- **Presentation Layer**: React components with Tailwind CSS
- **State Management**: Zustand stores for global state
- **Service Layer**: Axios-based API communication
- **Utility Layer**: Helper functions and constants

#### Backend Architecture
- **API Layer**: FastAPI with async support
- **Business Logic Layer**: Processing pipelines and algorithms
- **Data Access Layer**: Database and file system operations
- **Integration Layer**: External API and service integrations

### Design Patterns

#### Frontend Patterns
- **Component Composition**: Reusable UI components
- **Service Pattern**: Centralized API communication
- **State Pattern**: Centralized state management
- **Observer Pattern**: Real-time status updates

#### Backend Patterns
- **Pipeline Pattern**: Sequential processing steps
- **Strategy Pattern**: Multiple processing strategies
- **Factory Pattern**: Model and storage creation
- **Repository Pattern**: Data access abstraction

## System Components

### Frontend Components

#### Core Application Structure
```
App.jsx
├── Layout Components
│   ├── Header.jsx
│   ├── Sidebar.jsx
│   └── Footer.jsx
├── Mode Components
│   ├── ModeSelector.jsx
│   ├── FastMode.jsx
│   ├── Config1Mode.jsx
│   ├── DeepConfigMode.jsx
│   └── CampaignMode.jsx
├── Feature Components
│   ├── Search/SearchInterface.jsx
│   ├── Export/ExportSection.jsx
│   └── DeepConfig/*.jsx
└── UI Components
    ├── Button.jsx
    ├── Input.jsx
    ├── Card.jsx
    └── Modal.jsx
```

#### State Management Architecture
```
Zustand Stores
├── appStore.js (Global Application State)
│   ├── currentMode
│   ├── processStatus
│   ├── fileInfo
│   ├── apiResults
│   └── retrievalResults
├── uiStore.js (UI State)
│   ├── sidebarCollapsed
│   ├── showDatabaseModal
│   └── showMetadataModal
└── campaignStore.js (Campaign State)
    ├── campaignResults
    ├── useSmartRetrieval
    └── campaignConfig
```

#### Service Layer Architecture
```
Services
├── api.js (Base API Configuration)
├── fastMode.service.js
├── config1.service.js
├── deepConfig.service.js
├── campaign.service.js
├── retrieval.service.js
├── export.service.js
└── system.service.js
```

### Backend Components

#### Core Processing Modules
```
Backend Modules
├── main.py (FastAPI Application)
│   ├── API Endpoints
│   ├── Middleware
│   ├── Error Handling
│   └── State Management
├── backend.py (Core Processing)
│   ├── Preprocessing Functions
│   ├── Chunking Strategies
│   ├── Embedding Generation
│   ├── Vector Storage
│   ├── Retrieval Functions
│   └── State Persistence
└── backend_campaign.py (Campaign Processing)
    ├── Campaign Pipeline
    ├── Contact Processing
    ├── Campaign Chunking
    └── Campaign Retrieval
```

#### Global State Variables
```
Global State
├── current_df (Current DataFrame)
├── current_chunks (Generated Chunks)
├── current_embeddings (Vector Embeddings)
├── current_model (Embedding Model)
├── current_store_info (Vector Store Info)
├── current_metadata (Chunk Metadata)
├── current_file_info (File Information)
└── campaign_state (Campaign State)
```

## Data Flow Architecture

### Input Data Flow
```
Data Sources
├── File Upload
│   ├── CSV Files
│   ├── Excel Files
│   └── JSON Files
├── Database Import
│   ├── MySQL Tables
│   ├── PostgreSQL Tables
│   └── SQLite Tables
└── Direct Input
    ├── Text Data
    └── Structured Data
```

### Processing Data Flow
```
Processing Pipeline
├── Data Ingestion
│   ├── File Validation
│   ├── Format Detection
│   └── Size Assessment
├── Preprocessing
│   ├── Data Cleaning
│   ├── Type Conversion
│   ├── Null Handling
│   └── Normalization
├── Chunking
│   ├── Strategy Selection
│   ├── Parameter Configuration
│   └── Chunk Generation
├── Embedding
│   ├── Model Selection
│   ├── Batch Processing
│   └── Vector Generation
├── Storage
│   ├── Vector Indexing
│   ├── Metadata Storage
│   └── Persistence
└── Retrieval
    ├── Query Processing
    ├── Similarity Search
    └── Result Ranking
```

### State Flow
```
State Management Flow
├── Frontend State
│   ├── User Interactions
│   ├── API Calls
│   ├── State Updates
│   └── UI Rendering
├── Backend State
│   ├── Processing State
│   ├── Data State
│   ├── Model State
│   └── Storage State
└── Persistence
    ├── State Serialization
    ├── File Storage
    ├── State Recovery
    └── State Clearing
```

## Processing Workflows

### Fast Mode Workflow
```
Fast Mode Pipeline
├── Input Processing
│   ├── File Upload/DB Import
│   └── Basic Validation
├── Preprocessing
│   ├── Column Cleaning
│   ├── Basic Data Cleaning
│   └── Excel Safety
├── Chunking
│   ├── Semantic Clustering
│   └── Optimized Parameters
├── Embedding
│   ├── Default Model (paraphrase-MiniLM-L6-v2)
│   └── Batch Processing
├── Storage
│   ├── FAISS Index Creation
│   └── Metadata Storage
└── Results
    ├── Processing Summary
    └── Retrieval Ready
```

### Config-1 Mode Workflow
```
Config-1 Pipeline
├── Input Processing
│   ├── File Upload/DB Import
│   └── Configuration Loading
├── Preprocessing
│   ├── Configurable Cleaning
│   ├── Type Conversion
│   └── Data Validation
├── Chunking
│   ├── Recursive Text Splitting
│   ├── Custom Size/Overlap
│   └── Key-Value Formatting
├── Embedding
│   ├── Model Selection (Local/OpenAI)
│   ├── Batch Configuration
│   └── Parallel Processing
├── Storage
│   ├── Storage Type Selection (FAISS/ChromaDB)
│   ├── Collection Management
│   └── Metadata Indexing
└── Results
    ├── Processing Summary
    ├── Storage Information
    └── Retrieval Configuration
```

### Deep Config Mode Workflow
```
Deep Config Pipeline (9 Steps)
├── Step 1: Preprocessing
│   ├── Data Loading
│   ├── Basic Cleaning
│   └── Column Analysis
├── Step 2: Type Conversion
│   ├── Automatic Detection
│   ├── Manual Override
│   └── Validation
├── Step 3: Null Handling
│   ├── Null Analysis
│   ├── Strategy Selection
│   └── Custom Values
├── Step 4: Duplicate Removal
│   ├── Duplicate Detection
│   ├── Strategy Selection
│   └── Removal Execution
├── Step 5: Stopword Processing
│   ├── Stopword Detection
│   ├── Removal Configuration
│   └── Processing
├── Step 6: Text Normalization
│   ├── Normalization Method
│   ├── Lemmatization
│   └── Processing
├── Step 7: Chunking
│   ├── Strategy Selection
│   ├── Parameter Configuration
│   └── Chunk Generation
├── Step 8: Embedding
│   ├── Model Selection
│   ├── Batch Processing
│   └── Vector Generation
└── Step 9: Storage
    ├── Storage Configuration
    ├── Vector Indexing
    └── Metadata Storage
```

### Campaign Mode Workflow
```
Campaign Pipeline
├── Input Processing
│   ├── Campaign Data Upload
│   ├── Contact Information
│   └── Campaign Metadata
├── Preprocessing
│   ├── Campaign-Specific Cleaning
│   ├── Contact Processing
│   └── Metadata Extraction
├── Chunking
│   ├── Contact-Based Chunking
│   ├── Campaign Grouping
│   └── Metadata Preservation
├── Embedding
│   ├── Campaign-Optimized Model
│   ├── Contact Embeddings
│   └── Campaign Embeddings
├── Storage
│   ├── ChromaDB Collection
│   ├── Campaign Metadata
│   └── Contact Indexing
└── Retrieval
    ├── Campaign Search
    ├── Contact Search
    └── Smart Retrieval
```

## State Management

### Frontend State Architecture

#### Global State (appStore)
```javascript
{
  currentMode: 'fast' | 'config1' | 'deep' | 'campaign',
  processStatus: {
    preprocessing: 'pending' | 'completed' | 'error',
    chunking: 'pending' | 'completed' | 'error',
    embedding: 'pending' | 'completed' | 'error',
    storage: 'pending' | 'completed' | 'error',
    retrieval: 'pending' | 'completed' | 'error'
  },
  fileInfo: {
    name: string,
    size: number,
    uploadTime: Date,
    location: string,
    largeFileProcessed: boolean,
    turboMode: boolean
  },
  apiResults: {
    mode: string,
    summary: {
      rows: number,
      chunks: number,
      stored: string,
      retrieval_ready: boolean
    }
  },
  retrievalResults: {
    query: string,
    results: Array<{
      rank: number,
      content: string,
      similarity: number,
      metadata: object
    }>
  }
}
```

#### UI State (uiStore)
```javascript
{
  sidebarCollapsed: boolean,
  showDatabaseModal: boolean,
  showMetadataModal: boolean
}
```

#### Campaign State (campaignStore)
```javascript
{
  campaignResults: object,
  useSmartRetrieval: boolean,
  campaignConfig: object
}
```

### Backend State Architecture

#### Global Variables
```python
# Core Data State
current_df = None              # Current DataFrame
current_chunks = None          # Generated chunks
current_embeddings = None      # Vector embeddings
current_metadata = None        # Chunk metadata

# Model and Storage State
current_model = None           # Embedding model
current_store_info = None      # Vector store information
current_file_info = None       # File processing info

# Campaign State
campaign_state = {
    'model': None,
    'store_info': None,
    'chunks': None,
    'embeddings': None,
    'metadata': None,
    'df': None,
    'file_info': None,
    'preprocessed_df': None,
    'distance_metric': 'cosine',
    'media_campaign_data': None,
    'contact_records': None
}
```

#### State Persistence
```python
# State File Structure
{
    'current_model': model_object,
    'current_store_info': store_info,
    'current_chunks': chunks_list,
    'current_embeddings': embeddings_array,
    'current_df': dataframe,
    'current_file_info': file_info
}
```

### State Flow Management

#### State Update Flow
```
User Action → Frontend State Update → API Call → Backend Processing → Backend State Update → State Persistence → Response → Frontend State Update → UI Update
```

#### State Recovery Flow
```
Application Start → State File Check → State Loading → Model Restoration → Storage Restoration → State Validation → Application Ready
```

#### State Clearing Flow
```
Reset Request → Backend State Clear → File System Cleanup → Frontend State Clear → UI Reset → Clean State
```

## API Architecture

### RESTful API Design

#### Endpoint Categories
```
API Endpoints
├── Processing Endpoints
│   ├── POST /run_fast
│   ├── POST /run_config1
│   ├── POST /deep_config/preprocess
│   ├── POST /deep_config/chunk
│   ├── POST /deep_config/embed
│   ├── POST /deep_config/store
│   └── POST /campaign/run
├── Search Endpoints
│   ├── POST /retrieve
│   ├── POST /retrieve_with_metadata
│   ├── POST /v1/retrieve
│   └── POST /campaign/retrieve
├── Export Endpoints
│   ├── GET /export/chunks
│   ├── GET /export/embeddings
│   └── GET /export/preprocessed
├── Database Endpoints
│   ├── POST /db/test_connection
│   ├── POST /db/list_tables
│   └── POST /db/import_one
└── System Endpoints
    ├── POST /api/reset-session
    ├── GET /api/status
    └── GET /system_info
```

#### Request/Response Patterns
```python
# Standard Request Format
{
    "file": UploadFile,
    "config": {
        "parameter1": "value1",
        "parameter2": "value2"
    }
}

# Standard Response Format
{
    "status": "success" | "error",
    "data": {
        "result": "processing_result",
        "metadata": "additional_info"
    },
    "error": "error_message" (if error)
}
```

### Async Processing Architecture

#### Async Endpoint Pattern
```python
@app.post("/endpoint")
async def endpoint_handler(request_data):
    try:
        # Input validation
        validated_data = validate_input(request_data)
        
        # Async processing
        result = await process_async(validated_data)
        
        # State update
        update_global_state(result)
        
        # Response
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"status": "error", "error": str(e)}
```

#### Error Handling Pattern
```python
# Global Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )
```

## Database Integration

### Database Connection Architecture

#### Connection Pool Management
```python
# Database Connection Classes
class DatabaseConnection:
    def __init__(self, db_type, host, port, username, password, database):
        self.db_type = db_type
        self.connection = None
        self.pool = None
    
    async def connect(self):
        # Establish connection
        pass
    
    async def disconnect(self):
        # Close connection
        pass
    
    async def execute_query(self, query):
        # Execute query
        pass
```

#### Supported Database Types
```
Database Support
├── MySQL
│   ├── Connection: PyMySQL
│   ├── Features: Full support
│   └── Limitations: None
├── PostgreSQL
│   ├── Connection: psycopg2
│   ├── Features: Full support
│   └── Limitations: None
└── SQLite
    ├── Connection: sqlite3
    ├── Features: Local only
    └── Limitations: Concurrent access
```

### Database Operations

#### Table Operations
```python
# Table Management Functions
def get_table_list(connection):
    """Get list of available tables"""
    pass

def get_table_size(connection, table_name):
    """Get table size in bytes"""
    pass

def is_large_table(connection, table_name, threshold=100):
    """Check if table is considered large"""
    pass

def import_table_to_dataframe(connection, table_name):
    """Import table data to DataFrame"""
    pass
```

#### Large Table Handling
```python
# Large Table Processing
def process_large_table(connection, table_name, processing_mode):
    """Process large table in batches"""
    if is_large_table(connection, table_name):
        return process_large_file_batch(connection, table_name, processing_mode)
    else:
        return process_table_direct(connection, table_name, processing_mode)
```

## Vector Storage Systems

### FAISS Integration

#### FAISS Index Management
```python
# FAISS Index Operations
class FAISSManager:
    def __init__(self):
        self.index = None
        self.data = {}
        self.metadata_index = {}
    
    def create_index(self, dimension, index_type="flat"):
        """Create FAISS index"""
        pass
    
    def add_vectors(self, vectors, metadata):
        """Add vectors to index"""
        pass
    
    def search(self, query_vector, k=5):
        """Search similar vectors"""
        pass
    
    def save_index(self, filepath):
        """Save index to disk"""
        pass
    
    def load_index(self, filepath):
        """Load index from disk"""
        pass
```

#### FAISS Storage Structure
```
FAISS Storage
├── Index Files
│   ├── index.faiss (FAISS index)
│   └── data.pkl (metadata)
├── Metadata
│   ├── Chunk IDs
│   ├── Content
│   └── Custom Fields
└── Configuration
    ├── Index Type
    ├── Dimensions
    └── Distance Metric
```

### ChromaDB Integration

#### ChromaDB Collection Management
```python
# ChromaDB Operations
class ChromaDBManager:
    def __init__(self):
        self.client = None
        self.collection = None
    
    def create_collection(self, name, metadata=None):
        """Create ChromaDB collection"""
        pass
    
    def add_documents(self, documents, embeddings, metadata):
        """Add documents to collection"""
        pass
    
    def query_collection(self, query_text, n_results=5):
        """Query collection for similar documents"""
        pass
    
    def get_collection_info(self):
        """Get collection information"""
        pass
```

#### ChromaDB Storage Structure
```
ChromaDB Storage
├── Collections
│   ├── Collection Name
│   ├── Documents
│   ├── Embeddings
│   └── Metadata
├── Persistence
│   ├── Local Storage
│   ├── Remote Storage
│   └── Backup
└── Configuration
    ├── Distance Metric
    ├── Embedding Function
    └── Collection Settings
```

## Embedding Pipeline

### Model Management

#### Local Model Support
```python
# Local Model Classes
class LocalEmbeddingModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load sentence-transformers model"""
        pass
    
    def encode(self, texts, batch_size=32):
        """Generate embeddings for texts"""
        pass
    
    def get_model_info(self):
        """Get model information"""
        pass
```

#### OpenAI Model Support
```python
# OpenAI Model Classes
class OpenAIEmbeddingModel:
    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
    
    def initialize_client(self):
        """Initialize OpenAI client"""
        pass
    
    def encode(self, texts, batch_size=100):
        """Generate embeddings using OpenAI API"""
        pass
    
    def get_usage_info(self):
        """Get API usage information"""
        pass
```

### Embedding Generation Process

#### Batch Processing
```python
# Batch Embedding Generation
def generate_embeddings_batch(texts, model, batch_size=32):
    """Generate embeddings in batches"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    
    return embeddings
```

#### Parallel Processing
```python
# Parallel Embedding Generation
def generate_embeddings_parallel(texts, model, num_workers=4):
    """Generate embeddings using parallel processing"""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        chunk_size = len(texts) // num_workers
        
        for i in range(num_workers):
            start = i * chunk_size
            end = start + chunk_size if i < num_workers - 1 else len(texts)
            chunk = texts[start:end]
            future = executor.submit(model.encode, chunk)
            futures.append(future)
        
        embeddings = []
        for future in futures:
            embeddings.extend(future.result())
    
    return embeddings
```

## Search and Retrieval

### Semantic Search Architecture

#### Search Query Processing
```python
# Search Query Handler
class SearchQueryProcessor:
    def __init__(self, model, store):
        self.model = model
        self.store = store
    
    def process_query(self, query, k=5, metadata_filter=None):
        """Process search query"""
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search vector store
        results = self.store.search(query_embedding, k, metadata_filter)
        
        # Format results
        formatted_results = self.format_results(results)
        
        return formatted_results
```

#### Metadata Filtering
```python
# Metadata Filter System
class MetadataFilter:
    def __init__(self, metadata_index):
        self.metadata_index = metadata_index
    
    def apply_filter(self, filter_criteria):
        """Apply metadata filter"""
        filtered_indices = set()
        
        for field, value in filter_criteria.items():
            if field in self.metadata_index:
                field_indices = self.metadata_index[field].get(str(value), set())
                if not filtered_indices:
                    filtered_indices = field_indices
                else:
                    filtered_indices = filtered_indices.intersection(field_indices)
        
        return list(filtered_indices)
```

### Retrieval Strategies

#### Basic Retrieval
```python
# Basic Similarity Search
def basic_retrieve(query, k=5):
    """Basic semantic search"""
    query_embedding = current_model.encode([query])
    results = current_store_info["index"].search(query_embedding, k)
    return format_basic_results(results)
```

#### Metadata-Filtered Retrieval
```python
# Metadata-Filtered Search
def retrieve_with_metadata(query, k=5, metadata_filter=None):
    """Search with metadata filtering"""
    query_embedding = current_model.encode([query])
    
    if metadata_filter:
        filtered_indices = apply_metadata_filter(metadata_filter)
        results = search_with_indices(query_embedding, filtered_indices, k)
    else:
        results = current_store_info["index"].search(query_embedding, k)
    
    return format_metadata_results(results)
```

#### LLM-Enhanced Retrieval
```python
# LLM-Enhanced Search
def llm_enhanced_retrieve(query, k=5):
    """Search with LLM enhancement"""
    # Get semantic search results
    search_results = basic_retrieve(query, k)
    
    # Generate LLM answer
    context = format_context(search_results)
    llm_answer = generate_llm_answer(query, context)
    
    return {
        "query": query,
        "answer": llm_answer,
        "sources": search_results,
        "metadata": {
            "search_results_count": len(search_results),
            "llm_model": "gemini-pro"
        }
    }
```

## User Interface Design

### Progressive UI Architecture

#### UI State Machine
```javascript
// UI State Machine
const UI_STATES = {
  INITIAL: 'initial',
  MODE_SELECTED: 'mode_selected',
  DATA_UPLOADED: 'data_uploaded',
  CONFIGURING: 'configuring',
  PROCESSING: 'processing',
  COMPLETED: 'completed',
  ERROR: 'error'
};

// State Transitions
const STATE_TRANSITIONS = {
  [UI_STATES.INITIAL]: [UI_STATES.MODE_SELECTED],
  [UI_STATES.MODE_SELECTED]: [UI_STATES.DATA_UPLOADED],
  [UI_STATES.DATA_UPLOADED]: [UI_STATES.CONFIGURING],
  [UI_STATES.CONFIGURING]: [UI_STATES.PROCESSING],
  [UI_STATES.PROCESSING]: [UI_STATES.COMPLETED, UI_STATES.ERROR],
  [UI_STATES.COMPLETED]: [UI_STATES.MODE_SELECTED],
  [UI_STATES.ERROR]: [UI_STATES.CONFIGURING]
};
```

#### Component Visibility Logic
```javascript
// Component Visibility Rules
const getVisibleComponents = (currentState, processingComplete) => {
  const components = {
    modeSelector: true,
    fileUpload: currentState >= UI_STATES.MODE_SELECTED,
    configuration: currentState >= UI_STATES.DATA_UPLOADED,
    processing: currentState === UI_STATES.PROCESSING,
    searchInterface: processingComplete,
    exportSection: processingComplete
  };
  
  return components;
};
```

### Responsive Design System

#### Breakpoint System
```css
/* Responsive Breakpoints */
:root {
  --breakpoint-sm: 640px;
  --breakpoint-md: 768px;
  --breakpoint-lg: 1024px;
  --breakpoint-xl: 1280px;
  --breakpoint-2xl: 1536px;
}

/* Responsive Utilities */
.container {
  width: 100%;
  margin: 0 auto;
  padding: 0 1rem;
}

@media (min-width: 640px) {
  .container { max-width: 640px; }
}

@media (min-width: 768px) {
  .container { max-width: 768px; }
}

@media (min-width: 1024px) {
  .container { max-width: 1024px; }
}
```

#### Component Responsive Patterns
```javascript
// Responsive Component Pattern
const ResponsiveComponent = ({ children, className = "" }) => {
  return (
    <div className={`
      w-full
      sm:w-11/12
      md:w-10/12
      lg:w-9/12
      xl:w-8/12
      ${className}
    `}>
      {children}
    </div>
  );
};
```

### Accessibility Architecture

#### ARIA Implementation
```javascript
// Accessibility Component Pattern
const AccessibleButton = ({ 
  children, 
  onClick, 
  disabled = false, 
  loading = false,
  ariaLabel 
}) => {
  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      aria-label={ariaLabel}
      aria-disabled={disabled || loading}
      className={`
        px-4 py-2 rounded-lg
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        ${loading ? 'animate-pulse' : ''}
      `}
    >
      {loading ? 'Loading...' : children}
    </button>
  );
};
```

#### Keyboard Navigation
```javascript
// Keyboard Navigation Hook
const useKeyboardNavigation = (elements) => {
  useEffect(() => {
    const handleKeyDown = (event) => {
      switch (event.key) {
        case 'Tab':
          // Handle tab navigation
          break;
        case 'Enter':
        case ' ':
          // Handle activation
          break;
        case 'Escape':
          // Handle escape actions
          break;
        case 'ArrowUp':
        case 'ArrowDown':
          // Handle arrow navigation
          break;
      }
    };
    
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [elements]);
};
```

## Error Handling and Logging

### Error Handling Architecture

#### Error Classification
```python
# Error Types
class ProcessingError(Exception):
    """Base class for processing errors"""
    pass

class ValidationError(ProcessingError):
    """Input validation errors"""
    pass

class ModelError(ProcessingError):
    """Model-related errors"""
    pass

class StorageError(ProcessingError):
    """Storage-related errors"""
    pass

class NetworkError(ProcessingError):
    """Network-related errors"""
    pass
```

#### Error Handling Strategy
```python
# Error Handling Decorator
def handle_errors(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError as e:
            logger.warning(f"Validation error: {str(e)}")
            return {"error": "Invalid input", "details": str(e)}
        except ModelError as e:
            logger.error(f"Model error: {str(e)}")
            return {"error": "Model processing failed", "details": str(e)}
        except StorageError as e:
            logger.error(f"Storage error: {str(e)}")
            return {"error": "Storage operation failed", "details": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": "Internal server error", "details": str(e)}
    return wrapper
```

### Logging Architecture

#### Logging Configuration
```python
# Logging Setup
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ichunk.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
```

#### Structured Logging
```python
# Structured Logging
class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_processing_step(self, step, status, details=None):
        """Log processing step with structured data"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "status": status,
            "details": details
        }
        self.logger.info(f"Processing step: {json.dumps(log_data)}")
    
    def log_performance_metric(self, operation, duration, metadata=None):
        """Log performance metrics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration": duration,
            "metadata": metadata
        }
        self.logger.info(f"Performance metric: {json.dumps(log_data)}")
```

### Frontend Error Handling

#### Error Boundary Component
```javascript
// Error Boundary
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  
  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    // Log to error reporting service
  }
  
  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <p>{this.state.error?.message}</p>
          <button onClick={() => this.setState({ hasError: false })}>
            Try again
          </button>
        </div>
      );
    }
    
    return this.props.children;
  }
}
```

#### API Error Handling
```javascript
// API Error Handler
const handleApiError = (error) => {
  if (error.response) {
    // Server responded with error status
    const { status, data } = error.response;
    
    switch (status) {
      case 400:
        return { type: 'validation', message: data.error };
      case 404:
        return { type: 'not_found', message: 'Resource not found' };
      case 500:
        return { type: 'server', message: 'Server error occurred' };
      default:
        return { type: 'unknown', message: 'An error occurred' };
    }
  } else if (error.request) {
    // Network error
    return { type: 'network', message: 'Network error occurred' };
  } else {
    // Other error
    return { type: 'unknown', message: 'An unexpected error occurred' };
  }
};
```

## Performance Optimization

### Backend Performance

#### Memory Management
```python
# Memory Management Utilities
class MemoryManager:
    @staticmethod
    def get_memory_usage():
        """Get current memory usage"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    @staticmethod
    def optimize_dataframe(df):
        """Optimize DataFrame memory usage"""
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
            elif df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
        return df
```

#### Caching Strategy
```python
# Caching Implementation
import functools
import hashlib
import pickle
import os

class CacheManager:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def cache_result(self, func):
        """Decorator to cache function results"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            
            # Check if cached result exists
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        return wrapper
    
    def _generate_cache_key(self, func_name, args, kwargs):
        """Generate cache key from function name and arguments"""
        key_data = f"{func_name}_{str(args)}_{str(kwargs)}"
        return hashlib.md5(key_data.encode()).hexdigest()
```

#### Batch Processing Optimization
```python
# Batch Processing Utilities
class BatchProcessor:
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
    
    def process_in_batches(self, data, process_func):
        """Process data in batches"""
        results = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_result = process_func(batch)
            results.extend(batch_result)
            
            # Memory cleanup
            del batch
            if i % (self.batch_size * 10) == 0:
                import gc
                gc.collect()
        
        return results
```

### Frontend Performance

#### Code Splitting
```javascript
// Code Splitting Implementation
import { lazy, Suspense } from 'react';

// Lazy load components
const DeepConfigMode = lazy(() => import('./components/Modes/DeepConfigMode'));
const CampaignMode = lazy(() => import('./components/Modes/CampaignMode'));

// Suspense wrapper
const LazyComponent = ({ component: Component, ...props }) => (
  <Suspense fallback={<div>Loading...</div>}>
    <Component {...props} />
  </Suspense>
);
```

#### Memoization
```javascript
// Memoization Hooks
import { useMemo, useCallback } from 'react';

const OptimizedComponent = ({ data, onUpdate }) => {
  // Memoize expensive calculations
  const processedData = useMemo(() => {
    return data.map(item => ({
      ...item,
      processed: expensiveCalculation(item)
    }));
  }, [data]);
  
  // Memoize callbacks
  const handleUpdate = useCallback((id, value) => {
    onUpdate(id, value);
  }, [onUpdate]);
  
  return (
    <div>
      {processedData.map(item => (
        <ItemComponent 
          key={item.id} 
          item={item} 
          onUpdate={handleUpdate} 
        />
      ))}
    </div>
  );
};
```

#### Virtual Scrolling
```javascript
// Virtual Scrolling Implementation
import { FixedSizeList as List } from 'react-window';

const VirtualizedList = ({ items, itemHeight = 50 }) => {
  const Row = ({ index, style }) => (
    <div style={style}>
      <ItemComponent item={items[index]} />
    </div>
  );
  
  return (
    <List
      height={400}
      itemCount={items.length}
      itemSize={itemHeight}
    >
      {Row}
    </List>
  );
};
```

## Security Architecture

### Backend Security

#### Input Validation
```python
# Input Validation
from pydantic import BaseModel, validator
from typing import Optional, List

class ProcessingRequest(BaseModel):
    file: Optional[str] = None
    config: dict = {}
    
    @validator('config')
    def validate_config(cls, v):
        allowed_keys = ['chunk_size', 'overlap', 'model', 'storage']
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Invalid config key: {key}")
        return v
```

#### File Upload Security
```python
# File Upload Security
import magic
import os

class FileValidator:
    ALLOWED_TYPES = ['text/csv', 'application/json', 'text/plain']
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    
    @staticmethod
    def validate_file(file):
        """Validate uploaded file"""
        # Check file size
        if file.size > FileValidator.MAX_FILE_SIZE:
            raise ValueError("File too large")
        
        # Check file type
        file_type = magic.from_buffer(file.file.read(1024), mime=True)
        if file_type not in FileValidator.ALLOWED_TYPES:
            raise ValueError("Invalid file type")
        
        # Reset file pointer
        file.file.seek(0)
        
        return True
```

#### API Security
```python
# API Security Middleware
from fastapi import Request, HTTPException
import time

class RateLimiter:
    def __init__(self, max_requests=100, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}
    
    async def check_rate_limit(self, request: Request):
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean old entries
        self.requests = {
            ip: times for ip, times in self.requests.items()
            if any(t > current_time - self.window for t in times)
        }
        
        # Check rate limit
        if client_ip in self.requests:
            if len(self.requests[client_ip]) >= self.max_requests:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            self.requests[client_ip].append(current_time)
        else:
            self.requests[client_ip] = [current_time]
```

### Frontend Security

#### XSS Prevention
```javascript
// XSS Prevention Utilities
const sanitizeInput = (input) => {
  const div = document.createElement('div');
  div.textContent = input;
  return div.innerHTML;
};

const SafeComponent = ({ userInput }) => {
  const sanitizedInput = sanitizeInput(userInput);
  return <div dangerouslySetInnerHTML={{ __html: sanitizedInput }} />;
};
```

#### CSRF Protection
```javascript
// CSRF Token Management
class CSRFManager {
  static getToken() {
    return document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
  }
  
  static addTokenToRequest(config) {
    const token = this.getToken();
    if (token) {
      config.headers['X-CSRF-Token'] = token;
    }
    return config;
  }
}
```

## Deployment Architecture

### Containerization

#### Docker Configuration
```dockerfile
# Backend Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", "main.py"]
```

```dockerfile
# Frontend Dockerfile
FROM node:16-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build

EXPOSE 5173

CMD ["npm", "run", "preview"]
```

#### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./iChunk_Op
    ports:
      - "8001:8001"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
  
  frontend:
    build: ./ichunk-react
    ports:
      - "5173:5173"
    depends_on:
      - backend
    restart: unless-stopped
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - frontend
      - backend
    restart: unless-stopped
```

### Production Configuration

#### Environment Configuration
```python
# Production Settings
class ProductionConfig:
    DEBUG = False
    LOG_LEVEL = "INFO"
    CORS_ORIGINS = ["https://yourdomain.com"]
    RATE_LIMIT = 1000
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    CACHE_TTL = 3600  # 1 hour
```

#### Nginx Configuration
```nginx
# nginx.conf
upstream backend {
    server backend:8001;
}

upstream frontend {
    server frontend:5173;
}

server {
    listen 80;
    
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Debugging Tools

### Backend Debugging

#### Debug Endpoints
```python
# Debug Endpoints
@app.get("/debug/state")
async def debug_state():
    """Debug current state"""
    return {
        "current_df": current_df is not None,
        "current_chunks": len(current_chunks) if current_chunks else 0,
        "current_embeddings": current_embeddings.shape if current_embeddings is not None else None,
        "current_model": current_model is not None,
        "current_store_info": current_store_info is not None,
        "memory_usage": get_memory_usage()
    }

@app.get("/debug/storage")
async def debug_storage():
    """Debug storage information"""
    return debug_storage_state()

@app.get("/debug/logs")
async def debug_logs():
    """Get recent logs"""
    with open('ichunk.log', 'r') as f:
        logs = f.readlines()[-100:]  # Last 100 lines
    return {"logs": logs}
```

#### Performance Profiling
```python
# Performance Profiling
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    """Profile function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        return result
    return wrapper
```

### Frontend Debugging

#### Debug Components
```javascript
// Debug Component
const DebugPanel = () => {
  const { currentMode, processStatus, apiResults } = useAppStore();
  
  return (
    <div className="debug-panel">
      <h3>Debug Information</h3>
      <div>
        <strong>Current Mode:</strong> {currentMode}
      </div>
      <div>
        <strong>Process Status:</strong>
        <pre>{JSON.stringify(processStatus, null, 2)}</pre>
      </div>
      <div>
        <strong>API Results:</strong>
        <pre>{JSON.stringify(apiResults, null, 2)}</pre>
      </div>
    </div>
  );
};
```

#### Performance Monitoring
```javascript
// Performance Monitoring
class PerformanceMonitor {
  static measureFunction(name, fn) {
    return function(...args) {
      const start = performance.now();
      const result = fn.apply(this, args);
      const end = performance.now();
      
      console.log(`${name} took ${end - start} milliseconds`);
      return result;
    };
  }
  
  static measureComponent(Component) {
    return class extends React.Component {
      componentDidMount() {
        console.log(`${Component.name} mounted`);
      }
      
      componentWillUnmount() {
        console.log(`${Component.name} unmounted`);
      }
      
      render() {
        return <Component {...this.props} />;
      }
    };
  }
}
```

## Monitoring and Observability

### Application Metrics

#### Backend Metrics
```python
# Metrics Collection
class MetricsCollector:
    def __init__(self):
        self.metrics = {}
    
    def record_processing_time(self, operation, duration):
        """Record processing time metric"""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
    
    def record_memory_usage(self, operation, memory_mb):
        """Record memory usage metric"""
        if f"{operation}_memory" not in self.metrics:
            self.metrics[f"{operation}_memory"] = []
        self.metrics[f"{operation}_memory"].append(memory_mb)
    
    def get_metrics_summary(self):
        """Get metrics summary"""
        summary = {}
        for operation, values in self.metrics.items():
            summary[operation] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }
        return summary
```

#### Frontend Metrics
```javascript
// Frontend Metrics
class FrontendMetrics {
  static trackPageView(page) {
    console.log(`Page view: ${page}`);
    // Send to analytics service
  }
  
  static trackUserAction(action, details = {}) {
    console.log(`User action: ${action}`, details);
    // Send to analytics service
  }
  
  static trackPerformance(name, duration) {
    console.log(`Performance: ${name} - ${duration}ms`);
    // Send to performance monitoring service
  }
}
```

### Health Checks

#### Backend Health Check
```python
# Health Check Endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {}
    }
    
    # Check database connections
    try:
        # Test database connection
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check vector stores
    try:
        if current_store_info:
            health_status["checks"]["vector_store"] = "healthy"
        else:
            health_status["checks"]["vector_store"] = "no_store"
    except Exception as e:
        health_status["checks"]["vector_store"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check memory usage
    memory_usage = get_memory_usage()
    health_status["checks"]["memory"] = f"{memory_usage}MB"
    
    return health_status
```

#### Frontend Health Check
```javascript
// Frontend Health Check
const healthCheck = async () => {
  try {
    const response = await fetch('/api/health');
    const health = await response.json();
    
    if (health.status === 'healthy') {
      console.log('Backend is healthy');
      return true;
    } else {
      console.error('Backend health issues:', health.checks);
      return false;
    }
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};
```

## Troubleshooting Guide

### Common Issues

#### Backend Issues

**Issue: Memory Usage Too High**
```
Symptoms: Slow processing, out of memory errors
Solutions:
1. Reduce batch size in processing
2. Use large file processing mode
3. Clear state more frequently
4. Optimize DataFrame memory usage
```

**Issue: Model Loading Failures**
```
Symptoms: Embedding generation fails
Solutions:
1. Check internet connection
2. Verify model names
3. Clear model cache
4. Use alternative models
```

**Issue: Vector Store Corruption**
```
Symptoms: Search returns no results, storage errors
Solutions:
1. Reset session to clear state
2. Rebuild vector store
3. Check file permissions
4. Verify storage configuration
```

#### Frontend Issues

**Issue: API Calls Failing**
```
Symptoms: Network errors, 404/500 responses
Solutions:
1. Check backend server status
2. Verify API endpoints
3. Check CORS configuration
4. Validate request format
```

**Issue: State Not Updating**
```
Symptoms: UI not reflecting changes
Solutions:
1. Check Zustand store updates
2. Verify component re-renders
3. Check state persistence
4. Clear browser cache
```

**Issue: File Upload Failures**
```
Symptoms: Files not uploading, validation errors
Solutions:
1. Check file size limits
2. Verify file types
3. Check network connectivity
4. Validate file format
```

### Debug Procedures

#### Backend Debugging Steps
1. Check logs for error messages
2. Verify global state variables
3. Test individual processing steps
4. Check file system permissions
5. Validate input data format

#### Frontend Debugging Steps
1. Check browser console for errors
2. Verify API responses
3. Check component state
4. Test API endpoints directly
5. Validate data flow

### Performance Troubleshooting

#### Slow Processing
```
Diagnosis Steps:
1. Check memory usage
2. Monitor CPU usage
3. Analyze processing logs
4. Check batch sizes
5. Verify model performance

Solutions:
1. Increase batch size
2. Use parallel processing
3. Optimize data types
4. Clear unnecessary state
5. Use faster models
```

#### High Memory Usage
```
Diagnosis Steps:
1. Monitor memory usage over time
2. Check for memory leaks
3. Analyze data sizes
4. Check caching behavior

Solutions:
1. Reduce batch sizes
2. Clear caches more frequently
3. Use streaming processing
4. Optimize data structures
5. Implement memory limits
```

## Development Workflow

### Development Environment Setup

#### Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_api_key"
export DEBUG=True

# Run development server
python main.py
```

#### Frontend Setup
```bash
# Install dependencies
npm install

# Set environment variables
export VITE_API_BASE_URL="http://localhost:8001"

# Run development server
npm run dev
```

### Development Guidelines

#### Code Style
- Follow PEP 8 for Python code
- Use ESLint for JavaScript code
- Implement proper error handling
- Add comprehensive logging
- Write meaningful comments

#### Testing Strategy
- Unit tests for individual functions
- Integration tests for API endpoints
- End-to-end tests for user workflows
- Performance tests for large datasets
- Error handling tests

#### Version Control
- Use meaningful commit messages
- Create feature branches
- Implement code reviews
- Maintain changelog
- Tag releases

### Deployment Process

#### Staging Deployment
1. Run full test suite
2. Deploy to staging environment
3. Perform integration testing
4. Validate performance metrics
5. Check security compliance

#### Production Deployment
1. Create production build
2. Run security scans
3. Deploy to production
4. Monitor system health
5. Validate functionality

This comprehensive documentation provides a complete understanding of the iChunk Optimizer system architecture, workflows, and operational procedures. It serves as a reference for developers, system administrators, and users working with the system.

