# iChunk Optimizer - RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system with multiple processing modes, advanced data preprocessing, vector embeddings, and semantic search capabilities.

## 🚀 Overview

iChunk Optimizer is a full-stack RAG application that processes various data formats (CSV, database tables) through multiple pipelines, creates vector embeddings, stores them in vector databases, and provides semantic search with LLM-enhanced answers.

## 📁 Project Structure

```
Ichunk-Optimizer/
├── iChunk_Op/                          # Backend (FastAPI)
│   ├── main.py                         # Main FastAPI application
│   ├── backend.py                      # Core processing logic
│   ├── backend_campaign.py             # Campaign mode processing
│   ├── requirements.txt                # Python dependencies
│   └── README.md                       # Backend documentation
├── ichunk-react/                       # Frontend (React)
│   ├── src/
│   │   ├── components/                 # React components
│   │   │   ├── Layout/                 # Layout components
│   │   │   │   ├── Header.jsx          # Application header
│   │   │   │   ├── Sidebar.jsx         # Navigation sidebar
│   │   │   │   └── Footer.jsx          # Application footer
│   │   │   ├── Modes/                  # Processing mode components
│   │   │   │   ├── ModeSelector.jsx    # Mode selection interface
│   │   │   │   ├── FastMode.jsx        # Fast processing mode
│   │   │   │   ├── Config1Mode.jsx     # Config-1 processing mode
│   │   │   │   ├── DeepConfigMode.jsx  # Deep Config processing mode
│   │   │   │   └── CampaignMode.jsx    # Campaign processing mode
│   │   │   ├── DeepConfig/             # Deep Config specific components
│   │   │   │   ├── MetadataColumnSelector.jsx
│   │   │   │   ├── NullAnalysisTable.jsx
│   │   │   │   ├── TypeConversionConfig.jsx
│   │   │   │   └── ChunkingConfig.jsx
│   │   │   ├── Search/                 # Search components
│   │   │   │   └── SearchInterface.jsx # Semantic search interface
│   │   │   ├── Export/                 # Export components
│   │   │   │   └── ExportSection.jsx   # Data export interface
│   │   │   └── UI/                     # Reusable UI components
│   │   │       ├── Button.jsx
│   │   │       ├── Input.jsx
│   │   │       ├── Card.jsx
│   │   │       └── Modal.jsx
│   │   ├── services/                   # API service layer
│   │   │   ├── api.js                  # Base API configuration
│   │   │   ├── fastMode.service.js     # Fast mode API calls
│   │   │   ├── config1.service.js      # Config-1 API calls
│   │   │   ├── deepConfig.service.js   # Deep Config API calls
│   │   │   ├── campaign.service.js     # Campaign API calls
│   │   │   ├── retrieval.service.js    # Search API calls
│   │   │   ├── export.service.js       # Export API calls
│   │   │   └── system.service.js       # System API calls
│   │   ├── stores/                     # State management (Zustand)
│   │   │   ├── appStore.js             # Main application state
│   │   │   ├── uiStore.js              # UI state management
│   │   │   └── campaignStore.js        # Campaign-specific state
│   │   ├── utils/                      # Utility functions
│   │   │   ├── constants.js            # Application constants
│   │   │   └── formatting.js           # Data formatting utilities
│   │   └── App.jsx                     # Main React application
│   ├── public/                         # Static assets
│   ├── package.json                    # Node.js dependencies
│   └── vite.config.js                  # Vite configuration
└── README.md                           # This file
```

## 🏗️ Architecture

### Backend (FastAPI)
- **Port**: 8001
- **Framework**: FastAPI with async support
- **State Management**: Global variables with pickle persistence
- **Vector Databases**: FAISS and ChromaDB support
- **Embedding Models**: Local (sentence-transformers) and OpenAI models

### Frontend (React)
- **Port**: 5173 (Vite dev server)
- **Framework**: React 18 with Vite
- **State Management**: Zustand stores
- **UI Library**: Custom components with Tailwind CSS
- **HTTP Client**: Axios for API communication

## 🔧 Processing Modes

### 1. Fast Mode
**Purpose**: Quick processing with optimized settings
- **Preprocessing**: Basic cleaning and normalization
- **Chunking**: Semantic clustering for large datasets
- **Embedding**: Default sentence-transformers model
- **Storage**: FAISS for fast similarity search
- **Use Case**: Quick data processing with minimal configuration

### 2. Config-1 Mode
**Purpose**: Balanced configuration with moderate customization
- **Preprocessing**: Configurable cleaning options
- **Chunking**: Recursive text splitting with custom parameters
- **Embedding**: Choice of local or OpenAI models
- **Storage**: FAISS or ChromaDB
- **Use Case**: Balanced performance and customization

### 3. Deep Config Mode
**Purpose**: Advanced configuration with full control
- **Preprocessing**: 9-step pipeline with detailed control
  1. Data preprocessing
  2. Type conversion
  3. Null handling
  4. Duplicate removal
  5. Stopword removal
  6. Text normalization
  7. Chunking
  8. Embedding
  9. Storage
- **Chunking**: Multiple strategies (recursive, semantic, document-based)
- **Embedding**: Advanced model selection and batch processing
- **Storage**: FAISS or ChromaDB with metadata support
- **Use Case**: Maximum control and customization

### 4. Campaign Mode
**Purpose**: Specialized for campaign data processing
- **Preprocessing**: Campaign-specific data handling
- **Chunking**: Contact and campaign-focused chunking
- **Embedding**: Optimized for campaign data
- **Storage**: ChromaDB with campaign metadata
- **Use Case**: Marketing campaigns and contact management

## 🔄 Data Flow

### 1. Data Input
- **File Upload**: CSV files via drag-and-drop or file picker
- **Database Import**: MySQL, PostgreSQL, SQLite support
- **Large File Handling**: Automatic batching for files >100MB

### 2. Preprocessing Pipeline
- **Column Cleaning**: Remove special characters, normalize names
- **Data Cleaning**: Handle nulls, duplicates, formatting
- **Type Conversion**: Automatic and manual data type conversion
- **Text Processing**: Stopword removal, normalization, lemmatization

### 3. Chunking Strategies
- **Recursive**: Split by separators (paragraphs, sentences, words)
- **Semantic**: Cluster-based chunking for related content
- **Document-based**: Group by document or record boundaries
- **Key-Value**: Structured data chunking with metadata

### 4. Embedding Generation
- **Local Models**: sentence-transformers (paraphrase-MiniLM-L6-v2, all-MiniLM-L6-v2)
- **OpenAI Models**: text-embedding-ada-002, text-embedding-3-small
- **Batch Processing**: Optimized for large datasets
- **Parallel Processing**: Multi-threaded embedding generation

### 5. Vector Storage
- **FAISS**: Fast similarity search with CPU/GPU support
- **ChromaDB**: Persistent vector database with metadata support
- **Metadata Indexing**: Filtered search capabilities
- **State Persistence**: Automatic saving and loading

### 6. Retrieval & Search
- **Semantic Search**: Vector similarity search
- **Metadata Filtering**: Filter by numeric and categorical columns
- **LLM Enhancement**: Google Gemini integration for answer generation
- **Smart Retrieval**: Context-aware search for campaign mode

## 🎯 Key Features

### Frontend Features
- **Progressive UI**: Elements appear based on processing stage
- **Real-time Status**: Live processing status updates
- **Mode Selection**: Visual mode selector with descriptions
- **File Management**: Drag-and-drop file upload
- **Database Integration**: Visual database connection interface
- **Search Interface**: Advanced search with filters
- **Export Options**: Download processed data in multiple formats
- **Session Management**: Reset session functionality

### Backend Features
- **Async Processing**: Non-blocking API endpoints
- **State Persistence**: Automatic state saving across restarts
- **Large File Support**: Memory-efficient processing
- **Multiple Databases**: MySQL, PostgreSQL, SQLite support
- **Vector Databases**: FAISS and ChromaDB integration
- **Error Handling**: Comprehensive error management
- **Logging**: Detailed processing logs
- **Caching**: Processing result caching

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup
```bash
cd iChunk_Op
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd ichunk-react
npm install
npm run dev
```

### Access Application
- Frontend: http://localhost:5173
- Backend API: http://localhost:8001
- API Documentation: http://localhost:8001/docs

## 📊 API Endpoints

### Core Processing
- `POST /run_fast` - Fast mode processing
- `POST /run_config1` - Config-1 mode processing
- `POST /deep_config/preprocess` - Deep config preprocessing
- `POST /deep_config/chunk` - Deep config chunking
- `POST /deep_config/embed` - Deep config embedding
- `POST /deep_config/store` - Deep config storage
- `POST /campaign/run` - Campaign mode processing

### Search & Retrieval
- `POST /retrieve` - Basic semantic search
- `POST /retrieve_with_metadata` - Metadata-filtered search
- `POST /v1/retrieve` - OpenAI-style retrieval
- `POST /campaign/retrieve` - Campaign-specific search

### Export & Download
- `GET /export/chunks` - Download chunks
- `GET /export/embeddings` - Download embeddings
- `GET /export/preprocessed` - Download preprocessed data

### System Management
- `POST /api/reset-session` - Clear all backend state
- `GET /api/status` - System status check

## 🔧 Configuration

### Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=your_username
DB_PASSWORD=your_password
```

### Processing Parameters
- **Chunk Size**: 200-2000 characters (default: 400)
- **Overlap**: 0-200 characters (default: 50)
- **Batch Size**: 32-512 embeddings per batch
- **Parallel Workers**: 2-8 threads for processing

## 🛠️ Customization

### Adding New Processing Modes
1. Create new component in `ichunk-react/src/components/Modes/`
2. Add API service in `ichunk-react/src/services/`
3. Implement backend logic in `iChunk_Op/backend.py`
4. Add endpoint in `iChunk_Op/main.py`

### Adding New Chunking Strategies
1. Implement function in `iChunk_Op/backend.py`
2. Add to chunking configuration
3. Update frontend chunking options

### Adding New Embedding Models
1. Add model to embedding configuration
2. Implement model loading logic
3. Update frontend model selection

## 📈 Performance Optimization

### Large File Processing
- Automatic batching for files >100MB
- Memory-efficient processing
- Progress tracking and status updates
- Parallel processing support

### Vector Search Optimization
- FAISS indexing for fast similarity search
- Metadata filtering for reduced search space
- Batch embedding generation
- Caching of processing results

### Database Optimization
- Connection pooling
- Query optimization
- Large table detection
- Streaming data processing

## 🔒 Security Features

- Input validation and sanitization
- SQL injection prevention
- File type validation
- API rate limiting
- Error message sanitization

## 📝 Logging & Monitoring

- Comprehensive processing logs
- Error tracking and reporting
- Performance metrics
- User action logging
- System health monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the backend logs for debugging

## 🔄 Version History

- **v1.0.0**: Initial release with all processing modes
- **v1.1.0**: Added campaign mode and LLM integration
- **v1.2.0**: Enhanced UI with progressive flow
- **v1.3.0**: Added state persistence and session management

---

**iChunk Optimizer** - Your comprehensive RAG solution for data processing and semantic search.

