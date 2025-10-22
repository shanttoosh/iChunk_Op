// src/utils/constants.js

// API Configuration
export const API_CONFIG = {
  BASE_URL: 'http://127.0.0.1:8001',
  TIMEOUT: 300000, // 5 minutes
  MAX_FILE_SIZE: 3 * 1024 * 1024 * 1024, // 3GB
};

// Processing Modes
export const PROCESSING_MODES = {
  FAST: 'fast',
  CONFIG1: 'config1',
  DEEP: 'deep',
  CAMPAIGN: 'campaign'
};

// Chunking Methods
export const CHUNKING_METHODS = {
  FIXED: 'fixed',
  RECURSIVE: 'recursive',
  SEMANTIC: 'semantic',
  DOCUMENT: 'document',
  RECORD_BASED: 'record_based',
  COMPANY_BASED: 'company_based',
  SOURCE_BASED: 'source_based',
  SEMANTIC_CLUSTERING: 'semantic_clustering'
};

// Storage Types
export const STORAGE_TYPES = {
  FAISS: 'faiss',
  CHROMA: 'chroma'
};

// Embedding Models
export const EMBEDDING_MODELS = {
  PARAPHRASE_MINI: 'paraphrase-MiniLM-L6-v2',
  ALL_MINI: 'all-MiniLM-L6-v2',
  PARAPHRASE_MPNET: 'paraphrase-mpnet-base-v2',
  OPENAI_ADA: 'text-embedding-ada-002'
};

// Retrieval Metrics
export const RETRIEVAL_METRICS = {
  COSINE: 'cosine',
  EUCLIDEAN: 'euclidean',
  DOT_PRODUCT: 'dot_product'
};

// Database Types
export const DATABASE_TYPES = {
  MYSQL: 'mysql',
  POSTGRESQL: 'postgresql'
};

// Process Steps
export const PROCESS_STEPS = {
  PREPROCESSING: 'preprocessing',
  CHUNKING: 'chunking',
  EMBEDDING: 'embedding',
  STORAGE: 'storage',
  RETRIEVAL: 'retrieval'
};

// Step Status
export const STEP_STATUS = {
  PENDING: 'pending',
  RUNNING: 'running',
  COMPLETED: 'completed',
  ERROR: 'error'
};

// LLM Modes
export const LLM_MODES = {
  NORMAL: 'Normal Retrieval',
  ENHANCED: 'LLM Enhanced'
};

// Default Values
export const DEFAULT_VALUES = {
  CHUNK_SIZE: 400,
  OVERLAP: 50,
  N_CLUSTERS: 10,
  TOKEN_LIMIT: 2000,
  BATCH_SIZE: 256,
  RETRIEVAL_K: 5,
  MAX_RETRIEVAL_K: 20
};

// File Upload Configuration
export const FILE_UPLOAD_CONFIG = {
  ACCEPTED_TYPES: '.csv',
  MAX_SIZE: API_CONFIG.MAX_FILE_SIZE,
  MULTIPLE: false
};

// UI Configuration
export const UI_CONFIG = {
  NOTIFICATION_DURATION: 5000,
  DEBOUNCE_DELAY: 300,
  ANIMATION_DURATION: 200
};


