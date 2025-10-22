// src/services/deepConfig.service.js
import api from './api';
import useUIStore from '../stores/uiStore';

export const deepConfigService = {
  /**
   * Step 1: Preprocess data (load and basic cleaning)
   */
  async preprocess(file, dbConfig = null) {
    const formData = new FormData();
    
    if (file) {
      formData.append('file', file);
    }
    
    if (dbConfig && dbConfig.dbType && dbConfig.host && dbConfig.tableName) {
      formData.append('db_type', dbConfig.dbType);
      formData.append('host', dbConfig.host);
      formData.append('port', dbConfig.port);
      formData.append('username', dbConfig.username);
      formData.append('password', dbConfig.password);
      formData.append('database', dbConfig.database);
      formData.append('table_name', dbConfig.tableName);
    }
    
    const response = await api.post('/deep_config/preprocess', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      }
    });
    
    return response.data;
  },

  /**
   * Step 2: Convert column data types
   */
  async typeConvert(typeConversions) {
    const formData = new FormData();
    formData.append('type_conversions', JSON.stringify(typeConversions));
    
    const response = await api.post('/deep_config/type_convert', formData);
    return response.data;
  },

  /**
   * Step 3: Handle null values
   */
  async nullHandle(nullStrategies) {
    const formData = new FormData();
    formData.append('null_strategies', JSON.stringify(nullStrategies));
    
    const response = await api.post('/deep_config/null_handle', formData);
    return response.data;
  },

  /**
   * Step 4: Remove duplicate rows
   */
  async duplicates(strategy) {
    const formData = new FormData();
    formData.append('strategy', strategy);
    
    const response = await api.post('/deep_config/duplicates', formData);
    return response.data;
  },

  /**
   * Step 5: Remove stop words
   */
  async stopwords(removeStopwords) {
    const formData = new FormData();
    formData.append('remove_stopwords', removeStopwords);
    
    const response = await api.post('/deep_config/stopwords', formData);
    return response.data;
  },

  /**
   * Step 6: Text normalization
   */
  async normalize(textProcessing) {
    const formData = new FormData();
    formData.append('text_processing', textProcessing);
    
    const response = await api.post('/deep_config/normalize', formData);
    return response.data;
  },

  /**
   * Step 7: Create text chunks with metadata
   */
  async chunk(config) {
    const formData = new FormData();
    formData.append('chunk_method', config.method);
    formData.append('chunk_size', config.chunkSize);
    formData.append('overlap', config.overlap);
    formData.append('key_column', config.keyColumn || '');
    formData.append('token_limit', config.tokenLimit);
    formData.append('n_clusters', config.nClusters);
    formData.append('store_metadata', config.storeMetadata);
    formData.append('selected_numeric_columns', JSON.stringify(config.numericColumns));
    formData.append('selected_categorical_columns', JSON.stringify(config.categoricalColumns));
    
    const response = await api.post('/deep_config/chunk', formData);
    return response.data;
  },

  /**
   * Step 8: Generate embeddings
   */
  async embed(config) {
    const formData = new FormData();
    formData.append('model_name', config.model);
    formData.append('batch_size', config.batchSize);
    formData.append('use_parallel', config.useParallel);
    
    if (config.openaiApiKey) {
      formData.append('openai_api_key', config.openaiApiKey);
    }
    if (config.openaiBaseUrl) {
      formData.append('openai_base_url', config.openaiBaseUrl);
    }
    
    const response = await api.post('/deep_config/embed', formData);
    return response.data;
  },

  /**
   * Step 9: Store vectors
   */
  async store(config) {
    const formData = new FormData();
    formData.append('storage_type', config.type);
    formData.append('collection_name', config.collectionName);
    
    const response = await api.post('/deep_config/store', formData);
    return response.data;
  },

  /**
   * Get metadata columns for chunking
   */
  async getMetadataColumns() {
    const response = await api.get('/deep_config/metadata_columns');
    return response.data;
  },

  async analyzeNulls() {
    const response = await api.get('/deep_config/analyze_nulls');
    return response.data;
  },

  async analyzeDuplicates() {
    const response = await api.get('/deep_config/analyze_duplicates');
    return response.data;
  },

  /**
   * Get null profile for current data
   */
  async getNullProfile() {
    const response = await api.get('/deep_config/null_profile');
    return response.data;
  },

  /**
   * Get duplicate analysis
   */
  async getDuplicateAnalysis() {
    const response = await api.get('/deep_config/duplicate_analysis');
    return response.data;
  },

  /**
   * Run complete deep config pipeline (alternative to step-by-step)
   */
  async runCompletePipeline(file, config, dbConfig = null) {
    const formData = new FormData();
    
    if (file) {
      formData.append('file', file);
    }
    
    if (dbConfig && dbConfig.dbType && dbConfig.host && dbConfig.tableName) {
      formData.append('db_type', dbConfig.dbType);
      formData.append('host', dbConfig.host);
      formData.append('port', dbConfig.port);
      formData.append('username', dbConfig.username);
      formData.append('password', dbConfig.password);
      formData.append('database', dbConfig.database);
      formData.append('table_name', dbConfig.tableName);
    }
    
    // Add all configuration parameters
    formData.append('preprocessing_config', JSON.stringify(config.preprocessing || {}));
    formData.append('chunking_config', JSON.stringify(config.chunking || {}));
    formData.append('embedding_config', JSON.stringify(config.embedding || {}));
    formData.append('storage_config', JSON.stringify(config.storage || {}));
    formData.append('process_large_files', config.processLargeFiles || true);
    formData.append('use_turbo', config.useTurbo || false);
    formData.append('batch_size', config.batchSize || 64);
    
    const response = await api.post('/run_deep_config', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          useUIStore.getState().setUploadProgress(progress);
        }
      }
    });
    
    return response.data;
  },

  /**
   * Export processed data
   */
  async exportData(type = 'chunks') {
    const response = await api.get(`/export/${type}`);
    return response.data;
  },

  /**
   * Get processing status
   */
  async getStatus() {
    const response = await api.get('/deep_config/status');
    return response.data;
  },

  /**
   * Reset deep config state
   */
  async reset() {
    const response = await api.post('/deep_config/reset');
    return response.data;
  }
};

export default deepConfigService;