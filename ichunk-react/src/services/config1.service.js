// src/services/config1.service.js
import api from './api';
import useUIStore from '../stores/uiStore';

export const config1Service = {
  /**
   * Run Config-1 Mode processing
   * @param {File} file - CSV file to process
   * @param {Object} config - Configuration options
   * @returns {Promise<Object>} Processing results
   */
  async runConfig1Mode(file, config = {}) {
    const formData = new FormData();
    
    if (file) {
      formData.append('file', file);
    }
    
    // Add database parameters if provided
    if (config.dbType && config.host && config.tableName) {
      formData.append('db_type', config.dbType);
      formData.append('host', config.host);
      formData.append('port', config.port);
      formData.append('username', config.username);
      formData.append('password', config.password);
      formData.append('database', config.database);
      formData.append('table_name', config.tableName);
    }
    
    // Chunking configuration
    formData.append('chunk_method', config.chunkMethod || 'recursive');
    formData.append('chunk_size', config.chunkSize || 400);
    formData.append('overlap', config.overlap || 50);
    formData.append('n_clusters', config.nClusters || 10);
    formData.append('document_key_column', config.documentKeyColumn || '');
    formData.append('token_limit', config.tokenLimit || 2000);
    formData.append('retrieval_metric', config.retrievalMetric || 'cosine');
    
    // Agentic chunking parameters
    if (config.chunkMethod === 'agentic') {
      formData.append('agentic_strategy', config.agenticStrategy || 'auto');
      if (config.userContext) {
        formData.append('user_context', config.userContext);
      }
    }
    
    // Model and storage configuration
    formData.append('model_choice', config.modelChoice || 'paraphrase-MiniLM-L6-v2');
    formData.append('storage_choice', config.storageChoice || 'faiss');
    formData.append('apply_default_preprocessing', config.applyDefaultPreprocessing || true);
    
    // Performance options
    formData.append('process_large_files', config.processLargeFiles || true);
    formData.append('use_turbo', config.useTurbo || false);
    formData.append('batch_size', config.batchSize || 256);
    
    const response = await api.post('/run_config1', formData, {
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
  }
};

export default config1Service;
