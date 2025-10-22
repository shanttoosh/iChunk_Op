// src/services/campaign.service.js
import api from './api';
import useUIStore from '../stores/uiStore';

export const campaignService = {
  /**
   * Run Campaign Mode processing
   */
  async runCampaign(file, config = {}) {
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
    
    formData.append('chunk_method', config.chunkMethod || 'record_based');
    formData.append('chunk_size', config.chunkSize || '5');
    formData.append('model_choice', config.modelChoice || 'paraphrase-MiniLM-L6-v2');
    formData.append('storage_choice', config.storageChoice || 'faiss');
    formData.append('use_openai', config.useOpenai || 'false');
    formData.append('openai_api_key', config.openaiApiKey || '');
    formData.append('openai_base_url', config.openaiBaseUrl || '');
    formData.append('process_large_files', config.processLargeFiles || 'true');
    formData.append('use_turbo', config.useTurbo || 'true');
    formData.append('batch_size', config.batchSize || '256');
    formData.append('preserve_record_structure', config.preserveRecordStructure || 'true');
    formData.append('document_key_column', config.documentKeyColumn || '');
    
    const response = await api.post('/campaign/run', formData, {
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
   * Standard campaign retrieval
   */
  async retrieve(query, searchField = 'all', k = 5, includeCompleteRecords = true) {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('search_field', searchField);
    formData.append('k', k);
    formData.append('include_complete_records', includeCompleteRecords);
    
    const response = await api.post('/campaign/retrieve', formData);
    return response.data;
  },

  /**
   * Smart two-stage company retrieval
   */
  async smartRetrieval(query, searchField = 'auto', k = 5, includeCompleteRecords = true) {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('search_field', searchField);
    formData.append('k', k);
    formData.append('include_complete_records', includeCompleteRecords);
    
    const response = await api.post('/campaign/smart_retrieval', formData);
    return response.data;
  },

  /**
   * Campaign LLM answer
   */
  async llmAnswer(query) {
    const formData = new FormData();
    formData.append('query', query);
    
    const response = await api.post('/campaign/llm_answer', formData);
    return response.data;
  }
};

export default campaignService;
