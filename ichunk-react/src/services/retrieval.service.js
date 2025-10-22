// src/services/retrieval.service.js
import api from './api';

export const retrievalService = {
  /**
   * Basic retrieval
   */
  async retrieve(query, k = 5) {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('k', k);
    
    const response = await api.post('/retrieve', formData);
    return response.data;
  },

  /**
   * Retrieval with metadata filtering
   */
  async retrieveWithMetadata(query, k = 5, metadataFilter = {}) {
    const formData = new FormData();
    formData.append('query', query);
    formData.append('k', k);
    formData.append('metadata_filter', JSON.stringify(metadataFilter));
    
    const response = await api.post('/retrieve_with_metadata', formData);
    return response.data;
  },

  /**
   * LLM answer generation
   */
  async llmAnswer(query) {
    const formData = new FormData();
    formData.append('query', query);
    
    const response = await api.post('/llm/answer', formData);
    return response.data;
  }
};

export default retrievalService;


