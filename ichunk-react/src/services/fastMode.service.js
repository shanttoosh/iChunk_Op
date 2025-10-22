// src/services/fastMode.service.js
import api from './api';
import useUIStore from '../stores/uiStore';

export const fastModeService = {
  /**
   * Run Fast Mode processing
   * @param {File} file - CSV file to process
   * @param {Object} options - Processing options
   * @returns {Promise<Object>} Processing results
   */
  async runFastMode(file, options = {}) {
    const formData = new FormData();
    
    if (file) {
      formData.append('file', file);
    }
    
    // Add database parameters if provided
    if (options.dbType && options.host && options.tableName) {
      formData.append('db_type', options.dbType);
      formData.append('host', options.host);
      formData.append('port', options.port);
      formData.append('username', options.username);
      formData.append('password', options.password);
      formData.append('database', options.database);
      formData.append('table_name', options.tableName);
    }
    
    formData.append('process_large_files', options.processLargeFiles || true);
    formData.append('use_turbo', options.useTurbo || true);
    formData.append('batch_size', options.batchSize || 256);
    
    const response = await api.post('/run_fast', formData, {
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

export default fastModeService;
