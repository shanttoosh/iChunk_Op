// src/services/database.service.js
import api from './api';

export const databaseService = {
  /**
   * Test database connection
   */
  async testConnection(dbConfig) {
    const formData = new FormData();
    formData.append('db_type', dbConfig.dbType);
    formData.append('host', dbConfig.host);
    formData.append('port', dbConfig.port);
    formData.append('username', dbConfig.username);
    formData.append('password', dbConfig.password);
    formData.append('database', dbConfig.database);
    
    const response = await api.post('/db/test_connection', formData);
    return response.data;
  },

  /**
   * List database tables
   */
  async listTables(dbConfig) {
    const formData = new FormData();
    formData.append('db_type', dbConfig.dbType);
    formData.append('host', dbConfig.host);
    formData.append('port', dbConfig.port);
    formData.append('username', dbConfig.username);
    formData.append('password', dbConfig.password);
    formData.append('database', dbConfig.database);
    
    const response = await api.post('/db/list_tables', formData);
    return response.data;
  },

  /**
   * Import and process database table
   */
  async importTable(dbConfig, tableName, processingMode = 'fast', options = {}) {
    const formData = new FormData();
    formData.append('db_type', dbConfig.dbType);
    formData.append('host', dbConfig.host);
    formData.append('port', dbConfig.port);
    formData.append('username', dbConfig.username);
    formData.append('password', dbConfig.password);
    formData.append('database', dbConfig.database);
    formData.append('table_name', tableName);
    formData.append('processing_mode', processingMode);
    formData.append('use_turbo', options.useTurbo || false);
    formData.append('batch_size', options.batchSize || 256);
    
    const response = await api.post('/db/import_one', formData);
    return response.data;
  }
};

export default databaseService;


