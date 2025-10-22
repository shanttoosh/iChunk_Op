// src/services/system.service.js
import api from './api';

export const systemService = {
  /**
   * Get system information
   */
  async getSystemInfo() {
    const response = await api.get('/system_info');
    return response.data;
  },

  /**
   * Get file information
   */
  async getFileInfo() {
    const response = await api.get('/file_info');
    return response.data;
  },

  /**
   * Reset backend session (clear all server-side state)
   */
  async resetSession() {
    const response = await api.post('/api/reset-session');
    return response.data;
  },

  /**
   * Health check
   */
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  /**
   * Get API capabilities
   */
  async getCapabilities() {
    const response = await api.get('/capabilities');
    return response.data;
  }
};

export default systemService;


