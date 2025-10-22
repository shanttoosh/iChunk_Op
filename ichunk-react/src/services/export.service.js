// src/services/export.service.js
import api from './api';

export const exportService = {
  /**
   * Download file helper
   */
  async downloadFile(url, filename) {
    try {
      const response = await api.get(url, { responseType: 'blob' });
      const downloadUrl = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(downloadUrl);
    } catch (error) {
      console.error('Download failed:', error);
      throw error;
    }
  },

  /**
   * Export preprocessed data (Fast/Config-1 modes)
   */
  async exportPreprocessed() {
    return this.downloadFile('/export/preprocessed', 'preprocessed_data.csv');
  },

  /**
   * Export chunks (Fast/Config-1 modes)
   */
  async exportChunks() {
    return this.downloadFile('/export/chunks', 'chunks.csv');
  },

  /**
   * Export embeddings (Fast/Config-1 modes)
   */
  async exportEmbeddings() {
    return this.downloadFile('/export/embeddings_text', 'embeddings.json');
  },

  /**
   * Export Deep Config preprocessed data
   */
  async exportDeepPreprocessed() {
    return this.downloadFile('/deep_config/export/preprocessed', 'deep_preprocessed_data.csv');
  },

  /**
   * Export Deep Config chunks
   */
  async exportDeepChunks() {
    return this.downloadFile('/deep_config/export/chunks', 'deep_chunks.csv');
  },

  /**
   * Export Deep Config embeddings
   */
  async exportDeepEmbeddings() {
    return this.downloadFile('/deep_config/export/embeddings', 'deep_embeddings.json');
  },

  /**
   * Export Campaign preprocessed data
   */
  async exportCampaignPreprocessed() {
    return this.downloadFile('/campaign/export/preprocessed', 'campaign_preprocessed_data.txt');
  },

  /**
   * Export Campaign chunks
   */
  async exportCampaignChunks() {
    return this.downloadFile('/campaign/export/chunks', 'campaign_chunks.csv');
  },

  /**
   * Export Campaign embeddings
   */
  async exportCampaignEmbeddings() {
    return this.downloadFile('/campaign/export/embeddings', 'campaign_embeddings.json');
  }
};

export default exportService;


