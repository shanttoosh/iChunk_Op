// src/services/api.js
import axios from 'axios';
import useUIStore from '../stores/uiStore';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8001',
  timeout: 300000, // 5 minutes for large files
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    useUIStore.getState().setProcessing(true);
    return config;
  },
  (error) => {
    useUIStore.getState().setProcessing(false);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    useUIStore.getState().setProcessing(false);
    return response;
  },
  (error) => {
    useUIStore.getState().setProcessing(false);
    const message = error.response?.data?.error || 'An error occurred';
    useUIStore.getState().addNotification({
      type: 'error',
      message: message
    });
    return Promise.reject(error);
  }
);

export default api;


