// src/utils/validation.js

/**
 * Validate CSV file
 * @param {File} file - File to validate
 * @returns {Object} Validation result
 */
export const validateCSVFile = (file) => {
  const errors = [];
  
  if (!file) {
    errors.push('No file selected');
    return { isValid: false, errors };
  }
  
  // Check file type
  if (!file.name.toLowerCase().endsWith('.csv')) {
    errors.push('File must be a CSV file');
  }
  
  // Check file size (3GB limit)
  const maxSize = 3 * 1024 * 1024 * 1024; // 3GB
  if (file.size > maxSize) {
    errors.push('File size must be less than 3GB');
  }
  
  // Check if file is empty
  if (file.size === 0) {
    errors.push('File cannot be empty');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

/**
 * Validate database configuration
 * @param {Object} config - Database configuration
 * @returns {Object} Validation result
 */
export const validateDatabaseConfig = (config) => {
  const errors = [];
  
  if (!config.dbType) {
    errors.push('Database type is required');
  }
  
  if (!config.host) {
    errors.push('Host is required');
  }
  
  if (!config.port) {
    errors.push('Port is required');
  } else if (isNaN(config.port) || config.port < 1 || config.port > 65535) {
    errors.push('Port must be a valid number between 1 and 65535');
  }
  
  if (!config.username) {
    errors.push('Username is required');
  }
  
  if (!config.password) {
    errors.push('Password is required');
  }
  
  if (!config.database) {
    errors.push('Database name is required');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

/**
 * Validate chunking configuration
 * @param {Object} config - Chunking configuration
 * @returns {Object} Validation result
 */
export const validateChunkingConfig = (config) => {
  const errors = [];
  
  if (!config.method) {
    errors.push('Chunking method is required');
  }
  
  if (config.chunkSize && (isNaN(config.chunkSize) || config.chunkSize < 1)) {
    errors.push('Chunk size must be a positive number');
  }
  
  if (config.overlap && (isNaN(config.overlap) || config.overlap < 0)) {
    errors.push('Overlap must be a non-negative number');
  }
  
  if (config.nClusters && (isNaN(config.nClusters) || config.nClusters < 2)) {
    errors.push('Number of clusters must be at least 2');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};

/**
 * Validate search query
 * @param {string} query - Search query
 * @returns {Object} Validation result
 */
export const validateSearchQuery = (query) => {
  const errors = [];
  
  if (!query || query.trim().length === 0) {
    errors.push('Search query is required');
  }
  
  if (query && query.trim().length < 2) {
    errors.push('Search query must be at least 2 characters long');
  }
  
  return {
    isValid: errors.length === 0,
    errors
  };
};


