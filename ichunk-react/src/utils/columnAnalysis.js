// src/utils/columnAnalysis.js

/**
 * Analyze columns to identify low cardinality columns suitable for grouping
 * @param {Array} data - Array of data objects
 * @param {number} maxCardinality - Maximum number of unique values to consider as low cardinality (default: 20)
 * @param {number} minSamples - Minimum number of samples to analyze (default: 100)
 * @returns {Object} Analysis results
 */
export const analyzeColumnCardinality = (data, maxCardinality = 20, minSamples = 100) => {
  if (!data || !Array.isArray(data) || data.length === 0) {
    return {
      lowCardinalityColumns: [],
      highCardinalityColumns: [],
      columnStats: {}
    };
  }

  const columnStats = {};
  const sampleSize = Math.min(data.length, minSamples);
  const sampleData = data.slice(0, sampleSize);

  // Get all column names from the first object
  const columns = Object.keys(sampleData[0] || {});

  columns.forEach(column => {
    const values = sampleData.map(row => row[column]);
    const uniqueValues = new Set(values.filter(val => val !== null && val !== undefined && val !== ''));
    const cardinality = uniqueValues.size;
    const nullCount = values.filter(val => val === null || val === undefined || val === '').length;
    const nullPercentage = (nullCount / values.length) * 100;

    columnStats[column] = {
      cardinality,
      uniqueValues: Array.from(uniqueValues),
      nullCount,
      nullPercentage,
      totalSamples: values.length,
      isLowCardinality: cardinality <= maxCardinality && cardinality > 1,
      isSuitableForGrouping: cardinality <= maxCardinality && cardinality > 1 && nullPercentage < 50
    };
  });

  const lowCardinalityColumns = columns.filter(col => 
    columnStats[col].isSuitableForGrouping
  );

  const highCardinalityColumns = columns.filter(col => 
    !columnStats[col].isSuitableForGrouping
  );

  return {
    lowCardinalityColumns,
    highCardinalityColumns,
    columnStats,
    totalColumns: columns.length,
    analyzedSamples: sampleSize
  };
};

/**
 * Get column options for document-based chunking dropdown
 * @param {Array} data - Array of data objects
 * @param {number} maxCardinality - Maximum cardinality for grouping columns
 * @returns {Array} Array of column options for dropdown
 */
export const getDocumentGroupingColumns = (data, maxCardinality = 20) => {
  const analysis = analyzeColumnCardinality(data, maxCardinality);
  
  return analysis.lowCardinalityColumns.map(column => ({
    value: column,
    label: `${column} (${analysis.columnStats[column].cardinality} groups)`,
    cardinality: analysis.columnStats[column].cardinality,
    nullPercentage: analysis.columnStats[column].nullPercentage
  }));
};

/**
 * Check if a column is suitable for document-based grouping
 * @param {string} column - Column name
 * @param {Object} columnStats - Column statistics
 * @param {number} maxCardinality - Maximum cardinality threshold
 * @returns {boolean} Whether column is suitable for grouping
 */
export const isSuitableForGrouping = (column, columnStats, maxCardinality = 20) => {
  if (!columnStats[column]) return false;
  
  const stats = columnStats[column];
  return (
    stats.cardinality <= maxCardinality &&
    stats.cardinality > 1 &&
    stats.nullPercentage < 50
  );
};

export default {
  analyzeColumnCardinality,
  getDocumentGroupingColumns,
  isSuitableForGrouping
};


