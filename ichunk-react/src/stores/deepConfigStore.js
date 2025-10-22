// src/stores/deepConfigStore.js
import { create } from 'zustand';

const useDeepConfigStore = create((set) => ({
  // Current step (0-9 for 10 steps)
  currentStep: 0,
  
  // Step 1: Preprocessing
  preprocessingComplete: false,
  
  // Step 2: Type conversions
  typeConversions: {},
  
  // Step 3: Null strategies
  nullStrategies: {},
  
  // Step 4: Duplicate handling
  duplicateStrategy: 'keep_first',
  
  // Step 5: Stopwords
  removeStopwords: false,
  
  // Step 6: Text normalization
  textProcessing: 'none',
  
  // Step 7: Chunking + Metadata
  chunkConfig: {
    method: 'fixed',
    chunkSize: 400,
    overlap: 50,
    keyColumn: null,
    tokenLimit: 2000,
    nClusters: 10,
    preserveHeaders: true,
    storeMetadata: false,
    selectedNumericColumns: [],
    selectedCategoricalColumns: []
  },
  
  // Step 8: Embedding
  embeddingConfig: {
    modelName: 'paraphrase-MiniLM-L6-v2',
    batchSize: 64,
    useParallel: true
  },
  
  // Step 9: Storage
  storageConfig: {
    type: 'chroma',
    collectionName: 'deep_config_collection',
    retrievalMetric: 'cosine'
  },
  
  // Data from API
  preprocessedData: null,
  chunks: null,
  embeddings: null,
  metadataColumns: {
    numeric: [],
    categorical: []
  },
  
  // New state for enhanced UI
  previewData: null,
  nullAnalysis: null,
  duplicateAnalysis: null,
  conversionsSummary: {},
  nullStrategiesSummary: {},
  duplicatesSummary: {},
  
  // Actions
  nextStep: () => set((state) => ({ 
    currentStep: Math.min(state.currentStep + 1, 9) 
  })),
  
  prevStep: () => set((state) => ({ 
    currentStep: Math.max(state.currentStep - 1, 0) 
  })),
  
  goToStep: (step) => set({ currentStep: step }),
  
  setPreprocessingComplete: (complete) => 
    set({ preprocessingComplete: complete }),
  
  updateTypeConversions: (conversions) => 
    set({ typeConversions: conversions }),
  
  updateNullStrategies: (strategies) => 
    set({ nullStrategies: strategies }),
  
  setDuplicateStrategy: (strategy) => 
    set({ duplicateStrategy: strategy }),
  
  setRemoveStopwords: (remove) => 
    set({ removeStopwords: remove }),
  
  setTextProcessing: (processing) => 
    set({ textProcessing: processing }),
  
  updateChunkConfig: (config) => 
    set((state) => ({
      chunkConfig: { ...state.chunkConfig, ...config }
    })),
  
  updateEmbeddingConfig: (config) => 
    set((state) => ({
      embeddingConfig: { ...state.embeddingConfig, ...config }
    })),
  
  updateStorageConfig: (config) => 
    set((state) => ({
      storageConfig: { ...state.storageConfig, ...config }
    })),
  
  setPreprocessedData: (data) => 
    set({ preprocessedData: data }),
  
  setChunks: (chunks) => 
    set({ chunks: chunks }),
  
  setEmbeddings: (embeddings) => 
    set({ embeddings: embeddings }),
  
  setMetadataColumns: (columns) => 
    set({ metadataColumns: columns }),
  
  // New actions for enhanced UI
  setPreviewData: (data) => 
    set({ previewData: data }),
  
  setNullAnalysis: (analysis) => 
    set({ nullAnalysis: analysis }),
  
  setDuplicateAnalysis: (analysis) => 
    set({ duplicateAnalysis: analysis }),
  
  setConversionsSummary: (summary) => 
    set({ conversionsSummary: summary }),
  
  setNullStrategiesSummary: (summary) => 
    set({ nullStrategiesSummary: summary }),
  
  setDuplicatesSummary: (summary) => 
    set({ duplicatesSummary: summary }),
  
  reset: () => set({
    currentStep: 0,
    preprocessingComplete: false,
    typeConversions: {},
    nullStrategies: {},
    duplicateStrategy: 'keep_first',
    removeStopwords: false,
    textProcessing: 'none',
    chunkConfig: {
      method: 'fixed',
      chunkSize: 400,
      overlap: 50,
      keyColumn: null,
      tokenLimit: 2000,
      nClusters: 10,
      preserveHeaders: true,
      storeMetadata: false,
      selectedNumericColumns: [],
      selectedCategoricalColumns: []
    },
    embeddingConfig: {
      modelName: 'paraphrase-MiniLM-L6-v2',
      batchSize: 64,
      useParallel: true
    },
    storageConfig: {
      type: 'chroma',
      collectionName: 'deep_config_collection',
      retrievalMetric: 'cosine'
    },
    preprocessedData: null,
    chunks: null,
    embeddings: null,
    previewData: null,
    nullAnalysis: null,
    duplicateAnalysis: null,
    conversionsSummary: {},
    nullStrategiesSummary: {},
    duplicatesSummary: {}
  })
}));

export default useDeepConfigStore;

