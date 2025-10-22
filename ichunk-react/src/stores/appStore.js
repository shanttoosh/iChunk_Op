// src/stores/appStore.js
import { create } from 'zustand';

const useAppStore = create((set) => ({
  // Current mode
  currentMode: null, // 'fast' | 'config1' | 'deep' | 'campaign' | null
  
  // Process tracking
  processStatus: {
    preprocessing: 'pending',
    chunking: 'pending',
    embedding: 'pending',
    storage: 'pending',
    retrieval: 'pending'
  },
  
  // File info
  fileInfo: {
    name: null,
    size: null,
    uploadTime: null,
    location: null,
    largeFileProcessed: false,
    turboMode: false
  },
  
  // API results
  apiResults: null,
  
  // Retrieval results
  retrievalResults: null,
  
  // LLM mode
  llmMode: 'Normal Retrieval', // 'Normal Retrieval' | 'LLM Enhanced'
  
  // System info
  systemInfo: {
    memoryUsage: null,
    availableMemory: null,
    totalMemory: null,
    batchSize: 256
  },
  
  // Actions
  setCurrentMode: (mode) => set({ currentMode: mode }),
  
  updateProcessStatus: (step, status) => 
    set((state) => ({
      processStatus: {
        ...state.processStatus,
        [step]: status
      }
    })),
  
  setFileInfo: (info) => 
    set((state) => ({
      fileInfo: { ...state.fileInfo, ...info }
    })),
  
  setApiResults: (results) => set({ apiResults: results }),
  
  setRetrievalResults: (results) => set({ retrievalResults: results }),
  
  toggleLLMMode: () => 
    set((state) => ({
      llmMode: state.llmMode === 'Normal Retrieval' 
        ? 'LLM Enhanced' 
        : 'Normal Retrieval'
    })),
  
  setSystemInfo: (info) => 
    set((state) => ({
      systemInfo: { ...state.systemInfo, ...info }
    })),
  
  resetSession: () => set({
    currentMode: null,
    processStatus: {
      preprocessing: 'pending',
      chunking: 'pending',
      embedding: 'pending',
      storage: 'pending',
      retrieval: 'pending'
    },
    fileInfo: {
      name: null,
      size: null,
      uploadTime: null,
      location: null,
      largeFileProcessed: false,
      turboMode: false
    },
    apiResults: null,
    retrievalResults: null
  })
}));

export default useAppStore;


