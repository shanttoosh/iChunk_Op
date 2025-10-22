// src/stores/campaignStore.js
import { create } from 'zustand';

const useCampaignStore = create((set) => ({
  // Campaign results
  campaignResults: null,
  
  // Smart retrieval toggle
  useSmartRetrieval: true,
  
  // Chunk method
  chunkMethod: 'record_based',
  
  // Search field
  searchField: 'all',
  
  // Field mapping (auto-detected)
  fieldMapping: null,
  
  // Actions
  setCampaignResults: (results) => 
    set({ campaignResults: results }),
  
  toggleSmartRetrieval: () => 
    set((state) => ({ useSmartRetrieval: !state.useSmartRetrieval })),
  
  setChunkMethod: (method) => 
    set({ chunkMethod: method }),
  
  setSearchField: (field) => 
    set({ searchField: field }),
  
  setFieldMapping: (mapping) => 
    set({ fieldMapping: mapping }),
  
  reset: () => set({
    campaignResults: null,
    useSmartRetrieval: true,
    chunkMethod: 'record_based',
    searchField: 'all',
    fieldMapping: null
  })
}));

export default useCampaignStore;


