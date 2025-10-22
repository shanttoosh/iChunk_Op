// src/stores/uiStore.js
import { create } from 'zustand';

const useUIStore = create((set) => ({
  // Modals
  showDatabaseModal: false,
  showMetadataModal: false,
  
  // Sidebar
  sidebarCollapsed: false,
  
  // Loading states
  isUploading: false,
  uploadProgress: 0,
  isProcessing: false,
  
  // Notifications
  notifications: [],
  
  // Actions
  toggleDatabaseModal: () => 
    set((state) => ({ showDatabaseModal: !state.showDatabaseModal })),
  
  toggleMetadataModal: () => 
    set((state) => ({ showMetadataModal: !state.showMetadataModal })),
  
  toggleSidebar: () => 
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  
  setUploading: (uploading) => 
    set({ isUploading: uploading }),
  
  setUploadProgress: (progress) => 
    set({ uploadProgress: progress }),
  
  setProcessing: (processing) => 
    set({ isProcessing: processing }),
  
  addNotification: (notification) => 
    set((state) => ({
      notifications: [
        ...state.notifications,
        { ...notification, id: Date.now().toString() }
      ]
    })),
  
  removeNotification: (id) => 
    set((state) => ({
      notifications: state.notifications.filter(n => n.id !== id)
    }))
}));

export default useUIStore;
