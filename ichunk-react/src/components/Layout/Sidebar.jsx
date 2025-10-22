// src/components/Layout/Sidebar.jsx
import React, { useEffect } from 'react';
import { 
  Zap, 
  Settings, 
  Microscope, 
  Target, 
  Database, 
  RotateCcw,
  MemoryStick,
  HardDrive,
  Menu,
  X
} from 'lucide-react';
import useAppStore from '../../stores/appStore';
import useUIStore from '../../stores/uiStore';
import ProcessStep from '../UI/ProcessStep';
import Button from '../UI/Button';
import { formatFileSize } from '../../utils/formatting';
import { PROCESS_STEPS } from '../../utils/constants';
import { systemService } from '../../services/system.service';

const Sidebar = () => {
  const {
    currentMode,
    processStatus,
    fileInfo,
    llmMode,
    systemInfo,
    apiResults,
    toggleLLMMode,
    resetSession
  } = useAppStore();

  const { sidebarCollapsed, toggleSidebar } = useUIStore();

  const handleResetSession = async () => {
    try {
      console.log('Starting reset session...');
      
      // Reset backend state first
      const response = await systemService.resetSession();
      console.log('Backend state cleared:', response);
      
      // Then reset frontend state
      resetSession();
      console.log('Frontend state cleared');
      
      // Show success message
      console.log('Reset session completed successfully');
    } catch (error) {
      console.error('Error resetting session:', error);
      
      // Still reset frontend state even if backend fails
      resetSession();
      console.log('Frontend state cleared (backend reset failed)');
    }
  };

  const modeIcons = {
    fast: <Zap className="h-5 w-5" />,
    config1: <Settings className="h-5 w-5" />,
    deep: <Microscope className="h-5 w-5" />,
    campaign: <Target className="h-5 w-5" />
  };

  const modeLabels = {
    fast: 'Fast Mode',
    config1: 'Config-1 Mode',
    deep: 'Deep Config Mode',
    campaign: 'Campaign Mode'
  };

  return (
    <div className={`${sidebarCollapsed ? 'w-16' : 'w-80'} bg-primaryDark border-r border-border flex flex-col h-screen fixed left-0 top-0 transition-all duration-300`}>
      {/* Toggle Button - Left Side */}
      <div className="p-4 border-b border-border flex-shrink-0">
        <button
          onClick={toggleSidebar}
          className="p-2 rounded-lg bg-highlight hover:bg-highlight/80 transition-colors border border-highlight/20"
          title={sidebarCollapsed ? "Show Sidebar" : "Hide Sidebar"}
        >
          {sidebarCollapsed ? (
            <Menu className="h-5 w-5 text-white" />
          ) : (
            <X className="h-5 w-5 text-white" />
          )}
        </button>
      </div>

      {/* Scrollable Content Area */}
      <div className="flex-1 overflow-y-auto custom-scrollbar max-h-full">
        {/* Header */}
        {!sidebarCollapsed && (
          <div className="p-6 border-b border-border">
            {/* Elevate Logo */}
            <div className="flex items-center justify-center mb-4">
              <img 
                src="/elevate-logo.svg" 
                alt="Elevate" 
                className="h-6 w-auto opacity-90 hover:opacity-100 transition-opacity duration-200"
                onError={(e) => {
                  // Fallback to text if logo not found
                  e.target.style.display = 'none';
                  e.target.nextSibling.style.display = 'block';
                }}
              />
              {/* Fallback text logo */}
              <div className="hidden h-6 flex items-center">
                <span className="text-highlight font-bold text-sm">ELEVATE</span>
              </div>
            </div>
            
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-8 h-8 bg-highlight rounded flex items-center justify-center">
                <span className="text-white font-bold text-sm">iC</span>
              </div>
              <div>
                <h1 className="text-lg font-semibold text-textPrimary">iChunk Optimizer</h1>
                <p className="text-sm text-textSecondary">Data Processing & Vector Search</p>
              </div>
            </div>
            
            {currentMode && (
              <div className="flex items-center space-x-2 text-sm">
                {modeIcons[currentMode]}
                <span className="text-textPrimary">{modeLabels[currentMode]}</span>
              </div>
            )}
          </div>
        )}

        {/* Process Tracker */}
        {!sidebarCollapsed && (
          <div className="p-6 border-b border-border">
            <h3 className="text-sm font-semibold text-textPrimary mb-4">Process Status</h3>
            <div className="space-y-2">
              {Object.entries(PROCESS_STEPS).map(([key, stepName]) => (
                <ProcessStep
                  key={key}
                  name={stepName}
                  status={processStatus[key]}
                />
              ))}
            </div>
          </div>
        )}

        {/* LLM Mode Toggle */}
        {!sidebarCollapsed && (
          <div className="p-6 border-b border-border">
            <h3 className="text-sm font-semibold text-textPrimary mb-4">LLM Mode</h3>
            <div className="space-y-2">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name="llmMode"
                  checked={llmMode === 'Normal Retrieval'}
                  onChange={() => toggleLLMMode()}
                  className="text-highlight focus:ring-highlight"
                />
                <span className="text-sm text-textPrimary">Normal Retrieval</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name="llmMode"
                  checked={llmMode === 'LLM Enhanced'}
                  onChange={() => toggleLLMMode()}
                  className="text-highlight focus:ring-highlight"
                />
                <span className="text-sm text-textPrimary">LLM Enhanced</span>
              </label>
            </div>
          </div>
        )}

        {/* File Info */}
        {fileInfo.name && !sidebarCollapsed && (
          <div className="p-6 border-b border-border">
            <h3 className="text-sm font-semibold text-textPrimary mb-4">File Info</h3>
            <div className="space-y-2 text-sm">
              <div>
                <span className="text-textSecondary">Name:</span>
                <p className="text-textPrimary font-medium truncate">{fileInfo.name}</p>
              </div>
              <div>
                <span className="text-textSecondary">Size:</span>
                <p className="text-textPrimary">{formatFileSize(fileInfo.size)}</p>
              </div>
              {fileInfo.largeFileProcessed && (
                <div className="flex items-center space-x-1 text-warning">
                  <HardDrive className="h-4 w-4" />
                  <span className="text-xs">Large File Mode</span>
                </div>
              )}
              {fileInfo.turboMode && (
                <div className="flex items-center space-x-1 text-highlight">
                  <Zap className="h-4 w-4" />
                  <span className="text-xs">Turbo Mode</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* System Info */}
        {!sidebarCollapsed && (
          <div className="p-6 border-b border-border">
            <h3 className="text-sm font-semibold text-textPrimary mb-4">System Info</h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center space-x-2">
                <MemoryStick className="h-4 w-4 text-textSecondary" />
                <span className="text-textSecondary">Memory:</span>
                <span className="text-textPrimary">{systemInfo.memoryUsage || 'N/A'}</span>
              </div>
              <div className="flex items-center space-x-2">
                <Database className="h-4 w-4 text-textSecondary" />
                <span className="text-textSecondary">Batch Size:</span>
                <span className="text-textPrimary">{systemInfo.batchSize || 256}</span>
              </div>
            </div>
          </div>
        )}

        {/* Results Summary */}
        {apiResults && !sidebarCollapsed && (
          <div className="p-6 border-b border-border">
            <h3 className="text-sm font-semibold text-textPrimary mb-4">Last Results</h3>
            <div className="space-y-2 text-sm">
              <div>
                <span className="text-textSecondary">Rows:</span>
                <span className="text-textPrimary ml-2">{apiResults.rows || 'N/A'}</span>
              </div>
              <div>
                <span className="text-textSecondary">Chunks:</span>
                <span className="text-textPrimary ml-2">{apiResults.chunks || 'N/A'}</span>
              </div>
              <div>
                <span className="text-textSecondary">Storage:</span>
                <span className="text-textPrimary ml-2">{apiResults.stored || 'N/A'}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Actions - Fixed at Bottom */}
      {!sidebarCollapsed && (
        <div className="p-6 border-t border-border flex-shrink-0">
          <div className="space-y-3">
            <Button
              variant="outline"
              size="sm"
              className="w-full flex items-center justify-center space-x-2"
              onClick={handleResetSession}
            >
              <RotateCcw className="h-4 w-4" />
              <span className="text-sm font-medium">Reset Session</span>
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Sidebar;
