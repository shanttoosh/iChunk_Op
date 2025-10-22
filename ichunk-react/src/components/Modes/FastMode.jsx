// src/components/Modes/FastMode.jsx
import React, { useState } from 'react';
import { Upload, Database, Play, AlertCircle, Zap } from 'lucide-react';
import useAppStore from '../../stores/appStore';
import useUIStore from '../../stores/uiStore';
import Card from '../UI/Card';
import Button from '../UI/Button';
import FileUpload from '../UI/FileUpload';
import DatabaseModal from '../Database/DatabaseModal';
import SearchInterface from '../Search/SearchInterface';
import ExportSection from '../Export/ExportSection';
import fastModeService from '../../services/fastMode.service';
import { PROCESSING_MODES } from '../../utils/constants';

const FastMode = () => {
  const [file, setFile] = useState(null);
  const [dbConfig, setDbConfig] = useState(null);
  const [inputMethod, setInputMethod] = useState('file'); // 'file' or 'database'
  const [isProcessing, setIsProcessing] = useState(false);
  const [dataUploaded, setDataUploaded] = useState(false);
  const [processingComplete, setProcessingComplete] = useState(false);

  const {
    updateProcessStatus,
    setFileInfo,
    setApiResults
  } = useAppStore();

  const { showDatabaseModal, toggleDatabaseModal } = useUIStore();

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setDbConfig(null);
    setInputMethod('file');
    setDataUploaded(true);
    setProcessingComplete(false);
  };

  const handleDatabaseImport = (config) => {
    setDbConfig(config);
    setFile(null);
    setInputMethod('database');
    setDataUploaded(true);
    setProcessingComplete(false);
  };

  const handleProcess = async () => {
    if (!file && !dbConfig) {
      alert('Please select a file or configure database import');
      return;
    }

    setIsProcessing(true);
    
    try {
      // Reset process status
      Object.keys(PROCESSING_MODES).forEach(step => {
        updateProcessStatus(step, 'running');
      });

      // Store file info
      if (file) {
        setFileInfo({
          name: file.name,
          size: file.size,
          type: file.type
        });
      }

      // Call API
      const result = await fastModeService.runFastMode(file, {
        chunkMethod: 'recursive',
        chunkSize: 400,
        overlap: 50,
        modelChoice: 'paraphrase-MiniLM-L6-v2',
        storageChoice: 'faiss',
        useTurbo: false,
        batchSize: 256,
        applyDefaultPreprocessing: true,
        processLargeFiles: true,
        ...dbConfig
      });

      // Update process status
      updateProcessStatus('preprocessing', 'completed');
      updateProcessStatus('chunking', 'completed');
      updateProcessStatus('embedding', 'completed');
      updateProcessStatus('storage', 'completed');

      // Store results
      setApiResults(result.summary);
      setProcessingComplete(true);

    } catch (error) {
      console.error('Processing error:', error);
      updateProcessStatus('preprocessing', 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="space-y-8">
      <Card>
        <div className="flex items-center space-x-3 mb-4">
          <div className="w-8 h-8 bg-highlight rounded flex items-center justify-center">
            <Zap className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-textPrimary">Fast Mode</h2>
            <p className="text-textSecondary text-sm">Automatic processing with optimized settings</p>
          </div>
        </div>

        {/* Input Method Selection */}
        <div className="mb-6">
          <h3 className="text-lg font-medium text-textPrimary mb-4">Data Source</h3>
          <div className="flex space-x-4">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="inputMethod"
                value="file"
                checked={inputMethod === 'file'}
                onChange={(e) => setInputMethod(e.target.value)}
                className="text-highlight focus:ring-highlight"
              />
              <Upload className="h-5 w-5 text-textSecondary" />
              <span className="text-textPrimary">File Upload</span>
            </label>
            
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="radio"
                name="inputMethod"
                value="database"
                checked={inputMethod === 'database'}
                onChange={(e) => setInputMethod(e.target.value)}
                className="text-highlight focus:ring-highlight"
              />
              <Database className="h-5 w-5 text-textSecondary" />
              <span className="text-textPrimary">Database Import</span>
            </label>
          </div>
        </div>

        {/* File Upload */}
        {inputMethod === 'file' && (
          <div className="mb-6">
            <h3 className="text-lg font-medium text-textPrimary mb-4">Upload CSV File</h3>
            <FileUpload
              onFileSelect={handleFileSelect}
              file={file}
              disabled={isProcessing}
            />
          </div>
        )}

        {/* Database Import */}
        {inputMethod === 'database' && (
          <div className="mb-6">
            <h3 className="text-lg font-medium text-textPrimary mb-4">Database Import</h3>
            <Button
              variant="outline"
              onClick={toggleDatabaseModal}
              disabled={isProcessing}
            >
              <Database className="h-4 w-4 mr-2" />
              Configure Database Connection
            </Button>
            
            {dbConfig && (
              <div className="mt-4 p-4 bg-secondary rounded-card">
                <h4 className="font-medium text-textPrimary mb-2">Selected Configuration:</h4>
                <div className="text-sm text-textSecondary space-y-1">
                  <p><strong>Type:</strong> {dbConfig.dbType}</p>
                  <p><strong>Host:</strong> {dbConfig.host}:{dbConfig.port}</p>
                  <p><strong>Database:</strong> {dbConfig.database}</p>
                  <p><strong>Table:</strong> {dbConfig.tableName}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Auto-Optimized Pipeline Info - Only show after data upload */}
        {dataUploaded && (
          <div className="mb-6 p-4 bg-secondary rounded-card">
            <h3 className="text-lg font-medium text-textPrimary mb-3">Auto-Optimized Pipeline</h3>
            <div className="space-y-2 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-success rounded-full"></div>
                <span className="text-textPrimary">Automatic preprocessing with HTML removal and normalization</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-success rounded-full"></div>
                <span className="text-textPrimary">Semantic clustering with KMeans (10 clusters)</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-success rounded-full"></div>
                <span className="text-textPrimary">paraphrase-MiniLM-L6-v2 embeddings</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-success rounded-full"></div>
                <span className="text-textPrimary">FAISS storage with cosine similarity</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-highlight rounded-full"></div>
                <span className="text-textPrimary">Turbo mode enabled (parallel processing)</span>
              </div>
            </div>
          </div>
        )}

        {/* Large File Warning - Only show after file upload */}
        {dataUploaded && file && file.size > 100 * 1024 * 1024 && (
          <div className="mb-6 p-4 bg-warning bg-opacity-20 border border-warning rounded-card">
            <div className="flex items-center space-x-2 text-warning">
              <AlertCircle className="h-5 w-5" />
              <span className="font-medium">Large File Detected</span>
            </div>
            <p className="text-sm text-textSecondary mt-2">
              This file will be processed using streaming I/O to handle large datasets efficiently.
            </p>
          </div>
        )}

        {/* Process Button - Only show after data upload */}
        {dataUploaded && (
          <div className="flex justify-start mt-6">
            <Button
              variant="primary"
              size="sm"
              onClick={handleProcess}
              disabled={(!file && !dbConfig) || isProcessing}
              loading={isProcessing}
              className="flex items-center space-x-1"
            >
              <Play className="h-4 w-4" />
              <span>{isProcessing ? 'Processing...' : 'Run'}</span>
            </Button>
          </div>
        )}
      </Card>

      {/* Search Interface - Only show after processing is complete */}
      {processingComplete && (
        <SearchInterface />
      )}

      {/* Export Section - Only show after processing is complete */}
      {processingComplete && (
        <ExportSection />
      )}

      {/* Database Modal */}
      <DatabaseModal
        isOpen={showDatabaseModal}
        onClose={toggleDatabaseModal}
        onImport={handleDatabaseImport}
      />
    </div>
  );
};

export default FastMode;
