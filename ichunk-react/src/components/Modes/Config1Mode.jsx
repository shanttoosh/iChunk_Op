// src/components/Modes/Config1Mode.jsx
import React, { useState, useEffect } from 'react';
import { Upload, Database, Play, Settings, Zap, HardDrive, Info } from 'lucide-react';
import useAppStore from '../../stores/appStore';
import useUIStore from '../../stores/uiStore';
import Card from '../UI/Card';
import Button from '../UI/Button';
import FileUpload from '../UI/FileUpload';
import Input from '../UI/Input';
import Select from '../UI/Select';
import DatabaseModal from '../Database/DatabaseModal';
import SearchInterface from '../Search/SearchInterface';
import ExportSection from '../Export/ExportSection';
import config1Service from '../../services/config1.service';
import { CHUNKING_METHODS, STORAGE_TYPES, EMBEDDING_MODELS, RETRIEVAL_METRICS } from '../../utils/constants';
import { getDocumentGroupingColumns } from '../../utils/columnAnalysis';

const Config1Mode = () => {
  const [file, setFile] = useState(null);
  const [dbConfig, setDbConfig] = useState(null);
  const [inputMethod, setInputMethod] = useState('file');
  const [activeTab, setActiveTab] = useState('preprocessing');
  const [isProcessing, setIsProcessing] = useState(false);
  const [documentGroupingColumns, setDocumentGroupingColumns] = useState([]);
  const [dataUploaded, setDataUploaded] = useState(false);
  const [processingComplete, setProcessingComplete] = useState(false);

  // Configuration state
  const [config, setConfig] = useState({
    // Chunking
    chunkMethod: 'recursive',
    chunkSize: 400,
    overlap: 50,
    nClusters: 10,
    documentKeyColumn: '',
    tokenLimit: 2000,
    preserveHeaders: false,
    
    // Embedding
    modelChoice: 'paraphrase-MiniLM-L6-v2',
    useTurbo: false,
    batchSize: 256,
    
    // Storage
    storageChoice: 'faiss',
    retrievalMetric: 'cosine',
    
    // Preprocessing
    applyDefaultPreprocessing: true,
    processLargeFiles: true
  });

  const {
    updateProcessStatus,
    setFileInfo,
    setApiResults
  } = useAppStore();

  const { showDatabaseModal, toggleDatabaseModal } = useUIStore();

  // Analyze columns for document-based chunking when data is available
  useEffect(() => {
    // This would be triggered when data is loaded from preprocessing
    // For now, we'll simulate with some example columns
    // In a real implementation, this would come from the preprocessing step
    const exampleColumns = [
      { value: 'category', label: 'category (5 groups)' },
      { value: 'department', label: 'department (8 groups)' },
      { value: 'status', label: 'status (3 groups)' },
      { value: 'priority', label: 'priority (4 groups)' }
    ];
    setDocumentGroupingColumns(exampleColumns);
  }, [file, dbConfig]);

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setDbConfig(null);
    setInputMethod('file');
    setDataUploaded(true);
    setProcessingComplete(false);
  };

  const handleDatabaseImport = (dbConfig) => {
    setDbConfig(dbConfig);
    setFile(null);
    setInputMethod('database');
    setDataUploaded(true);
    setProcessingComplete(false);
  };

  const handleConfigChange = (field, value) => {
    setConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleProcess = async () => {
    if (!file && !dbConfig) {
      alert('Please select a file or configure database import');
      return;
    }

    setIsProcessing(true);
    
    try {
      // Reset process status
      updateProcessStatus('preprocessing', 'running');
      updateProcessStatus('chunking', 'running');
      updateProcessStatus('embedding', 'running');
      updateProcessStatus('storage', 'running');

      // Store file info
      if (file) {
        setFileInfo({
          name: file.name,
          size: file.size,
          type: file.type
        });
      }

      // Call API
      const result = await config1Service.runConfig1Mode(file, {
        ...config,
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

  const tabs = [
    { id: 'preprocessing', name: 'Preprocessing', icon: Settings },
    { id: 'chunking', name: 'Chunking', icon: HardDrive },
    { id: 'embedding', name: 'Embedding', icon: Zap },
    { id: 'storage', name: 'Storage', icon: Database }
  ];

  const chunkingMethodOptions = [
    { value: 'fixed', label: 'Fixed Size' },
    { value: 'recursive', label: 'Recursive Character' },
    { value: 'semantic', label: 'Semantic Clustering' },
    { value: 'document', label: 'Document Based' }
  ];

  const embeddingModelOptions = [
    { value: 'paraphrase-MiniLM-L6-v2', label: 'paraphrase-MiniLM-L6-v2' },
    { value: 'all-MiniLM-L6-v2', label: 'all-MiniLM-L6-v2' },
    { value: 'paraphrase-mpnet-base-v2', label: 'paraphrase-mpnet-base-v2' }
  ];

  const storageOptions = [
    { value: 'faiss', label: 'FAISS' },
    { value: 'chroma', label: 'ChromaDB' }
  ];

  const retrievalMetricOptions = [
    { value: 'cosine', label: 'Cosine Similarity' },
    { value: 'euclidean', label: 'Euclidean Distance' },
    { value: 'dot_product', label: 'Dot Product' }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'preprocessing':
        return (
          <div className="space-y-6">
            <div className="p-4 bg-secondary rounded-lg">
              <h3 className="text-lg font-medium text-textPrimary mb-3">Preprocessing Settings</h3>
              <div className="space-y-3">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.applyDefaultPreprocessing}
                    onChange={(e) => handleConfigChange('applyDefaultPreprocessing', e.target.checked)}
                    className="text-highlight focus:ring-highlight"
                  />
                  <span className="text-textPrimary">Apply Default Preprocessing</span>
                </label>
                <p className="text-sm text-textSecondary">
                  Automatically applies HTML removal, text normalization, and basic cleaning
                </p>
              </div>
            </div>
          </div>
        );

      case 'chunking':
        return (
          <div className="space-y-6">
            <Select
              label="Chunking Method"
              value={config.chunkMethod}
              onChange={(e) => handleConfigChange('chunkMethod', e.target.value)}
              options={chunkingMethodOptions}
            />

            {/* Method-specific parameters */}
            {(config.chunkMethod === 'fixed' || config.chunkMethod === 'recursive') && (
              <div className="p-4 bg-secondary rounded-lg">
                <div className="flex items-center space-x-2 mb-3">
                  <Info className="h-4 w-4 text-blue-400" />
                  <span className="text-sm text-textSecondary">
                    {config.chunkMethod === 'fixed' 
                      ? 'Splits data into fixed-size chunks of characters with overlap'
                      : 'Splits key-value formatted lines with recursive separators and overlap'
                    }
                  </span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Input
                    label="Chunk Size (characters)"
                    type="number"
                    min="50"
                    max="20000"
                    step="50"
                    value={config.chunkSize}
                    onChange={(e) => handleConfigChange('chunkSize', parseInt(e.target.value))}
                    placeholder="400"
                  />
                  <Input
                    label="Overlap (characters)"
                    type="number"
                    min="0"
                    max={parseInt(config.chunkSize) - 1}
                    value={config.overlap}
                    onChange={(e) => handleConfigChange('overlap', parseInt(e.target.value))}
                    placeholder="50"
                  />
                </div>
              </div>
            )}

            {config.chunkMethod === 'semantic' && (
              <div className="p-4 bg-secondary rounded-lg">
                <div className="flex items-center space-x-2 mb-3">
                  <Info className="h-4 w-4 text-blue-400" />
                  <span className="text-sm text-textSecondary">Clusters rows semantically and concatenates each cluster as a chunk</span>
                </div>
                <Input
                  label="Number of clusters"
                  type="number"
                  min="2"
                  max="50"
                  value={config.nClusters}
                  onChange={(e) => handleConfigChange('nClusters', parseInt(e.target.value))}
                  placeholder="10"
                />
              </div>
            )}

            {config.chunkMethod === 'document' && (
              <div className="p-4 bg-secondary rounded-lg">
                <div className="flex items-center space-x-2 mb-3">
                  <Info className="h-4 w-4 text-blue-400" />
                  <span className="text-sm text-textSecondary">Group by a key column and split by token limit (headers optional)</span>
                </div>
                <div className="space-y-3">
                  <Select
                    label="Document Key Column"
                    value={config.documentKeyColumn}
                    onChange={(e) => handleConfigChange('documentKeyColumn', e.target.value)}
                    options={documentGroupingColumns.length > 0 ? documentGroupingColumns : [
                      { value: '', label: 'No suitable columns found - upload data first' }
                    ]}
                    placeholder="Select a column for grouping"
                    disabled={documentGroupingColumns.length === 0}
                  />
                  {documentGroupingColumns.length === 0 && (
                    <div className="text-xs text-warning">
                      ⚠️ Upload data first to see available grouping columns
                    </div>
                  )}
                  <Input
                    label="Token limit per chunk"
                    type="number"
                    min="200"
                    max="10000"
                    step="100"
                    value={config.tokenLimit}
                    onChange={(e) => handleConfigChange('tokenLimit', parseInt(e.target.value))}
                    placeholder="2000"
                  />
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={config.preserveHeaders}
                      onChange={(e) => handleConfigChange('preserveHeaders', e.target.checked)}
                      className="text-highlight focus:ring-highlight"
                    />
                    <span className="text-textPrimary">Include headers in each chunk</span>
                  </label>
                </div>
              </div>
            )}
          </div>
        );

      case 'embedding':
        return (
          <div className="space-y-6">
            <Select
              label="Embedding Model"
              value={config.modelChoice}
              onChange={(e) => handleConfigChange('modelChoice', e.target.value)}
              options={embeddingModelOptions}
            />

            <div className="space-y-4">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.useTurbo}
                  onChange={(e) => handleConfigChange('useTurbo', e.target.checked)}
                  className="text-highlight focus:ring-highlight"
                />
                <span className="text-textPrimary">Use Turbo Mode</span>
              </label>
              <p className="text-sm text-textSecondary">
                Enable parallel processing for faster embedding generation
              </p>
            </div>

            <Input
              label="Batch Size"
              type="number"
              value={config.batchSize}
              onChange={(e) => handleConfigChange('batchSize', parseInt(e.target.value))}
              placeholder="256"
            />
          </div>
        );

      case 'storage':
        return (
          <div className="space-y-6">
            <Select
              label="Storage Type"
              value={config.storageChoice}
              onChange={(e) => handleConfigChange('storageChoice', e.target.value)}
              options={storageOptions}
            />

            <Select
              label="Retrieval Metric"
              value={config.retrievalMetric}
              onChange={(e) => handleConfigChange('retrievalMetric', e.target.value)}
              options={retrievalMetricOptions}
            />

            <div className="p-4 bg-secondary rounded-lg">
              <h4 className="font-medium text-textPrimary mb-2">Storage Information</h4>
              <div className="text-sm text-textSecondary space-y-1">
                <p><strong>FAISS:</strong> Fast similarity search with CPU/GPU support</p>
                <p><strong>ChromaDB:</strong> Persistent vector database with metadata support</p>
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="space-y-8">
      <Card>
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-8 h-8 bg-highlight rounded flex items-center justify-center">
            <Settings className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-textPrimary">Config-1 Mode</h2>
            <p className="text-textSecondary text-sm">Configurable processing with custom options</p>
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
              <div className="mt-4 p-4 bg-secondary rounded-lg">
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

        {/* Tabbed Configuration - Only show after data upload */}
        {dataUploaded && (
          <div className="mb-6">
            <h3 className="text-lg font-medium text-textPrimary mb-4">Configuration</h3>
            
            {/* Tab Navigation */}
            <div className="flex space-x-1 mb-6 border-b border-border">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`
                      flex items-center space-x-2 px-4 py-2 text-sm font-medium transition-all duration-200
                      ${activeTab === tab.id 
                        ? 'tab-active text-highlight border-b-2 border-highlight' 
                        : 'text-textSecondary hover:text-textPrimary'
                      }
                    `}
                  >
                    <Icon className="h-4 w-4" />
                    <span>{tab.name}</span>
                  </button>
                );
              })}
            </div>

            {/* Tab Content */}
            <div className="min-h-[300px]">
              {renderTabContent()}
            </div>
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

export default Config1Mode;