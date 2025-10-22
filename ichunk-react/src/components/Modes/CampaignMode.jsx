// src/components/Modes/CampaignMode.jsx
import React, { useState } from 'react';
import { Upload, Database, Play, Target, Zap, Users, Building, Info } from 'lucide-react';
import useAppStore from '../../stores/appStore';
import useCampaignStore from '../../stores/campaignStore';
import useUIStore from '../../stores/uiStore';
import Card from '../UI/Card';
import Button from '../UI/Button';
import FileUpload from '../UI/FileUpload';
import Input from '../UI/Input';
import Select from '../UI/Select';
import DatabaseModal from '../Database/DatabaseModal';
import SearchInterface from '../Search/SearchInterface';
import ExportSection from '../Export/ExportSection';
import campaignService from '../../services/campaign.service';
import { getDocumentGroupingColumns } from '../../utils/columnAnalysis';

const CampaignMode = () => {
  const [file, setFile] = useState(null);
  const [dbConfig, setDbConfig] = useState(null);
  const [inputMethod, setInputMethod] = useState('file');
  const [isProcessing, setIsProcessing] = useState(false);
  const [documentGroupingColumns, setDocumentGroupingColumns] = useState([]);
  const [dataUploaded, setDataUploaded] = useState(false);
  const [processingComplete, setProcessingComplete] = useState(false);

  // Configuration state
  const [config, setConfig] = useState({
    chunkMethod: 'record_based',
    chunkSize: '5',
    modelChoice: 'paraphrase-MiniLM-L6-v2',
    storageChoice: 'faiss',
    useOpenai: 'false',
    openaiApiKey: '',
    openaiBaseUrl: '',
    processLargeFiles: 'true',
    useTurbo: 'true',
    batchSize: '256',
    preserveRecordStructure: 'true',
    documentKeyColumn: ''
  });

  const {
    updateProcessStatus,
    setFileInfo,
    setApiResults
  } = useAppStore();

  const {
    setCampaignResults,
    setFieldMapping
  } = useCampaignStore();

  const { showDatabaseModal, toggleDatabaseModal } = useUIStore();

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
      const result = await campaignService.runCampaign(file, {
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
      setCampaignResults(result);
      setProcessingComplete(true);
      
      // Set field mapping if available
      if (result.field_mapping) {
        setFieldMapping(result.field_mapping);
      }

    } catch (error) {
      console.error('Processing error:', error);
      updateProcessStatus('preprocessing', 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  const chunkMethodOptions = [
    { value: 'record_based', label: 'Record Based' },
    { value: 'company_based', label: 'Company Based' },
    { value: 'source_based', label: 'Source Based' },
    { value: 'semantic_clustering', label: 'Semantic Clustering' },
    { value: 'document', label: 'Document Based' }
  ];

  const embeddingModelOptions = [
    { value: 'paraphrase-MiniLM-L6-v2', label: 'paraphrase-MiniLM-L6-v2' },
    { value: 'all-MiniLM-L6-v2', label: 'all-MiniLM-L6-v2' },
    { value: 'paraphrase-mpnet-base-v2', label: 'paraphrase-mpnet-base-v2' },
    { value: 'text-embedding-ada-002', label: 'OpenAI Ada-002' }
  ];

  const storageOptions = [
    { value: 'faiss', label: 'FAISS' },
    { value: 'chroma', label: 'ChromaDB' }
  ];

  return (
    <div className="space-y-8">
      <Card>
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-8 h-8 bg-highlight rounded flex items-center justify-center">
            <Target className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-textPrimary">Campaign Mode</h2>
            <p className="text-textSecondary text-sm">Specialized for media campaigns and contact data</p>
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

        {/* Campaign Configuration - Only show after data upload */}
        {dataUploaded && (
          <div className="mb-6">
            <h3 className="text-lg font-medium text-textPrimary mb-4">Campaign Configuration</h3>
          
          <div className="space-y-4 mb-4">
            <Select
              label="Chunking Method"
              value={config.chunkMethod}
              onChange={(e) => handleConfigChange('chunkMethod', e.target.value)}
              options={chunkMethodOptions}
            />

            {/* Method-specific parameters */}
            {config.chunkMethod === 'record_based' && (
              <div className="p-4 bg-secondary rounded-lg">
                <div className="flex items-center space-x-2 mb-3">
                  <Info className="h-4 w-4 text-blue-400" />
                  <span className="text-sm text-textSecondary">Number of contact records per chunk</span>
                </div>
                <Input
                  label="Records per Chunk"
                  type="number"
                  min="1"
                  max="20"
                  value={config.chunkSize}
                  onChange={(e) => handleConfigChange('chunkSize', e.target.value)}
                  placeholder="5"
                />
              </div>
            )}

            {config.chunkMethod === 'company_based' && (
              <div className="p-4 bg-secondary rounded-lg">
                <div className="flex items-center space-x-2 mb-3">
                  <Info className="h-4 w-4 text-blue-400" />
                  <span className="text-sm text-textSecondary">Groups contacts by company name</span>
                </div>
                <Input
                  label="Max records per company"
                  type="number"
                  min="5"
                  max="50"
                  value={config.chunkSize}
                  onChange={(e) => handleConfigChange('chunkSize', e.target.value)}
                  placeholder="10"
                />
              </div>
            )}

            {config.chunkMethod === 'source_based' && (
              <div className="p-4 bg-secondary rounded-lg">
                <div className="flex items-center space-x-2 mb-3">
                  <Info className="h-4 w-4 text-blue-400" />
                  <span className="text-sm text-textSecondary">Groups contacts by lead source</span>
                </div>
                <Input
                  label="Max records per source"
                  type="number"
                  min="5"
                  max="30"
                  value={config.chunkSize}
                  onChange={(e) => handleConfigChange('chunkSize', e.target.value)}
                  placeholder="8"
                />
              </div>
            )}

            {config.chunkMethod === 'semantic_clustering' && (
              <div className="p-4 bg-secondary rounded-lg">
                <div className="flex items-center space-x-2 mb-3">
                  <Info className="h-4 w-4 text-blue-400" />
                  <span className="text-sm text-textSecondary">AI-powered grouping of similar contacts</span>
                </div>
                <Input
                  label="Number of clusters"
                  type="number"
                  min="3"
                  max="20"
                  value={config.chunkSize}
                  onChange={(e) => handleConfigChange('chunkSize', e.target.value)}
                  placeholder="10"
                />
              </div>
            )}

            {config.chunkMethod === 'document_based' && (
              <div className="p-4 bg-secondary rounded-lg">
                <div className="flex items-center space-x-2 mb-3">
                  <Info className="h-4 w-4 text-blue-400" />
                  <span className="text-sm text-textSecondary">Groups contacts by document key column</span>
                </div>
                <Select
                  label="Document Key Column (Optional)"
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
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <Select
              label="Embedding Model"
              value={config.modelChoice}
              onChange={(e) => handleConfigChange('modelChoice', e.target.value)}
              options={embeddingModelOptions}
            />

            <Select
              label="Storage Type"
              value={config.storageChoice}
              onChange={(e) => handleConfigChange('storageChoice', e.target.value)}
              options={storageOptions}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <Input
              label="Batch Size"
              value={config.batchSize}
              onChange={(e) => handleConfigChange('batchSize', e.target.value)}
              placeholder="256"
            />

            <Select
              label="Document Key Column (Optional)"
              value={config.documentKeyColumn}
              onChange={(e) => handleConfigChange('documentKeyColumn', e.target.value)}
              options={documentGroupingColumns.length > 0 ? documentGroupingColumns : [
                { value: '', label: 'No suitable columns found - upload data first' }
              ]}
              placeholder="Select a column for grouping"
              disabled={documentGroupingColumns.length === 0}
            />
          </div>

          {/* OpenAI Configuration */}
          <div className="mb-4 p-4 bg-secondary rounded-lg">
            <h4 className="font-medium text-textPrimary mb-3">OpenAI Configuration (Optional)</h4>
            <div className="space-y-3">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={config.useOpenai === 'true'}
                  onChange={(e) => handleConfigChange('useOpenai', e.target.checked ? 'true' : 'false')}
                  className="text-highlight focus:ring-highlight"
                />
                <span className="text-textPrimary">Use OpenAI Embeddings</span>
              </label>
              
              {config.useOpenai === 'true' && (
                <div className="space-y-3">
                  <Input
                    label="OpenAI API Key"
                    type="password"
                    value={config.openaiApiKey}
                    onChange={(e) => handleConfigChange('openaiApiKey', e.target.value)}
                    placeholder="Enter your OpenAI API key"
                  />
                  <Input
                    label="OpenAI Base URL (Optional)"
                    value={config.openaiBaseUrl}
                    onChange={(e) => handleConfigChange('openaiBaseUrl', e.target.value)}
                    placeholder="https://api.openai.com/v1"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Performance Options */}
          <div className="space-y-3">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={config.useTurbo === 'true'}
                onChange={(e) => handleConfigChange('useTurbo', e.target.checked ? 'true' : 'false')}
                className="text-highlight focus:ring-highlight"
              />
              <span className="text-textPrimary">Use Turbo Mode</span>
            </label>
            
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={config.preserveRecordStructure === 'true'}
                onChange={(e) => handleConfigChange('preserveRecordStructure', e.target.checked ? 'true' : 'false')}
                className="text-highlight focus:ring-highlight"
              />
              <span className="text-textPrimary">Preserve Record Structure</span>
            </label>
          </div>
        </div>
        )}

        {/* Campaign Features Info - Only show after data upload */}
        {dataUploaded && (
          <div className="mb-6 p-4 bg-secondary rounded-lg">
            <h3 className="text-lg font-medium text-textPrimary mb-3">Campaign Mode Features</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <Users className="h-4 w-4 text-highlight" />
                <span className="text-textPrimary">Smart Company Retrieval</span>
              </div>
              <div className="flex items-center space-x-2">
                <Building className="h-4 w-4 text-highlight" />
                <span className="text-textPrimary">Contact Management</span>
              </div>
              <div className="flex items-center space-x-2">
                <Target className="h-4 w-4 text-highlight" />
                <span className="text-textPrimary">Campaign Optimization</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <Zap className="h-4 w-4 text-highlight" />
                <span className="text-textPrimary">Two-Stage Retrieval</span>
              </div>
              <div className="flex items-center space-x-2">
                <Database className="h-4 w-4 text-highlight" />
                <span className="text-textPrimary">Complete Records</span>
              </div>
              <div className="flex items-center space-x-2">
                <Target className="h-4 w-4 text-highlight" />
                <span className="text-textPrimary">Match Type Detection</span>
              </div>
            </div>
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

export default CampaignMode;