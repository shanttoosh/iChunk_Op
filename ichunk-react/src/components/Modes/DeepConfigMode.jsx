// src/components/Modes/DeepConfigMode.jsx
import React, { useState, useEffect, useCallback } from 'react';
import { 
  ChevronLeft, 
  ChevronRight, 
  Upload, 
  Database, 
  Settings, 
  FileText, 
  Scissors, 
  Brain, 
  HardDrive, 
  Search, 
  CheckCircle,
  AlertCircle,
  Info,
  Play,
  Save,
  Loader,
  Eye,
  RefreshCw,
  ArrowRight,
  ArrowLeft,
  Download
} from 'lucide-react';
import { toast } from 'react-hot-toast';
import useAppStore from '../../stores/appStore';
import useDeepConfigStore from '../../stores/deepConfigStore';
import useUIStore from '../../stores/uiStore';
import Card from '../UI/Card';
import Button from '../UI/Button';
import FileUpload from '../UI/FileUpload';
import Input from '../UI/Input';
import Select from '../UI/Select';
import DatabaseModal from '../Database/DatabaseModal';
import DataPreviewTable from '../UI/DataPreviewTable';
import DataTypeTable from '../UI/DataTypeTable';
import DuplicateAnalysisTable from '../UI/DuplicateAnalysisTable';
import DatasetPreviewTable from '../DeepConfig/DatasetPreviewTable';
import TypeConversionTable from '../DeepConfig/TypeConversionTable';
import NullAnalysisTable from '../DeepConfig/NullAnalysisTable';
import DuplicateAnalysisCard from '../DeepConfig/DuplicateAnalysisCard';
import SearchInterface from '../Search/SearchInterface';
import ExportSection from '../Export/ExportSection';
import deepConfigService from '../../services/deepConfig.service';
import { getDocumentGroupingColumns } from '../../utils/columnAnalysis';
import MetadataColumnSelector from '../DeepConfig/MetadataColumnSelector';

const DeepConfigMode = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [file, setFile] = useState(null);
  const [dbConfig, setDbConfig] = useState(null);
  const [inputMethod, setInputMethod] = useState('file');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [dataPreview, setDataPreview] = useState(null);
  const [stepResults, setStepResults] = useState({});
  const [columnInfo, setColumnInfo] = useState(null);
  const [nullProfile, setNullProfile] = useState([]);
  const [duplicateAnalysis, setDuplicateAnalysis] = useState(null);
  const [metadataColumns, setMetadataColumns] = useState([]);
  const [dataUploaded, setDataUploaded] = useState(false);
  const [processingComplete, setProcessingComplete] = useState(false);

  // Deep configuration state
  const [config, setConfig] = useState({
    // Step 1: Preprocessing
    preprocessing: {
      completed: false,
      fileInfo: null,
      dataTypes: {},
      columnNames: []
    },
    
    // Step 2: Type Conversion
    typeConversion: {
      completed: false,
      groupConversions: {},
      selectedColumns: {},
      conversions: {},
      suggestions: {}
    },
    
    // Step 3: Null Handling
    nullHandling: {
      completed: false,
      nullProfile: [],
      strategies: {},
      customValues: {}
    },
    
    // Step 4: Duplicates
    duplicates: {
      completed: false,
      strategy: 'keep_first',
      removedCount: 0
    },
    
    // Step 5: Stopwords
    stopwords: {
      completed: false,
      enabled: false
    },
    
    // Step 6: Normalization
    normalization: {
      completed: false,
      method: 'none'
    },
    
    // Step 7: Chunking
    chunking: {
      completed: false,
      method: 'fixed',
      chunkSize: 400,
      overlap: 50,
      keyColumn: null,
      tokenLimit: 2000,
      nClusters: 10,
      storeMetadata: true,
      numericColumns: [],
      categoricalColumns: [],
      // Agentic chunking parameters
      agenticStrategy: 'auto',
      userContext: ''
    },
    
    // Step 8: Embedding
    embedding: {
      completed: false,
      model: 'paraphrase-MiniLM-L6-v2',
      batchSize: 64,
      useParallel: true,
      openaiApiKey: '',
      openaiBaseUrl: ''
    },
    
    // Step 9: Storage
    storage: {
      completed: false,
      type: 'faiss',
      collectionName: '',
      metadata: true
    }
  });

  const {
    updateProcessStatus,
    setFileInfo,
    setApiResults
  } = useAppStore();

  const {
    setDeepConfigResults,
    setValidationResults,
    setConfigHistory
  } = useDeepConfigStore();

  const { showDatabaseModal, toggleDatabaseModal } = useUIStore();

  const steps = [
    { id: 0, title: 'Preprocess', icon: FileText, description: 'Load and clean data' },
    { id: 1, title: 'Type Convert', icon: Settings, description: 'Convert column data types' },
    { id: 2, title: 'Null Handle', icon: AlertCircle, description: 'Handle null values' },
    { id: 3, title: 'Duplicates', icon: RefreshCw, description: 'Remove duplicate rows' },
    { id: 4, title: 'Stopwords', icon: Scissors, description: 'Remove stop words' },
    { id: 5, title: 'Normalize', icon: Settings, description: 'Text normalization' },
    { id: 6, title: 'Chunk', icon: Scissors, description: 'Create text chunks' },
    { id: 7, title: 'Embed', icon: Brain, description: 'Generate embeddings' },
    { id: 8, title: 'Store', icon: HardDrive, description: 'Store vectors' }
  ];

  const handleConfigChange = (section, field, value) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }));
  };

  // Memoized callback for metadata column selection to prevent infinite re-renders
  const handleMetadataSelectionChange = useCallback((selection) => {
    handleConfigChange('chunking', 'numericColumns', selection.numeric);
    handleConfigChange('chunking', 'categoricalColumns', selection.categorical);
  }, []);

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

  const nextStep = () => {
    if (currentStep < 8) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleNextStep = () => {
    if (currentStep < 8) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleProcessStep = async (stepName, apiCall, configData = null, stepNumber = null) => {
    // Set global processing state based on step
    const stepMapping = {
      'Preprocessing': 'preprocessing',
      'Null Handling': 'preprocessing',
      'Duplicate Removal': 'preprocessing',
      'Type Conversion': 'preprocessing',
      'Chunking': 'chunking',
      'Embedding': 'embedding',
      'Storage': 'storage'
    };
    
    const globalStep = stepMapping[stepName] || 'preprocessing';
    setIsProcessing(true);
    
    toast.loading(`Processing ${stepName}...`);
    try {
      // Handle different API call signatures
      let response;
      if (stepName === 'Preprocessing') {
        response = await apiCall(file, dbConfig, configData);
      } else {
        // For other steps, pass configData directly
        response = await apiCall(configData);
      }
      toast.dismiss();
      
      // Handle different response structures
      const responseData = response.data || response;
      
      if (responseData && responseData.error) {
        toast.error(`Error in ${stepName}: ${responseData.error}`);
        return;
      }
      
      toast.success(`${stepName} completed!`);
      
      // Update global processing state
      updateProcessStatus(globalStep, 'completed');
      
      // Store step results
      if (stepNumber !== null) {
        setStepResults(prev => ({
          ...prev,
          [stepNumber]: responseData
        }));
      }
      
      // Update UI state with results
      if (responseData && responseData.data_preview) {
        setDataPreview(responseData.data_preview);
      }
      if (responseData && responseData.column_info) {
        setColumnInfo(responseData.column_info);
      }
      if (responseData && responseData.null_profile) {
        setNullProfile(responseData.null_profile);
      }
      if (responseData && responseData.duplicate_analysis) {
        setDuplicateAnalysis(responseData.duplicate_analysis);
      }
      if (responseData && responseData.metadata_columns) {
        setMetadataColumns(responseData.metadata_columns);
      }
      
      // Update apiResults for export functionality
      if (stepNumber === 6) {
        // After chunking - enable chunk download
        setApiResults({
          mode: 'deep',
          summary: {
            rows: stepResults[0]?.rows || 0,
            chunks: stepResults[6]?.chunks?.length || 0,
            stored: 'pending',
            collection_name: 'pending',
            retrieval_ready: false
          }
        });
      } else if (stepNumber === 7) {
        // After embedding - enable embedding download
        setApiResults({
          mode: 'deep',
          summary: {
            rows: stepResults[0]?.rows || 0,
            chunks: stepResults[6]?.chunks?.length || 0,
            stored: 'pending',
            collection_name: 'pending',
            retrieval_ready: false
          }
        });
      } else if (stepNumber === 8) {
        // After storage - enable all downloads and retrieval
        setApiResults({
          mode: 'deep',
          summary: {
            rows: stepResults[0]?.rows || 0,
            chunks: stepResults[6]?.chunks?.length || 0,
            stored: config.storage.type,
            collection_name: config.storage.collectionName,
            retrieval_ready: true
          }
        });
        setProcessingComplete(true);
      }
      
      handleNextStep();
    } catch (error) {
      toast.dismiss();
      toast.error(`API Error during ${stepName}: ${error.message}`);
      console.error(`API Error during ${stepName}:`, error);
    } finally {
      setIsProcessing(false);
    }
  };

  // Helper function to convert frontend type conversion config to API format
  const convertTypeConversionToAPI = (typeConversionConfig) => {
    const conversions = {};
    
    // Convert groupConversions and selectedColumns to simple column-to-type mapping
    if (typeConversionConfig.groupConversions && typeConversionConfig.selectedColumns) {
      Object.entries(typeConversionConfig.groupConversions).forEach(([currentType, targetType]) => {
        if (targetType && targetType !== 'No change') {
          Object.entries(typeConversionConfig.selectedColumns).forEach(([column, isSelected]) => {
            if (isSelected) {
              // Map frontend type names to API type names
              const apiType = targetType === 'int64' ? 'integer' : 
                             targetType === 'float64' ? 'float' :
                             targetType === 'datetime' ? 'datetime' :
                             targetType === 'boolean' ? 'boolean' :
                             targetType === 'object' ? 'string' : targetType;
              conversions[column] = apiType;
            }
          });
        }
      });
    }
    
    return conversions;
  };
  const groupColumnsByType = (columnInfo) => {
    if (!columnInfo || !columnInfo.columnNames || !columnInfo.dataTypes) {
      return {};
    }
    
    const groups = {};
    columnInfo.columnNames.forEach(col => {
      const currentType = columnInfo.dataTypes[col] || 'unknown';
      if (!groups[currentType]) {
        groups[currentType] = [];
      }
      groups[currentType].push(col);
    });
    
    return groups;
  };

  // Helper function to check if any type conversions are selected
  const hasTypeConversions = () => {
    const conversions = config.typeConversion.selectedColumns || {};
    return Object.values(conversions).some(Boolean);
  };

  // Helper function to get null handling options based on column data type
  const getNullOptionsForColumn = (dtype) => {
    if (dtype.includes('int') || dtype.includes('float')) {
      return [
        { value: 'No change', label: 'No change' },
        { value: 'drop', label: 'Drop rows' },
        { value: 'mean', label: 'Fill with mean' },
        { value: 'median', label: 'Fill with median' },
        { value: 'mode', label: 'Fill with mode' },
        { value: 'zero', label: 'Fill with zero' },
        { value: 'ffill', label: 'Forward fill' },
        { value: 'bfill', label: 'Backward fill' },
        { value: 'Custom Value', label: 'Custom value' }
      ];
    } else if (dtype === 'object') {
      return [
        { value: 'No change', label: 'No change' },
        { value: 'drop', label: 'Drop rows' },
        { value: 'mode', label: 'Fill with mode' },
        { value: 'unknown', label: 'Fill with "Unknown"' },
        { value: 'ffill', label: 'Forward fill' },
        { value: 'bfill', label: 'Backward fill' },
        { value: 'Custom Value', label: 'Custom value' }
      ];
    } else if (dtype.includes('datetime')) {
      return [
        { value: 'No change', label: 'No change' },
        { value: 'drop', label: 'Drop rows' },
        { value: 'ffill', label: 'Forward fill' },
        { value: 'bfill', label: 'Backward fill' },
        { value: 'Custom Value', label: 'Custom value' }
      ];
    } else if (dtype === 'bool') {
      return [
        { value: 'No change', label: 'No change' },
        { value: 'drop', label: 'Drop rows' },
        { value: 'mode', label: 'Fill with mode' },
        { value: 'ffill', label: 'Forward fill' },
        { value: 'bfill', label: 'Backward fill' },
        { value: 'Custom Value', label: 'Custom value' }
      ];
    } else {
      return [
        { value: 'No change', label: 'No change' },
        { value: 'drop', label: 'Drop rows' },
        { value: 'mode', label: 'Fill with mode' },
        { value: 'unknown', label: 'Fill with "Unknown"' },
        { value: 'ffill', label: 'Forward fill' },
        { value: 'bfill', label: 'Backward fill' },
        { value: 'Custom Value', label: 'Custom value' }
      ];
    }
  };

  // Helper function to get strategy descriptions
  const getStrategyDescription = (strategy) => {
    const descriptions = {
      'No change': 'Leave null values as-is',
      'drop': 'Remove rows containing null values',
      'mean': 'Fill nulls with column mean (numeric only)',
      'median': 'Fill nulls with column median (numeric only)',
      'mode': 'Fill nulls with most frequent value',
      'zero': 'Fill nulls with 0 (numeric) or "Unknown" (text)',
      'unknown': 'Fill nulls with "Unknown"',
      'ffill': 'Forward fill (use previous value)',
      'bfill': 'Backward fill (use next value)',
      'Custom Value': 'Fill nulls with user-provided value'
    };
    return descriptions[strategy] || '';
  };

  const executeStep = async (stepNumber) => {
    setIsProcessing(true);
    
    try {
      let result;
      
      switch (stepNumber) {
        case 0: // Preprocess
          result = await deepConfigService.preprocess(file, dbConfig);
          break;
        case 1: // Type Convert
          result = await deepConfigService.typeConvert(convertTypeConversionToAPI(config.typeConversion));
          break;
        case 2: // Null Handle
          result = await deepConfigService.nullHandle(config.nullHandling.strategies);
          break;
        case 3: // Duplicates
          result = await deepConfigService.duplicates(config.duplicates.strategy);
          break;
        case 4: // Stopwords
          result = await deepConfigService.stopwords(config.stopwords.enabled);
          break;
        case 5: // Normalize
          result = await deepConfigService.normalize(config.normalization.method);
          break;
        case 6: // Chunk
          result = await deepConfigService.chunk({
            method: config.chunking.method,
            chunkSize: config.chunking.chunkSize,
            overlap: config.chunking.overlap,
            keyColumn: config.chunking.keyColumn,
            tokenLimit: config.chunking.tokenLimit,
            nClusters: config.chunking.nClusters,
            storeMetadata: config.chunking.storeMetadata,
            numericColumns: config.chunking.numericColumns,
            categoricalColumns: config.chunking.categoricalColumns
          });
          break;
        case 7: // Embed
          result = await deepConfigService.embed({
            model: config.embedding.model,
            batchSize: config.embedding.batchSize,
            useParallel: config.embedding.useParallel,
            openaiApiKey: config.embedding.openaiApiKey,
            openaiBaseUrl: config.embedding.openaiBaseUrl
          });
          break;
        case 8: // Store
          result = await deepConfigService.store({
            type: config.storage.type,
            collectionName: config.storage.collectionName,
            metadata: config.storage.metadata
          });
          break;
        default:
          throw new Error(`Unknown step: ${stepNumber}`);
      }

      // Update step results
      setStepResults(prev => ({
        ...prev,
        [stepNumber]: result.data || result
      }));

      // Mark step as completed
      const stepKey = Object.keys(config)[stepNumber];
      handleConfigChange(stepKey, 'completed', true);

      // Update apiResults for export functionality when storage step is completed
      if (stepNumber === 8) {
        setApiResults({
          mode: 'deep',
          summary: {
            rows: stepResults[0]?.rows || 0,
            chunks: stepResults[6]?.chunks?.length || 0,
            stored: config.storage.type,
            collection_name: config.storage.collectionName,
            retrieval_ready: true
          }
        });
        setProcessingComplete(true);
      }

      // Update data preview and other state variables if available
      const responseData = result.data || result;
      
      // Handle different response structures based on step
      if (stepNumber === 0) {
        // Preprocessing step - has column_names and data_types directly
        if (responseData && responseData.column_names && responseData.data_types) {
          setColumnInfo({
            columnNames: responseData.column_names,
            dataTypes: responseData.data_types,
            sampleValues: responseData.sample_values || {}
          });
        }
        if (responseData && responseData.data_preview) {
          setDataPreview(responseData.data_preview);
        }
        
        // After preprocessing, analyze nulls and duplicates
        try {
          const nullAnalysis = await deepConfigService.analyzeNulls();
          if (nullAnalysis && nullAnalysis.null_profile) {
            setNullProfile(nullAnalysis.null_profile);
          }
          
          const duplicateAnalysis = await deepConfigService.analyzeDuplicates();
          if (duplicateAnalysis && duplicateAnalysis.duplicate_analysis) {
            setDuplicateAnalysis(duplicateAnalysis.duplicate_analysis);
          }
        } catch (error) {
          console.error('Error analyzing nulls/duplicates:', error);
        }
      } else {
        // Other steps - use the expected structure
        if (responseData && responseData.data_preview) {
          setDataPreview(responseData.data_preview);
        }
        if (responseData && responseData.column_info) {
          setColumnInfo(responseData.column_info);
        }
        if (responseData && responseData.null_profile) {
          setNullProfile(responseData.null_profile);
        }
        if (responseData && responseData.duplicate_analysis) {
          setDuplicateAnalysis(responseData.duplicate_analysis);
        }
        if (responseData && responseData.metadata_columns) {
          setMetadataColumns(responseData.metadata_columns);
        }
      }

      // Move to next step
      if (stepNumber < 8) {
        setCurrentStep(stepNumber + 1);
      }

    } catch (error) {
      console.error(`Step ${stepNumber} error:`, error);
      // Handle error - could show toast notification
    } finally {
      setIsProcessing(false);
    }
  };

  // Debug logging for step results
  useEffect(() => {
    if (currentStep === 2) { // Step 3 (0-indexed)
      console.log('Step 3 - stepResults:', stepResults);
      console.log('Step 3 - stepResults[0]:', stepResults[0]);
      console.log('Step 3 - stepResults[1]:', stepResults[1]);
    }
  }, [currentStep, stepResults]);

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <h3 className="text-lg font-medium text-textPrimary mb-2">Step 1: Data Preprocessing</h3>
              <p className="text-textSecondary text-sm">Load and clean your data</p>
            </div>

            <div className="space-y-4">
              <label className="flex items-center space-x-3 cursor-pointer p-4 border border-border rounded-lg hover:bg-secondary transition-colors">
                <input
                  type="radio"
                  name="dataSource"
                  value="file"
                  checked={inputMethod === 'file'}
                  onChange={(e) => setInputMethod(e.target.value)}
                  className="text-highlight focus:ring-highlight"
                />
                <Upload className="h-6 w-6 text-highlight" />
                <div>
                  <div className="font-medium text-textPrimary">File Upload</div>
                  <div className="text-sm text-textSecondary">Upload CSV files directly</div>
                </div>
              </label>

              <label className="flex items-center space-x-3 cursor-pointer p-4 border border-border rounded-lg hover:bg-secondary transition-colors">
                <input
                  type="radio"
                  name="dataSource"
                  value="database"
                  checked={inputMethod === 'database'}
                  onChange={(e) => setInputMethod(e.target.value)}
                  className="text-highlight focus:ring-highlight"
                />
                <Database className="h-6 w-6 text-highlight" />
                <div>
                  <div className="font-medium text-textPrimary">Database Import</div>
                  <div className="text-sm text-textSecondary">Connect to MySQL/PostgreSQL</div>
                </div>
              </label>
            </div>

            {inputMethod === 'file' && (
              <div className="mt-6">
                <FileUpload
                  onFileSelect={handleFileSelect}
                  file={file}
                  disabled={isProcessing}
                />
              </div>
            )}

            {inputMethod === 'database' && (
              <div className="mt-6">
                <Button
                  variant="outline"
                  onClick={toggleDatabaseModal}
                  disabled={isProcessing}
                  className="w-full"
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

            {(file || dbConfig) && (
              <div className="mt-6">
                <Button
                  variant="primary"
                  onClick={() => executeStep(0)}
                  disabled={isProcessing}
                  loading={isProcessing}
                  className="w-full"
                >
                  <Play className="h-5 w-5 mr-2" />
                  {isProcessing ? 'Processing...' : 'Start Preprocessing'}
                </Button>
              </div>
            )}

            {config.preprocessing.completed && (
              <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <CheckCircle className="h-5 w-5 text-green-400" />
                  <span className="font-medium text-green-400">Preprocessing Completed</span>
                </div>
                <div className="text-sm text-textSecondary">
                  <p><strong>Rows:</strong> {stepResults[0]?.rows || 'N/A'}</p>
                  <p><strong>Columns:</strong> {stepResults[0]?.columns || 'N/A'}</p>
                  <p><strong>Data Types:</strong> {stepResults[0]?.column_names?.length || 0}</p>
                </div>
              </div>
            )}

            {/* Enhanced Data Preview */}
            {stepResults[0]?.data_preview && (
              <DatasetPreviewTable
                data={stepResults[0].data_preview}
                columns={stepResults[0].column_names || []}
                totalRows={stepResults[0].rows || 0}
              />
            )}
          </div>
        );

      case 1:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <h3 className="text-lg font-medium text-textPrimary mb-2">Step 2: Type Conversion</h3>
              <p className="text-textSecondary text-sm">Convert column data types</p>
            </div>

            {/* Enhanced Type Conversion Table */}
            {stepResults[0]?.column_names && stepResults[0]?.data_types && (
              <TypeConversionTable
                columns={stepResults[0].column_names}
                dataTypes={stepResults[0].data_types}
                sampleValues={stepResults[0].sample_values || {}}
                onConvert={(conversions) => {
                  const formData = new FormData();
                  formData.append('type_conversions', JSON.stringify(conversions));
                  handleProcessStep('Type Conversion', deepConfigService.typeConvert, formData, 1);
                }}
              />
            )}

            {/* Action Buttons */}
            <div className="flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={prevStep}
                disabled={isProcessing}
                className="flex items-center space-x-1"
              >
                <ArrowLeft className="h-3 w-3" />
                <span>Prev</span>
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  toast.success('Type conversion skipped.');
                  nextStep();
                }}
                disabled={isProcessing}
                className="flex items-center space-x-1"
              >
                <span>Skip</span>
              </Button>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <h3 className="text-lg font-medium text-textPrimary mb-2">Step 3: Null Handling</h3>
              <p className="text-textSecondary text-sm">Handle null values with smart strategies</p>
            </div>

            {/* Enhanced Null Analysis Table */}
            {nullProfile && nullProfile.length > 0 ? (
              <div>
                <div className="mb-4 p-3 bg-secondary rounded-lg">
                  <h4 className="font-medium text-textPrimary mb-2">Null Analysis Available</h4>
                  <div className="text-sm text-textSecondary">
                    <p>Columns with nulls: {nullProfile.length}</p>
                    <p>Total columns: {(stepResults[1]?.column_names || stepResults[0]?.column_names)?.length || 0}</p>
                    <p>Data types: {Object.keys(stepResults[1]?.data_types || stepResults[0]?.data_types || {}).length}</p>
                  </div>
                </div>
                <NullAnalysisTable
                  nullProfile={nullProfile}
                  columns={stepResults[1]?.column_names || stepResults[0]?.column_names}
                  dataTypes={stepResults[1]?.data_types || stepResults[0]?.data_types}
                  onApply={(strategies) => {
                    console.log('Applying null strategies:', strategies);
                    handleProcessStep('Null Handling', deepConfigService.nullHandle, strategies, 2);
                  }}
                />
              </div>
            ) : (
              <div className="p-6 bg-secondary rounded-lg text-center">
                <div className="text-textSecondary mb-2">
                  <CheckCircle className="h-8 w-8 mx-auto mb-3 text-green-400" />
                  <h4 className="font-medium text-textPrimary mb-2">No Null Values Detected!</h4>
                  <p className="text-sm text-textSecondary">
                    All columns in your dataset are complete. You can proceed to the next step.
                  </p>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={prevStep}
                disabled={isProcessing}
                className="flex items-center space-x-1"
              >
                <ArrowLeft className="h-3 w-3" />
                <span>Prev</span>
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => {
                  toast.success('Null handling skipped.');
                  nextStep();
                }}
                disabled={isProcessing}
                className="flex items-center space-x-1"
              >
                <span>Skip</span>
              </Button>
            </div>

            {/* Show updated preview after null handling */}
            {stepResults[2]?.data_preview && (
              <DatasetPreviewTable
                data={stepResults[2].data_preview}
                columns={stepResults[2].column_names || []}
                totalRows={stepResults[2].rows || 0}
              />
            )}
          </div>
        );

      case 3:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <h3 className="text-lg font-medium text-textPrimary mb-2">Step 4: Duplicate Removal</h3>
              <p className="text-textSecondary text-sm">Remove duplicate rows</p>
            </div>

            {/* Enhanced Duplicate Analysis Card */}
            <DuplicateAnalysisCard
              onApply={(result) => {
                // Store the result and proceed to next step
                setStepResults(prev => ({
                  ...prev,
                  [3]: result
                }));
                nextStep();
              }}
            />

            {/* Show updated preview after duplicate removal */}
            {stepResults[3]?.data_preview && (
              <DatasetPreviewTable
                data={stepResults[3].data_preview}
                columns={stepResults[3].column_names || []}
                totalRows={stepResults[3].rows || 0}
              />
            )}

            {config.duplicates.completed && (
              <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <CheckCircle className="h-5 w-5 text-green-400" />
                  <span className="font-medium text-green-400">Duplicate Removal Completed</span>
                </div>
                <div className="text-sm text-textSecondary">
                  <p><strong>Strategy:</strong> {config.duplicates.strategy}</p>
                  <p><strong>Removed:</strong> {config.duplicates.removedCount} duplicate rows</p>
                </div>
              </div>
            )}
          </div>
        );

      case 4:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <h3 className="text-lg font-medium text-textPrimary mb-2">Step 5: Stopword Removal</h3>
              <p className="text-textSecondary text-sm">Remove common stop words (optional)</p>
            </div>

            <div className="space-y-4">
              <div className="p-4 bg-secondary rounded-lg">
                <h4 className="font-medium text-textPrimary mb-3">Stopword Options</h4>
                <div className="space-y-3">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={config.stopwords.enabled}
                      onChange={(e) => handleConfigChange('stopwords', 'enabled', e.target.checked)}
                      className="text-highlight focus:ring-highlight"
                    />
                    <span className="text-textPrimary">Remove stop words using spaCy</span>
                  </label>
                  <p className="text-sm text-textSecondary">
                    This will remove common English words like "the", "and", "is", etc. from text columns.
                  </p>
                </div>
              </div>
            </div>

            <div className="flex space-x-4">
              <Button
                variant="outline"
                onClick={prevStep}
                disabled={isProcessing}
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Previous
              </Button>
              
              <Button
                variant="primary"
                onClick={() => executeStep(4)}
                disabled={isProcessing}
                loading={isProcessing}
              >
                <Play className="h-5 w-5 mr-2" />
                {isProcessing ? 'Processing...' : 'Process Stopwords'}
              </Button>
            </div>

            {config.stopwords.completed && (
              <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <CheckCircle className="h-5 w-5 text-green-400" />
                  <span className="font-medium text-green-400">Stopword Processing Completed</span>
                </div>
                <div className="text-sm text-textSecondary">
                  <p><strong>Enabled:</strong> {config.stopwords.enabled ? 'Yes' : 'No'}</p>
                </div>
              </div>
            )}
          </div>
        );

      case 5:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <h3 className="text-lg font-medium text-textPrimary mb-2">Step 6: Text Normalization</h3>
              <p className="text-textSecondary text-sm">Normalize text using lemmatization or stemming</p>
            </div>

            <div className="space-y-4">
              <div className="p-4 bg-secondary rounded-lg">
                <h4 className="font-medium text-textPrimary mb-3">Normalization Method</h4>
                <Select
                  value={config.normalization.method}
                  onChange={(e) => handleConfigChange('normalization', 'method', e.target.value)}
                  options={[
                    { value: 'none', label: 'No Normalization' },
                    { value: 'lemmatize', label: 'Lemmatization (Recommended)' },
                    { value: 'stem', label: 'Stemming' }
                  ]}
                />
                <p className="text-sm text-textSecondary mt-2">
                  Lemmatization reduces words to their root form (e.g., "running" → "run").
                  Stemming is faster but less accurate.
                </p>
              </div>
            </div>

            <div className="flex space-x-4">
              <Button
                variant="outline"
                onClick={prevStep}
                disabled={isProcessing}
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Previous
              </Button>
              
              <Button
                variant="primary"
                onClick={() => executeStep(5)}
                disabled={isProcessing}
                loading={isProcessing}
              >
                <Play className="h-5 w-5 mr-2" />
                {isProcessing ? 'Processing...' : 'Apply Normalization'}
              </Button>
            </div>

            {config.normalization.completed && (
              <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <CheckCircle className="h-5 w-5 text-green-400" />
                  <span className="font-medium text-green-400">Normalization Completed</span>
                </div>
                <div className="text-sm text-textSecondary">
                  <p><strong>Method:</strong> {config.normalization.method}</p>
                </div>
              </div>
            )}
          </div>
        );

      case 6:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <h3 className="text-lg font-medium text-textPrimary mb-2">Step 7: Chunking</h3>
              <p className="text-textSecondary text-sm">Create text chunks with metadata extraction</p>
            </div>

            <div className="space-y-4">
              <Select
                label="Chunking Method"
                value={config.chunking.method}
                onChange={(e) => handleConfigChange('chunking', 'method', e.target.value)}
                options={[
                  { value: 'fixed', label: 'Fixed Size' },
                  { value: 'recursive', label: 'Recursive' },
                  { value: 'semantic', label: 'Semantic Clustering' },
                  { value: 'document', label: 'Document Based' },
                  { value: 'agentic', label: '🤖 Agentic (AI-Powered)' }
                ]}
              />

              {/* Method-specific parameters */}
              {config.chunking.method === 'fixed' && (
                <div className="p-4 bg-secondary rounded-lg">
                  <div className="flex items-center space-x-2 mb-3">
                    <Info className="h-4 w-4 text-blue-400" />
                    <span className="text-sm text-textSecondary">Splits data into fixed-size chunks of characters with overlap</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Input
                      label="Chunk Size (characters)"
                      type="number"
                      min="50"
                      max="20000"
                      step="50"
                      value={config.chunking.chunkSize}
                      onChange={(e) => handleConfigChange('chunking', 'chunkSize', parseInt(e.target.value))}
                      placeholder="400"
                    />
                    <Input
                      label="Overlap (characters)"
                      type="number"
                      min="0"
                      max={config.chunking.chunkSize - 1}
                      value={config.chunking.overlap}
                      onChange={(e) => handleConfigChange('chunking', 'overlap', parseInt(e.target.value))}
                      placeholder="50"
                    />
                  </div>
                </div>
              )}

              {config.chunking.method === 'recursive' && (
                <div className="p-4 bg-secondary rounded-lg">
                  <div className="flex items-center space-x-2 mb-3">
                    <Info className="h-4 w-4 text-blue-400" />
                    <span className="text-sm text-textSecondary">Splits key-value formatted lines with recursive separators and overlap</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Input
                      label="Chunk Size (characters)"
                      type="number"
                      min="50"
                      max="20000"
                      step="50"
                      value={config.chunking.chunkSize}
                      onChange={(e) => handleConfigChange('chunking', 'chunkSize', parseInt(e.target.value))}
                      placeholder="400"
                    />
                    <Input
                      label="Overlap (characters)"
                      type="number"
                      min="0"
                      max={config.chunking.chunkSize - 1}
                      value={config.chunking.overlap}
                      onChange={(e) => handleConfigChange('chunking', 'overlap', parseInt(e.target.value))}
                      placeholder="50"
                    />
                  </div>
                </div>
              )}

              {config.chunking.method === 'semantic' && (
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
                    value={config.chunking.nClusters}
                    onChange={(e) => handleConfigChange('chunking', 'nClusters', parseInt(e.target.value))}
                    placeholder="10"
                  />
                </div>
              )}

              {config.chunking.method === 'document' && (
                <div className="p-4 bg-secondary rounded-lg">
                  <div className="flex items-center space-x-2 mb-3">
                    <Info className="h-4 w-4 text-blue-400" />
                    <span className="text-sm text-textSecondary">Group by a key column and split by token limit (headers optional)</span>
                  </div>
                  <div className="space-y-3">
                    <Select
                      label="Key Column"
                      value={config.chunking.keyColumn || ''}
                      onChange={(e) => handleConfigChange('chunking', 'keyColumn', e.target.value)}
                      options={(() => {
                        if (!stepResults[0]?.data_preview) {
                          return [{ value: '', label: 'Run preprocessing first to see columns' }];
                        }
                        const groupingColumns = getDocumentGroupingColumns(stepResults[0].data_preview);
                        return groupingColumns.length > 0 ? groupingColumns : [
                          { value: '', label: 'No suitable grouping columns found' }
                        ];
                      })()}
                      disabled={!stepResults[0]?.data_preview}
                    />
                    <Input
                      label="Token limit per chunk"
                      type="number"
                      min="200"
                      max="10000"
                      step="100"
                      value={config.chunking.tokenLimit}
                      onChange={(e) => handleConfigChange('chunking', 'tokenLimit', parseInt(e.target.value))}
                      placeholder="2000"
                    />
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={config.chunking.preserveHeaders}
                        onChange={(e) => handleConfigChange('chunking', 'preserveHeaders', e.target.checked)}
                        className="text-highlight focus:ring-highlight"
                      />
                      <span className="text-textPrimary">Include headers in each chunk</span>
                    </label>
                  </div>
                </div>
              )}

              {/* Agentic Chunking Configuration */}
              {config.chunking.method === 'agentic' && (
                <div className="p-4 bg-secondary rounded-lg border-2 border-highlight/30">
                  <div className="flex items-center space-x-2 mb-4">
                    <Brain className="h-5 w-5 text-highlight" />
                    <h4 className="font-medium text-textPrimary">Agentic Chunking Configuration</h4>
                  </div>
                  
                  <div className="space-y-4">
                    <Select
                      label="Chunking Strategy"
                      value={config.chunking.agenticStrategy || 'auto'}
                      onChange={(e) => handleConfigChange('chunking', 'agenticStrategy', e.target.value)}
                      options={[
                        { value: 'auto', label: '🤖 Auto (AI Decides Best Strategy)' },
                        { value: 'schema', label: 'Schema-Aware (Analyzes table structure)' },
                        { value: 'entity', label: 'Entity-Centric (Groups by entities)' }
                      ]}
                    />
                    
                    <Input
                      label="User Context (Optional)"
                      value={config.chunking.userContext || ''}
                      onChange={(e) => handleConfigChange('chunking', 'userContext', e.target.value)}
                      placeholder="e.g., 'Analyze sales by region' or 'Focus on customer behavior'"
                    />
                    
                    <div className="p-3 bg-primary rounded-lg border border-highlight/20">
                      <div className="flex items-start space-x-2">
                        <Info className="h-4 w-4 text-highlight mt-0.5 flex-shrink-0" />
                        <div className="text-sm text-textSecondary">
                          <p className="font-medium text-textPrimary mb-1">How Agentic Chunking Works:</p>
                          <ul className="list-disc list-inside space-y-1">
                            <li>AI analyzes your data structure and column relationships</li>
                            <li>Identifies entities (users, products, companies)</li>
                            <li>Decides optimal grouping strategy automatically</li>
                            <li>Preserves context and semantic meaning</li>
                          </ul>
                          <p className="mt-2 text-xs text-highlight">
                            ⚠️ Requires GEMINI_API_KEY to be set in environment
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="p-4 bg-secondary rounded-lg">
                <h4 className="font-medium text-textPrimary mb-3">Metadata Extraction</h4>
                <div className="space-y-3">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={config.chunking.storeMetadata}
                      onChange={(e) => handleConfigChange('chunking', 'storeMetadata', e.target.checked)}
                      className="text-highlight focus:ring-highlight"
                    />
                    <span className="text-textPrimary">Store metadata for filtering</span>
                  </label>
                  
                  {config.chunking.storeMetadata && (
                    <div className="mt-4">
                      <MetadataColumnSelector
                        onSelectionChange={handleMetadataSelectionChange}
                        initialNumeric={config.chunking.numericColumns}
                        initialCategorical={config.chunking.categoricalColumns}
                        disabled={isProcessing}
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div className="flex space-x-4">
              <Button
                variant="outline"
                onClick={prevStep}
                disabled={isProcessing}
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Previous
              </Button>
              
              <Button
                variant="primary"
                onClick={() => executeStep(6)}
                disabled={isProcessing}
                loading={isProcessing}
              >
                <Play className="h-5 w-5 mr-2" />
                {isProcessing ? 'Chunking...' : 'Create Chunks'}
              </Button>
            </div>

            {config.chunking.completed && (
              <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <CheckCircle className="h-5 w-5 text-green-400" />
                  <span className="font-medium text-green-400">Chunking Completed</span>
                </div>
                <div className="text-sm text-textSecondary">
                  <p><strong>Method:</strong> {config.chunking.method}</p>
                  <p><strong>Chunks Created:</strong> {stepResults[6]?.chunkCount || 'N/A'}</p>
                  <p><strong>Metadata:</strong> {config.chunking.storeMetadata ? 'Enabled' : 'Disabled'}</p>
                </div>
              </div>
            )}

            {/* Chunked Data Preview */}
            {stepResults[6]?.chunks && (
              <DataPreviewTable
                data={stepResults[6].chunks}
                title="📊 Chunked Data Preview"
                maxRows={5}
                maxCols={3}
                className="mt-6"
              />
            )}
          </div>
        );

      case 7:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <h3 className="text-lg font-medium text-textPrimary mb-2">Step 8: Embedding Generation</h3>
              <p className="text-textSecondary text-sm">Generate vector embeddings for semantic search</p>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Select
                  label="Embedding Model"
                  value={config.embedding.model}
                  onChange={(e) => handleConfigChange('embedding', 'model', e.target.value)}
                  options={[
                    { value: 'paraphrase-MiniLM-L6-v2', label: 'paraphrase-MiniLM-L6-v2' },
                    { value: 'all-MiniLM-L6-v2', label: 'all-MiniLM-L6-v2' },
                    { value: 'paraphrase-mpnet-base-v2', label: 'paraphrase-mpnet-base-v2' },
                    { value: 'text-embedding-ada-002', label: 'OpenAI Ada-002' }
                  ]}
                />

                <Input
                  label="Batch Size"
                  type="number"
                  value={config.embedding.batchSize}
                  onChange={(e) => handleConfigChange('embedding', 'batchSize', parseInt(e.target.value))}
                  placeholder="64"
                />
              </div>

              <div className="space-y-3">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.embedding.useParallel}
                    onChange={(e) => handleConfigChange('embedding', 'useParallel', e.target.checked)}
                    className="text-highlight focus:ring-highlight"
                  />
                  <span className="text-textPrimary">Use Parallel Processing</span>
                </label>
              </div>

              {config.embedding.model === 'text-embedding-ada-002' && (
                <div className="p-4 bg-secondary rounded-lg">
                  <h4 className="font-medium text-textPrimary mb-3">OpenAI Configuration</h4>
                  <div className="space-y-3">
                    <Input
                      label="OpenAI API Key"
                      type="password"
                      value={config.embedding.openaiApiKey}
                      onChange={(e) => handleConfigChange('embedding', 'openaiApiKey', e.target.value)}
                      placeholder="Enter your OpenAI API key"
                    />
                    <Input
                      label="OpenAI Base URL (Optional)"
                      value={config.embedding.openaiBaseUrl}
                      onChange={(e) => handleConfigChange('embedding', 'openaiBaseUrl', e.target.value)}
                      placeholder="https://api.openai.com/v1"
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="flex space-x-4">
              <Button
                variant="outline"
                onClick={prevStep}
                disabled={isProcessing}
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Previous
              </Button>
              
              <Button
                variant="primary"
                onClick={() => executeStep(7)}
                disabled={isProcessing}
                loading={isProcessing}
              >
                <Play className="h-5 w-5 mr-2" />
                {isProcessing ? 'Generating...' : 'Generate Embeddings'}
              </Button>
            </div>

            {config.embedding.completed && (
              <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <CheckCircle className="h-5 w-5 text-green-400" />
                  <span className="font-medium text-green-400">Embedding Generation Completed</span>
                </div>
                <div className="text-sm text-textSecondary">
                  <p><strong>Model:</strong> {config.embedding.model}</p>
                  <p><strong>Batch Size:</strong> {config.embedding.batchSize}</p>
                  <p><strong>Parallel:</strong> {config.embedding.useParallel ? 'Yes' : 'No'}</p>
                </div>
              </div>
            )}
          </div>
        );

      case 8:
        return (
          <div className="space-y-6">
            <div className="text-center mb-6">
              <h3 className="text-lg font-medium text-textPrimary mb-2">Step 9: Vector Storage</h3>
              <p className="text-textSecondary text-sm">Store vectors for semantic search</p>
            </div>

            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Select
                  label="Storage Type"
                  value={config.storage.type}
                  onChange={(e) => handleConfigChange('storage', 'type', e.target.value)}
                  options={[
                    { value: 'faiss', label: 'FAISS' },
                    { value: 'chroma', label: 'ChromaDB' }
                  ]}
                />

                <Input
                  label="Collection Name"
                  value={config.storage.collectionName}
                  onChange={(e) => handleConfigChange('storage', 'collectionName', e.target.value)}
                  placeholder="deep_config_collection"
                />
              </div>

              <div className="space-y-3">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={config.storage.metadata}
                    onChange={(e) => handleConfigChange('storage', 'metadata', e.target.checked)}
                    className="text-highlight focus:ring-highlight"
                  />
                  <span className="text-textPrimary">Store metadata for filtering</span>
                </label>
              </div>
            </div>

            <div className="flex space-x-4">
              <Button
                variant="outline"
                onClick={prevStep}
                disabled={isProcessing}
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Previous
              </Button>
              
              <Button
                variant="primary"
                onClick={() => executeStep(8)}
                disabled={isProcessing}
                loading={isProcessing}
              >
                <Play className="h-5 w-5 mr-2" />
                {isProcessing ? 'Storing...' : 'Store Vectors'}
              </Button>
            </div>

            {config.storage.completed && (
              <div className="mt-6 p-4 bg-green-900/20 border border-green-500/30 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <CheckCircle className="h-5 w-5 text-green-400" />
                  <span className="font-medium text-green-400">Vector Storage Completed</span>
                </div>
                <div className="text-sm text-textSecondary">
                  <p><strong>Storage Type:</strong> {config.storage.type}</p>
                  <p><strong>Collection:</strong> {config.storage.collectionName}</p>
                  <p><strong>Metadata:</strong> {config.storage.metadata ? 'Enabled' : 'Disabled'}</p>
                </div>
              </div>
            )}

            {config.storage.completed && (
              <div className="mt-6 p-6 bg-gradient-to-r from-green-900/20 to-blue-900/20 border border-green-500/30 rounded-lg">
                <div className="text-center">
                  <CheckCircle className="h-12 w-12 text-green-400 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-textPrimary mb-2">Deep Config Pipeline Complete!</h3>
                  <p className="text-textSecondary mb-4">
                    Your data has been processed through all 9 steps and is ready for semantic search.
                  </p>
                  <div className="flex justify-center space-x-4">
                    <Button
                      variant="primary"
                      onClick={() => {
                        // Navigate to search interface
                        // This would typically update the app state
                      }}
                    >
                      <Search className="h-5 w-5 mr-2" />
                      Start Searching
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => {
                        // Reset the pipeline
                        setCurrentStep(0);
                        setConfig({
                          preprocessing: { completed: false, fileInfo: null, dataTypes: {}, columnNames: [] },
                          typeConversion: { completed: false, conversions: {}, suggestions: {} },
                          nullHandling: { completed: false, nullProfile: [], strategies: {} },
                          duplicates: { completed: false, strategy: 'keep_first', removedCount: 0 },
                          stopwords: { completed: false, enabled: false },
                          normalization: { completed: false, method: 'none' },
                          chunking: { completed: false, method: 'fixed', chunkSize: 400, overlap: 50, keyColumn: null, tokenLimit: 2000, nClusters: 10, storeMetadata: true, numericColumns: [], categoricalColumns: [] },
                          embedding: { completed: false, model: 'paraphrase-MiniLM-L6-v2', batchSize: 64, useParallel: true, openaiApiKey: '', openaiBaseUrl: '' },
                          storage: { completed: false, type: 'faiss', collectionName: '', metadata: true }
                        });
                        setStepResults({});
                        setDataPreview(null);
                      }}
                    >
                      <RefreshCw className="h-5 w-5 mr-2" />
                      Start New Pipeline
                    </Button>
                  </div>
                </div>
              </div>
            )}
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
            <h2 className="text-xl font-semibold text-textPrimary">Deep Config Mode</h2>
            <p className="text-textSecondary text-sm">9-step advanced data processing pipeline</p>
          </div>
        </div>

        {/* Step Progress */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <span className="text-sm text-textSecondary">
              Step {currentStep + 1} of {steps.length}
            </span>
            <span className="text-sm text-textSecondary">
              {Math.round(((currentStep + 1) / steps.length) * 100)}% Complete
            </span>
          </div>
          
          <div className="flex space-x-2">
            {steps.map((step) => {
              const Icon = step.icon;
              const isActive = currentStep === step.id;
              const isCompleted = config[Object.keys(config)[step.id]]?.completed;
              
              return (
                <div
                  key={step.id}
                  className={`flex-1 h-2 rounded-full transition-colors ${
                    isActive ? 'bg-highlight' : 
                    isCompleted ? 'bg-green-400' : 'bg-secondary'
                  }`}
                />
              );
            })}
          </div>
        </div>

        {/* Step Navigation */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={prevStep}
              disabled={currentStep === 0}
              className="flex items-center space-x-1"
            >
              <ChevronLeft className="h-3 w-3" />
              <span>Prev</span>
            </Button>
            
            <Button
              variant="outline"
              size="sm"
              onClick={nextStep}
              disabled={currentStep === 8}
              className="flex items-center space-x-1"
            >
              <span>Next</span>
              <ChevronRight className="h-3 w-3" />
            </Button>
          </div>

          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                // Save configuration
                setConfigHistory(config);
              }}
              className="flex items-center space-x-1"
            >
              <Save className="h-3 w-3" />
              <span>Save</span>
            </Button>
          </div>
        </div>

        {/* Step Content */}
        <div>
          {renderStepContent()}
        </div>
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

export default DeepConfigMode;