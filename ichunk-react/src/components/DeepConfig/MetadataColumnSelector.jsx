import React, { useState, useEffect } from 'react';
import { CheckCircle, AlertCircle, Loader, Hash, Tag, Calendar } from 'lucide-react';
import Card from '../UI/Card';
import Button from '../UI/Button';
import useUIStore from '../../stores/uiStore';
import { deepConfigService } from '../../services/deepConfig.service';

const MetadataColumnSelector = ({ 
  onSelectionChange, 
  initialNumeric = [], 
  initialCategorical = [],
  disabled = false 
}) => {
  const [metadataInfo, setMetadataInfo] = useState(null);
  const [selectedNumeric, setSelectedNumeric] = useState(initialNumeric);
  const [selectedCategorical, setSelectedCategorical] = useState(initialCategorical);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const { addNotification } = useUIStore();

  useEffect(() => {
    fetchMetadataColumns();
  }, []);

  useEffect(() => {
    // Notify parent of selection changes
    onSelectionChange({
      numeric: selectedNumeric,
      categorical: selectedCategorical,
      total: selectedNumeric.length + selectedCategorical.length
    });
  }, [selectedNumeric, selectedCategorical, onSelectionChange]);

  const fetchMetadataColumns = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await deepConfigService.getMetadataColumns();
      if (response && response.status === 'success') {
        setMetadataInfo(response);
      } else {
        setError(response?.error || 'Failed to fetch metadata columns');
        addNotification({ type: 'error', message: 'Failed to fetch metadata columns' });
      }
    } catch (error) {
      console.error('Error fetching metadata columns:', error);
      setError('Failed to fetch metadata columns');
      addNotification({ type: 'error', message: 'Failed to fetch metadata columns' });
    } finally {
      setLoading(false);
    }
  };

  const handleNumericToggle = (column) => {
    if (disabled) return;
    
    const maxNumeric = metadataInfo?.max_numeric || 0;
    const isSelected = selectedNumeric.includes(column);
    
    if (isSelected) {
      setSelectedNumeric(prev => prev.filter(col => col !== column));
    } else if (selectedNumeric.length < maxNumeric) {
      setSelectedNumeric(prev => [...prev, column]);
    } else {
      addNotification({ 
        type: 'warning', 
        message: `Maximum ${maxNumeric} numeric columns allowed` 
      });
    }
  };

  const handleCategoricalToggle = (column) => {
    if (disabled) return;
    
    const maxCategorical = metadataInfo?.max_categorical || 0;
    const isSelected = selectedCategorical.includes(column);
    
    if (isSelected) {
      setSelectedCategorical(prev => prev.filter(col => col !== column));
    } else if (selectedCategorical.length < maxCategorical) {
      setSelectedCategorical(prev => [...prev, column]);
    } else {
      addNotification({ 
        type: 'warning', 
        message: `Maximum ${maxCategorical} categorical columns allowed` 
      });
    }
  };

  const getColumnIcon = (column) => {
    // Simple heuristic based on column name
    const name = column.toLowerCase();
    if (name.includes('date') || name.includes('time')) {
      return <Calendar className="h-4 w-4 text-blue-400" />;
    }
    if (name.includes('id') || name.includes('num') || name.includes('count')) {
      return <Hash className="h-4 w-4 text-green-400" />;
    }
    return <Tag className="h-4 w-4 text-purple-400" />;
  };

  const getCardinalityColor = (cardinality) => {
    if (cardinality <= 5) return 'text-green-400';
    if (cardinality <= 10) return 'text-yellow-400';
    if (cardinality <= 20) return 'text-orange-400';
    return 'text-red-400';
  };

  if (loading) {
    return (
      <Card className="text-center py-8">
        <Loader className="h-8 w-8 mx-auto mb-4 animate-spin text-highlight" />
        <p className="text-textSecondary">Loading metadata columns...</p>
      </Card>
    );
  }

  if (error || !metadataInfo) {
    return (
      <Card className="text-center py-8">
        <AlertCircle className="h-8 w-8 mx-auto mb-4 text-red-400" />
        <p className="text-red-400 mb-2">Failed to load metadata columns</p>
        <Button variant="outline" onClick={fetchMetadataColumns}>
          Retry
        </Button>
      </Card>
    );
  }

  const { 
    numeric_columns = [], 
    categorical_columns = [], 
    max_numeric = 0,
    max_categorical = 0,
    numeric_samples = {},
    categorical_samples = {}
  } = metadataInfo;

  const maxNumeric = max_numeric;
  const maxCategorical = max_categorical;
  const totalSelected = selectedNumeric.length + selectedCategorical.length;
  const maxTotal = maxNumeric + maxCategorical;

  return (
    <Card>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-textPrimary">Metadata Column Selection</h3>
            <p className="text-sm text-textSecondary">
              Select columns to include as metadata for enhanced search and filtering
            </p>
          </div>
          <div className="text-right">
            <div className="text-sm text-textSecondary">Selection</div>
            <div className="text-lg font-semibold text-highlight">
              {totalSelected}/{maxTotal}
            </div>
          </div>
        </div>

        {/* Limits Info */}
        <div className="grid grid-cols-2 gap-4 p-4 bg-secondary rounded-lg border border-border">
          <div className="text-center">
            <div className="text-sm text-textSecondary">Numeric Columns</div>
            <div className="text-lg font-semibold text-textPrimary">
              {selectedNumeric.length}/{maxNumeric}
            </div>
          </div>
          <div className="text-center">
            <div className="text-sm text-textSecondary">Categorical Columns</div>
            <div className="text-lg font-semibold text-textPrimary">
              {selectedCategorical.length}/{maxCategorical}
            </div>
          </div>
        </div>

        {/* Numeric Columns */}
        {numeric_columns.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <Hash className="h-5 w-5 text-green-400" />
              <h4 className="font-medium text-textPrimary">Numeric Columns</h4>
              <span className="text-sm text-textSecondary">
                ({selectedNumeric.length}/{maxNumeric} selected)
              </span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {numeric_columns.map(column => {
                const isSelected = selectedNumeric.includes(column);
                const sampleValue = numeric_samples[column] || 'N/A';
                const canSelect = selectedNumeric.length < maxNumeric || isSelected;
                
                return (
                  <label 
                    key={column}
                    className={`flex items-center space-x-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                      isSelected 
                        ? 'border-highlight bg-highlight/10' 
                        : canSelect 
                          ? 'border-border hover:border-highlight/50' 
                          : 'border-border opacity-50 cursor-not-allowed'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => handleNumericToggle(column)}
                      disabled={disabled || !canSelect}
                      className="text-highlight focus:ring-highlight rounded"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-textPrimary truncate">{column}</div>
                      <div className="text-xs text-textSecondary">Sample: {sampleValue}</div>
                    </div>
                    <Hash className="h-4 w-4 text-green-400 flex-shrink-0" />
                  </label>
                );
              })}
            </div>
          </div>
        )}

        {/* Categorical Columns */}
        {categorical_columns.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center space-x-2">
              <Tag className="h-5 w-5 text-purple-400" />
              <h4 className="font-medium text-textPrimary">Categorical Columns</h4>
              <span className="text-sm text-textSecondary">
                ({selectedCategorical.length}/{maxCategorical} selected)
              </span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {categorical_columns.map(column => {
                const isSelected = selectedCategorical.includes(column);
                const sampleValue = categorical_samples[column] || 'N/A';
                const canSelect = selectedCategorical.length < maxCategorical || isSelected;
                
                return (
                  <label 
                    key={column}
                    className={`flex items-center space-x-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                      isSelected 
                        ? 'border-highlight bg-highlight/10' 
                        : canSelect 
                          ? 'border-border hover:border-highlight/50' 
                          : 'border-border opacity-50 cursor-not-allowed'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => handleCategoricalToggle(column)}
                      disabled={disabled || !canSelect}
                      className="text-highlight focus:ring-highlight rounded"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-textPrimary truncate">{column}</div>
                      <div className="text-xs text-textSecondary">Sample: {sampleValue}</div>
                    </div>
                    <Tag className="h-4 w-4 text-purple-400 flex-shrink-0" />
                  </label>
                );
              })}
            </div>
          </div>
        )}

        {/* No Eligible Columns */}
        {numeric_columns.length === 0 && categorical_columns.length === 0 && (
          <div className="text-center py-8">
            <AlertCircle className="h-8 w-8 mx-auto mb-4 text-yellow-400" />
            <h4 className="font-medium text-textPrimary mb-2">No Eligible Columns</h4>
            <p className="text-sm text-textSecondary">
              No columns meet the criteria for metadata selection. 
              Numeric columns and low-cardinality categorical columns are eligible.
            </p>
          </div>
        )}

        {/* Selection Summary */}
        {totalSelected > 0 && (
          <div className="p-4 bg-highlight/10 rounded-lg border border-highlight">
            <h4 className="font-medium text-textPrimary mb-2">Selected Columns:</h4>
            <div className="space-y-1">
              {selectedNumeric.map(col => (
                <div key={col} className="flex items-center space-x-2 text-sm">
                  <Hash className="h-3 w-3 text-green-400" />
                  <span className="text-textPrimary">{col}</span>
                  <span className="text-textSecondary">(numeric)</span>
                </div>
              ))}
              {selectedCategorical.map(col => (
                <div key={col} className="flex items-center space-x-2 text-sm">
                  <Tag className="h-3 w-3 text-purple-400" />
                  <span className="text-textPrimary">{col}</span>
                  <span className="text-textSecondary">(categorical)</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default MetadataColumnSelector;
