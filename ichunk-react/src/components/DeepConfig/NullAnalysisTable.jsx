import React, { useState, useEffect } from 'react';
import { AlertCircle } from 'lucide-react';
import Button from '../UI/Button';

const NullAnalysisTable = ({ nullProfile, columns, dataTypes, onApply }) => {
  const [strategies, setStrategies] = useState({});
  const [customValues, setCustomValues] = useState({});
  
  // Debug logging
  useEffect(() => {
    console.log('NullAnalysisTable received props:', {
      nullProfileLength: nullProfile?.length,
      columnsLength: columns?.length,
      dataTypesKeys: dataTypes ? Object.keys(dataTypes) : null,
      fullNullProfile: nullProfile,
      sampleColumns: columns?.slice(0, 3),
      sampleDataTypes: dataTypes ? Object.entries(dataTypes).slice(0, 3) : null
    });
  }, [nullProfile, columns, dataTypes]);
  
  // Add error handling for missing props
  if (!nullProfile || !columns || !dataTypes) {
    return (
      <div className="p-6 bg-secondary rounded-lg text-center">
        <div className="text-textSecondary mb-2">
          <AlertCircle className="h-8 w-8 mx-auto mb-3 text-warning" />
          <h4 className="font-medium text-textPrimary mb-2">Missing Data</h4>
          <p className="text-sm text-textSecondary">
            Required data (nullProfile, columns, dataTypes) is missing
          </p>
        </div>
      </div>
    );
  }
  
  // Use nullProfile directly - it already contains only columns with null values
  const nullAnalysis = nullProfile.map(item => ({
    col: item.column,
    dtype: item.dtype,
    nullCount: item.null_count,
    nullPct: item.null_pct
  })).sort((a, b) => b.nullPct - a.nullPct);
  
  // Get data-type aware strategy options
  const getStrategyOptions = (dtype) => {
    const baseOptions = [
      { value: 'no_change', label: 'No change' },
      { value: 'remove', label: 'Remove rows' },
      { value: 'custom', label: 'Custom value' }
    ];
    
    // Data-type specific options
    if (dtype === 'object' || dtype === 'string') {
      // Text/Category columns
      return [
        ...baseOptions,
        { value: 'mode', label: 'Fill with Mode' },
        { value: 'ffill', label: 'Forward Fill' },
        { value: 'bfill', label: 'Backward Fill' },
        { value: 'unknown', label: "Fill with 'Unknown'" }
      ];
    } else if (dtype === 'int64' || dtype === 'float64') {
      // Numeric columns
      return [
        ...baseOptions,
        { value: 'mean', label: 'Fill with Mean' },
        { value: 'median', label: 'Fill with Median' },
        { value: 'mode', label: 'Fill with Mode' },
        { value: 'ffill', label: 'Forward Fill' },
        { value: 'bfill', label: 'Backward Fill' }
      ];
    } else if (dtype === 'bool') {
      // Boolean columns
      return [
        { value: 'no_change', label: 'No change' },
        { value: 'mode', label: 'Fill with Mode' },
        { value: 'ffill', label: 'Forward Fill' },
        { value: 'bfill', label: 'Backward Fill' },
        { value: 'remove', label: 'Remove rows' },
        { value: 'custom', label: 'Custom value' }
      ];
    } else if (dtype === 'datetime64[ns]' || dtype === 'datetime') {
      // DateTime columns
      return [
        { value: 'no_change', label: 'No change' },
        { value: 'ffill', label: 'Forward Fill' },
        { value: 'bfill', label: 'Backward Fill' },
        { value: 'remove', label: 'Remove rows' },
        { value: 'custom', label: 'Custom value' }
      ];
    }
    
    // Default fallback
    return baseOptions;
  };
  
  // Format data type for display
  const formatDataType = (dtype) => {
    const typeMap = {
      'object': 'Object',
      'string': 'String',
      'int64': 'Integer',
      'float64': 'Float',
      'bool': 'Boolean',
      'datetime64[ns]': 'DateTime',
      'datetime': 'DateTime'
    };
    return typeMap[dtype] || dtype;
  };

  // Get color class based on null percentage
  const getNullColor = (nullPct) => {
    if (nullPct >= 50) return 'text-red-400 font-semibold';
    if (nullPct >= 25) return 'text-orange-400 font-medium';
    if (nullPct >= 10) return 'text-yellow-400';
    return 'text-textSecondary';
  };
  
  const handleStrategyChange = (col, strategy) => {
    setStrategies({...strategies, [col]: strategy});
  };
  
  const handleCustomValueChange = (col, value) => {
    setCustomValues({...customValues, [col]: value});
  };
  
  const handleApply = () => {
    const finalStrategies = {};
    
    // Debug logging
    console.log('NullAnalysisTable handleApply called');
    console.log('Strategies state:', strategies);
    console.log('Custom values state:', customValues);
    
    Object.entries(strategies).forEach(([col, strategy]) => {
      if (strategy === 'custom') {
        finalStrategies[col] = `custom_value:${customValues[col] || ''}`;
      } else if (strategy !== 'no_change') {
        finalStrategies[col] = strategy;
      }
    });
    
    console.log('Final strategies being sent:', finalStrategies);
    onApply(finalStrategies);
  };
  
  const hasStrategies = Object.keys(strategies).some(col => strategies[col] !== 'no_change');
  
  if (nullAnalysis.length === 0) {
    return (
      <div className="custom-card border-l-4 border-success text-center py-8">
        <div className="text-success text-5xl mb-3">✓</div>
        <div className="font-semibold text-lg mb-2">No Null Values Found!</div>
        <div className="text-textSecondary text-sm mb-4">
          All columns are complete - no null handling needed
        </div>
        <Button
          variant="primary"
          size="sm"
          onClick={() => onApply({})}
          className="flex items-center space-x-1"
        >
          <span>Continue to Next Step</span>
        </Button>
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold text-textPrimary mb-2">📊 Null Analysis</h3>
        <p className="text-sm text-textSecondary mb-3">
          Found {nullAnalysis.length} column(s) with null values (sorted by null percentage)
        </p>
      </div>
      
      <div className="bg-primary border border-border rounded-card overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-secondary">
            <tr>
              <th className="px-3 py-2 text-left font-semibold">Column Name</th>
              <th className="px-3 py-2 text-left font-semibold">Data Type</th>
              <th className="px-3 py-2 text-right font-semibold">Null Count</th>
              <th className="px-3 py-2 text-right font-semibold">Null %</th>
            </tr>
          </thead>
          <tbody>
            {nullAnalysis.map(({col, dtype, nullCount, nullPct}) => (
              <tr key={col} className="border-b border-border hover:bg-secondary transition">
                <td className="px-3 py-2 font-medium text-textPrimary">{col}</td>
                <td className="px-3 py-2 text-textSecondary">{formatDataType(dtype)}</td>
                <td className="px-3 py-2 text-right">
                  <span className={getNullColor(nullPct)}>
                    {nullCount}
                  </span>
                </td>
                <td className="px-3 py-2 text-right">
                  <span className={getNullColor(nullPct)}>
                    {nullPct.toFixed(1)}%
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div>
        <h3 className="text-lg font-semibold text-textPrimary mb-3">🛠️ Null Handling Strategies</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {nullAnalysis.map(({col, dtype}) => (
            <div key={col} className="custom-card">
              <label className="font-semibold text-sm text-textPrimary">{col}</label>
              <p className="text-xs text-textSecondary mb-2">{formatDataType(dtype)}</p>
              <select
                value={strategies[col] || 'no_change'}
                onChange={e => handleStrategyChange(col, e.target.value)}
                className="w-full bg-primary border border-border rounded px-2 py-1.5 text-sm text-textPrimary focus:border-highlight focus:outline-none transition"
              >
                {getStrategyOptions(dtype).map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              {strategies[col] === 'custom' && (
                <input
                  type="text"
                  placeholder="Enter custom value..."
                  value={customValues[col] || ''}
                  onChange={e => handleCustomValueChange(col, e.target.value)}
                  className="mt-2 w-full bg-primary border border-border rounded px-3 py-2 text-sm text-textPrimary focus:border-highlight focus:outline-none transition"
                />
              )}
            </div>
          ))}
        </div>
      </div>
      
      <div className="flex justify-between items-center pt-4">
        <div className="text-sm text-textSecondary">
          {hasStrategies ? (
            <span className="text-highlight">
              {Object.keys(strategies).filter(col => strategies[col] !== 'no_change').length} column(s) selected for null handling
            </span>
          ) : (
            'No null handling strategies selected'
          )}
        </div>
        <Button
          variant="primary"
          size="sm"
          onClick={handleApply}
          disabled={!hasStrategies}
          className="flex items-center space-x-1"
        >
          <span>Apply Null Handling</span>
        </Button>
      </div>
    </div>
  );
};

export default NullAnalysisTable;
