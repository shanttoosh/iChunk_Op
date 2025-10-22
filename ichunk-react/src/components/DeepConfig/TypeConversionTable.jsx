import React, { useState } from 'react';
import Select from '../UI/Select';
import Button from '../UI/Button';

const TypeConversionTable = ({ columns, dataTypes, sampleValues, onConvert }) => {
  const [conversions, setConversions] = useState({});
  
  // Group columns by current data type
  const groupedColumns = {};
  columns.forEach(col => {
    const dtype = dataTypes[col];
    if (!groupedColumns[dtype]) groupedColumns[dtype] = [];
    groupedColumns[dtype].push(col);
  });
  
  const formatDataType = (dtype) => {
    const typeMap = {
      'object': 'Text',
      'int64': 'Integer',
      'float64': 'Float',
      'datetime64[ns]': 'DateTime',
      'bool': 'Boolean'
    };
    return typeMap[dtype] || dtype;
  };
  
  const formatSampleValue = (value) => {
    if (value == null) return '-';
    const str = String(value);
    return str.length > 30 ? str.substring(0, 30) + '...' : str;
  };
  
  const handleConversionChange = (col, newType) => {
    if (newType === 'no_change') {
      const newConversions = { ...conversions };
      delete newConversions[col];
      setConversions(newConversions);
    } else {
      setConversions({ ...conversions, [col]: newType });
    }
  };
  
  const handleApply = () => {
    const finalConversions = {};
    Object.entries(conversions).forEach(([col, targetType]) => {
      if (targetType !== 'no_change') {
        finalConversions[col] = targetType;
      }
    });
    onConvert(finalConversions);
  };
  
  const hasConversions = Object.keys(conversions).length > 0;
  
  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-textPrimary mb-2">📊 Current Data Types Overview</h3>
        <p className="text-sm text-textSecondary">Select target types for columns you want to convert</p>
      </div>
      
      {Object.entries(groupedColumns).map(([dtype, cols]) => (
        <div key={dtype} className="custom-card">
          <div className="bg-secondaryDark text-highlight font-bold py-2 px-3 -m-3 mb-3 rounded-t-lg">
            {formatDataType(dtype)} ({cols.length} columns)
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-secondary">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold">Column Name</th>
                  <th className="px-3 py-2 text-left font-semibold">Current Type</th>
                  <th className="px-3 py-2 text-left font-semibold">Sample Value</th>
                  <th className="px-3 py-2 text-left font-semibold">Convert To</th>
                </tr>
              </thead>
              <tbody>
                {cols.map(col => (
                  <tr key={col} className={`border-b border-border hover:bg-secondary transition ${
                    conversions[col] ? 'bg-secondaryDark border-l-4 border-highlight' : ''
                  }`}>
                    <td className="px-3 py-2 font-medium text-textPrimary">{col}</td>
                    <td className="px-3 py-2 text-textSecondary">{formatDataType(dtype)}</td>
                    <td className="px-3 py-2 text-textTertiary italic text-xs">
                      {formatSampleValue(sampleValues[col])}
                    </td>
                    <td className="px-3 py-2">
                      <select
                        value={conversions[col] || 'no_change'}
                        onChange={e => handleConversionChange(col, e.target.value)}
                        className="w-full bg-primary border border-border rounded px-2 py-1 text-sm text-textPrimary focus:border-highlight focus:outline-none transition"
                      >
                        <option value="no_change">No change</option>
                        <option value="object">Text (object)</option>
                        <option value="int64">Integer (int64)</option>
                        <option value="float64">Float (float64)</option>
                        <option value="datetime">DateTime</option>
                        <option value="boolean">Boolean</option>
                      </select>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
      
      <div className="flex justify-between items-center pt-4">
        <div className="text-sm text-textSecondary">
          {hasConversions ? (
            <span className="text-highlight">
              {Object.keys(conversions).length} column(s) selected for conversion
            </span>
          ) : (
            'No conversions selected'
          )}
        </div>
        <Button
          variant="primary"
          size="sm"
          onClick={handleApply}
          disabled={!hasConversions}
          className="flex items-center space-x-1"
        >
          <span>Apply Type Conversions</span>
        </Button>
      </div>
    </div>
  );
};

export default TypeConversionTable;


