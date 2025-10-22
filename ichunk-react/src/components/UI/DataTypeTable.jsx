// src/components/UI/DataTypeTable.jsx
import React from 'react';

const DataTypeTable = ({ columnInfo, title = "Data Types Overview" }) => {
  if (!columnInfo || !columnInfo.columnNames || !columnInfo.dataTypes) {
    return (
      <div className="p-4 bg-secondary rounded-lg">
        <h4 className="font-medium text-textPrimary mb-3">{title}</h4>
        <div className="text-textSecondary text-sm">No column information available</div>
      </div>
    );
  }

  const { columnNames, dataTypes, sampleValues } = columnInfo;

  return (
    <div className="p-4 bg-secondary rounded-lg">
      <h4 className="font-medium text-textPrimary mb-3">{title}</h4>
      <div className="overflow-auto max-h-64 border border-border rounded">
        <table className="w-full text-sm">
          <thead className="bg-primary sticky top-0">
            <tr>
              <th className="text-left p-2 text-textPrimary font-medium border-b border-border">Column</th>
              <th className="text-left p-2 text-textPrimary font-medium border-b border-border">Data Type</th>
              <th className="text-left p-2 text-textPrimary font-medium border-b border-border">Sample Value</th>
            </tr>
          </thead>
          <tbody>
            {columnNames.map((col, index) => (
              <tr key={col} className="hover:bg-primary/50 transition-colors">
                <td className="p-2 text-textPrimary border-b border-border font-medium">{col}</td>
                <td className="p-2 text-textSecondary border-b border-border">
                  <code className="text-highlight">{dataTypes[col] || 'unknown'}</code>
                </td>
                <td className="p-2 text-textSecondary border-b border-border">
                  {sampleValues && sampleValues[col] ? String(sampleValues[col]) : 'N/A'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-xs text-textTertiary mt-2">
        {columnNames.length} columns total
      </div>
    </div>
  );
};

export default DataTypeTable;
