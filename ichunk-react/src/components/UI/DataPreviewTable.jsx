// src/components/UI/DataPreviewTable.jsx
import React from 'react';

const DataPreviewTable = ({ 
  data, 
  columns, 
  title, 
  maxRows = 5, 
  maxCols = 3,
  className = "" 
}) => {
  if (!data || !Array.isArray(data) || data.length === 0) {
    return (
      <div className={`p-4 bg-secondary rounded-lg ${className}`}>
        <h4 className="font-medium text-textPrimary mb-3">{title}</h4>
        <div className="text-textSecondary text-sm">No data available</div>
      </div>
    );
  }

  // Get column names from first row or provided columns
  const tableColumns = columns || Object.keys(data[0] || {});
  const displayColumns = tableColumns.slice(0, maxCols);
  const displayData = data.slice(0, maxRows);

  return (
    <div className={`p-4 bg-secondary rounded-lg ${className}`}>
      <h4 className="font-medium text-textPrimary mb-3">{title}</h4>
      <div className="overflow-auto max-h-64 border border-border rounded">
        <table className="w-full text-sm">
          <thead className="bg-primary sticky top-0">
            <tr>
              {displayColumns.map((col, index) => (
                <th key={index} className="text-left p-2 text-textPrimary font-medium border-b border-border">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayData.map((row, rowIndex) => (
              <tr key={rowIndex} className="hover:bg-primary/50 transition-colors">
                {displayColumns.map((col, colIndex) => (
                  <td key={colIndex} className="p-2 text-textSecondary border-b border-border">
                    {row[col] !== null && row[col] !== undefined ? String(row[col]) : 'N/A'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.length > maxRows && (
        <div className="text-xs text-textTertiary mt-2">
          Showing {maxRows} of {data.length} rows
        </div>
      )}
      {tableColumns.length > maxCols && (
        <div className="text-xs text-textTertiary mt-1">
          Showing {maxCols} of {tableColumns.length} columns
        </div>
      )}
    </div>
  );
};

export default DataPreviewTable;
