import React from 'react';

const DatasetPreviewTable = ({ data, columns, totalRows }) => {
  const visibleRows = 3;
  const visibleCols = 5;
  
  if (!data || !columns || data.length === 0) {
    return (
      <div className="bg-primary border border-border rounded-card overflow-hidden mt-4">
        <div className="p-3 bg-secondary border-b border-border">
          <h4 className="font-semibold">📊 Dataset Preview</h4>
          <p className="text-sm text-textSecondary">No data available</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="bg-primary border border-border rounded-card overflow-hidden mt-4">
      <div className="p-3 bg-secondary border-b border-border">
        <h4 className="font-semibold">📊 Dataset Preview</h4>
        <p className="text-sm text-textSecondary">
          Showing {Math.min(data.length, 100)} of {totalRows} rows × {columns.length} columns
        </p>
      </div>
      
      <div className="max-h-[400px] overflow-auto scrollbar-custom">
        <table className="w-full table-auto text-sm border-collapse">
          <thead className="bg-secondary sticky top-0 z-10">
            <tr>
              <th className="px-3 py-2 text-left font-semibold border-b border-border w-12">#</th>
              {columns.map((col, idx) => (
                <th key={col} className="px-3 py-2 text-left font-semibold border-b border-border min-w-[150px]">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((row, rowIdx) => (
              <tr key={rowIdx} className="hover:bg-secondary transition">
                <td className="px-3 py-2 border-b border-border text-textSecondary font-mono text-xs">
                  {rowIdx + 1}
                </td>
                {columns.map(col => (
                  <td key={col} className="px-3 py-2 border-b border-border max-w-[200px] truncate" title={row[col]}>
                    {row[col] != null ? String(row[col]) : <span className="text-textTertiary italic">null</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      <div className="p-2 bg-secondary border-t border-border text-xs text-textSecondary text-center">
        💡 Scroll horizontally and vertically to view all data
      </div>
    </div>
  );
};

export default DatasetPreviewTable;


