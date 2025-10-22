// src/components/UI/NullAnalysisTable.jsx
import React from 'react';

const NullAnalysisTable = ({ nullProfile, title = "Null Analysis" }) => {
  if (!nullProfile || !Array.isArray(nullProfile) || nullProfile.length === 0) {
    return (
      <div className="p-4 bg-success/20 rounded-lg">
        <h4 className="font-medium text-success mb-3">{title}</h4>
        <div className="flex items-center space-x-2">
          <span className="text-success">✅ No null values detected!</span>
        </div>
        <p className="text-textSecondary text-sm mt-1">You can proceed to the next step.</p>
      </div>
    );
  }

  return (
    <div className="p-4 bg-secondary rounded-lg">
      <h4 className="font-medium text-textPrimary mb-3">{title}</h4>
      <div className="overflow-auto max-h-64 border border-border rounded">
        <table className="w-full text-sm">
          <thead className="bg-primary sticky top-0">
            <tr>
              <th className="text-left p-2 text-textPrimary font-medium border-b border-border">Column</th>
              <th className="text-left p-2 text-textPrimary font-medium border-b border-border">Data Type</th>
              <th className="text-left p-2 text-textPrimary font-medium border-b border-border">Null Count</th>
              <th className="text-left p-2 text-textPrimary font-medium border-b border-border">Null %</th>
            </tr>
          </thead>
          <tbody>
            {nullProfile.map((col, index) => (
              <tr key={col.column} className="hover:bg-primary/50 transition-colors">
                <td className="p-2 text-textPrimary border-b border-border font-medium">{col.column}</td>
                <td className="p-2 text-textSecondary border-b border-border">{col.dtype}</td>
                <td className="p-2 text-textSecondary border-b border-border">{col.null_count}</td>
                <td className="p-2 text-textSecondary border-b border-border">
                  {col.null_pct ? `${col.null_pct.toFixed(1)}%` : '0.0%'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-xs text-textTertiary mt-2">
        {nullProfile.length} columns with null values
      </div>
    </div>
  );
};

export default NullAnalysisTable;
