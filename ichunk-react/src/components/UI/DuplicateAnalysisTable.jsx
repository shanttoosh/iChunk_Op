// src/components/UI/DuplicateAnalysisTable.jsx
import React from 'react';

const DuplicateAnalysisTable = ({ duplicateAnalysis, title = "Duplicate Analysis" }) => {
  if (!duplicateAnalysis) {
    return (
      <div className="p-4 bg-secondary rounded-lg">
        <h4 className="font-medium text-textPrimary mb-3">{title}</h4>
        <div className="text-textSecondary text-sm">No duplicate analysis available</div>
      </div>
    );
  }

  const { 
    total_rows, 
    duplicate_pairs_count, 
    duplicate_percentage, 
    duplicate_groups = [],
    has_duplicates 
  } = duplicateAnalysis;

  return (
    <div className="p-4 bg-secondary rounded-lg">
      <h4 className="font-medium text-textPrimary mb-3">{title}</h4>
      
      {/* Summary Metrics */}
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="p-3 bg-primary rounded border-l-4 border-blue-400">
          <div className="text-sm text-textSecondary">Total Rows</div>
          <div className="text-lg font-semibold text-textPrimary">{total_rows}</div>
        </div>
        <div className="p-3 bg-primary rounded border-l-4 border-yellow-400">
          <div className="text-sm text-textSecondary">Duplicate Pairs</div>
          <div className="text-lg font-semibold text-textPrimary">{duplicate_pairs_count}</div>
        </div>
        <div className="p-3 bg-primary rounded border-l-4 border-red-400">
          <div className="text-sm text-textSecondary">Duplicate %</div>
          <div className="text-lg font-semibold text-textPrimary">{duplicate_percentage?.toFixed(1)}%</div>
        </div>
      </div>

      {has_duplicates ? (
        <>
          <div className="mb-3 p-2 bg-warning/20 border border-warning/30 rounded">
            <span className="text-warning font-medium">
              ⚠️ Found {duplicate_pairs_count} duplicate pairs ({duplicate_percentage?.toFixed(1)}% of data)
            </span>
          </div>

          {/* Duplicate Groups Preview */}
          {duplicate_groups.length > 0 && (
            <div className="mt-4">
              <h5 className="font-medium text-textPrimary mb-2">Duplicate Groups Preview:</h5>
              <div className="overflow-auto max-h-48 border border-border rounded">
                <table className="w-full text-sm">
                  <thead className="bg-primary sticky top-0">
                    <tr>
                      <th className="text-left p-2 text-textPrimary font-medium border-b border-border">Group</th>
                      <th className="text-left p-2 text-textPrimary font-medium border-b border-border">Count</th>
                      <th className="text-left p-2 text-textPrimary font-medium border-b border-border">Sample Values</th>
                    </tr>
                  </thead>
                  <tbody>
                    {duplicate_groups.slice(0, 5).map((group, index) => (
                      <tr key={index} className="hover:bg-primary/50 transition-colors">
                        <td className="p-2 text-textPrimary border-b border-border font-medium">
                          Group {index + 1}
                        </td>
                        <td className="p-2 text-textSecondary border-b border-border">
                          {group.count} rows
                        </td>
                        <td className="p-2 text-textSecondary border-b border-border">
                          <div className="max-w-xs truncate">
                            {Object.entries(group.values || {})
                              .slice(0, 2)
                              .map(([key, value]) => `${key}: ${value}`)
                              .join(', ')}
                            {Object.keys(group.values || {}).length > 2 && '...'}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {duplicate_groups.length > 5 && (
                <div className="text-xs text-textTertiary mt-2">
                  Showing 5 of {duplicate_groups.length} duplicate groups
                </div>
              )}
            </div>
          )}
        </>
      ) : (
        <div className="p-3 bg-success/20 border border-success/30 rounded">
          <span className="text-success font-medium">✅ No duplicates found!</span>
        </div>
      )}
    </div>
  );
};

export default DuplicateAnalysisTable;
