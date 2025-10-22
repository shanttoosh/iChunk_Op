import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle, Trash2 } from 'lucide-react';
import Button from '../UI/Button';
import useUIStore from '../../stores/uiStore';
import api from '../../services/api';

const DuplicateAnalysisCard = ({ onApply }) => {
  const [analysis, setAnalysis] = useState(null);
  const [removeDuplicates, setRemoveDuplicates] = useState(false);
  const [loading, setLoading] = useState(true);
  const [applied, setApplied] = useState(false);
  
  const { addNotification } = useUIStore();
  
  useEffect(() => {
    fetchAnalysis();
  }, []);
  
  const fetchAnalysis = async () => {
    try {
      const response = await api.get('/deep_config/analyze_duplicates');
      if (response.data && response.data.duplicate_analysis) {
        setAnalysis(response.data.duplicate_analysis);
      }
    } catch (error) {
      console.error('Error fetching duplicate analysis:', error);
      addNotification({ type: 'error', message: 'Failed to analyze duplicates' });
    } finally {
      setLoading(false);
    }
  };
  
  const handleApply = async () => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('strategy', removeDuplicates ? 'keep_first' : 'keep_all');
      const response = await api.post('/deep_config/duplicates', formData);
      
      setApplied(true);
      
      if (removeDuplicates) {
        addNotification({ 
          type: 'success', 
          message: `Removed ${response.data.duplicates_removed} duplicate rows` 
        });
      } else {
        addNotification({ type: 'info', message: 'Skipped duplicate removal' });
      }
      
      onApply(response.data);
    } catch (error) {
      console.error('Error applying duplicate removal:', error);
      addNotification({ type: 'error', message: 'Failed to process duplicates' });
    } finally {
      setLoading(false);
    }
  };
  
  if (loading) {
    return (
      <div className="p-6 bg-secondary rounded-lg text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-accent mx-auto mb-3"></div>
        <p className="text-textSecondary">Analyzing duplicates...</p>
      </div>
    );
  }
  
  if (!analysis) {
    return (
      <div className="p-6 bg-secondary rounded-lg text-center">
        <AlertCircle className="h-8 w-8 mx-auto mb-3 text-warning" />
        <h4 className="font-medium text-textPrimary mb-2">Analysis Failed</h4>
        <p className="text-sm text-textSecondary">Could not analyze duplicates</p>
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      {/* Analysis Summary */}
      <div className="p-6 bg-secondary rounded-lg">
        <h3 className="text-lg font-medium text-textPrimary mb-4">🔍 Duplicate Analysis</h3>
        <p className="text-sm text-textSecondary mb-4">
          Rows are considered duplicates only if ALL column values match exactly
        </p>
        
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-textPrimary">{analysis.total_rows}</div>
            <div className="text-sm text-textSecondary">Total Rows</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-textPrimary">{analysis.duplicate_pairs_count}</div>
            <div className="text-sm text-textSecondary">Duplicate Groups</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-textPrimary">{analysis.duplicate_percentage.toFixed(1)}%</div>
            <div className="text-sm text-textSecondary">Duplicate %</div>
          </div>
        </div>
        
        {analysis.has_duplicates ? (
          <div className="flex items-center p-3 bg-orange-100 dark:bg-orange-900/20 rounded-lg mb-4">
            <AlertCircle className="h-5 w-5 text-orange-500 mr-2" />
            <span className="text-orange-700 dark:text-orange-300 text-sm">
              Found {analysis.duplicate_pairs_count} groups of duplicate rows. These are rows where ALL column values are identical.
            </span>
          </div>
        ) : (
          <div className="flex items-center p-3 bg-green-100 dark:bg-green-900/20 rounded-lg mb-4">
            <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
            <span className="text-green-700 dark:text-green-300 text-sm">
              No duplicate rows found!
            </span>
          </div>
        )}
      </div>
      
      {/* Simple Toggle */}
      <div className="p-6 bg-secondary rounded-lg">
        <h3 className="text-lg font-medium text-textPrimary mb-4">🛠️ Duplicate Handling</h3>
        
        <div className="flex items-center space-x-3 mb-4">
          <input
            type="checkbox"
            id="removeDuplicates"
            checked={removeDuplicates}
            onChange={(e) => setRemoveDuplicates(e.target.checked)}
            className="w-4 h-4 text-accent bg-background border-border rounded focus:ring-accent focus:ring-2"
          />
          <label htmlFor="removeDuplicates" className="text-textPrimary">
            Remove duplicate rows (keep first occurrence)
          </label>
        </div>
        
        <p className="text-sm text-textSecondary mb-4">
          {removeDuplicates 
            ? "Duplicate rows will be removed, keeping only the first occurrence of each duplicate group."
            : "All rows will be kept, including duplicates."
          }
        </p>
        
        <Button
          onClick={handleApply}
          disabled={loading || applied}
          className="w-full"
        >
          {loading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Processing...
            </div>
          ) : applied ? (
            <div className="flex items-center justify-center">
              <CheckCircle className="h-4 w-4 mr-2" />
              {removeDuplicates ? 'Duplicates Removed' : 'Skipped'}
            </div>
          ) : (
            <div className="flex items-center justify-center">
              <Trash2 className="h-4 w-4 mr-2" />
              Apply Duplicate Handling
            </div>
          )}
        </Button>
      </div>
    </div>
  );
};

export default DuplicateAnalysisCard;