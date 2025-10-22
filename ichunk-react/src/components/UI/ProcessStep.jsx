// src/components/UI/ProcessStep.jsx
import React from 'react';

const ProcessStep = ({ 
  name, 
  status, 
  timing = null,
  className = '' 
}) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'completed':
        return '✅';
      case 'running':
        return '🟡';
      case 'error':
        return '❌';
      default:
        return '⚪';
    }
  };

  const getStatusClass = () => {
    switch (status) {
      case 'completed':
        return 'border-l-success';
      case 'running':
        return 'border-l-highlight';
      case 'error':
        return 'border-l-danger';
      default:
        return 'border-l-secondary';
    }
  };

  return (
    <div className={`process-step ${getStatusClass()} ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <span className="text-lg">{getStatusIcon()}</span>
          <span className="font-medium text-textPrimary">{name}</span>
        </div>
        {timing && (
          <span className="text-textSecondary text-sm">{timing}</span>
        )}
      </div>
    </div>
  );
};

export default ProcessStep;


