// src/components/UI/Input.jsx
import React from 'react';

const Input = ({ 
  label,
  type = 'text',
  placeholder = '',
  value,
  onChange,
  error,
  disabled = false,
  className = '',
  ...props 
}) => {
  return (
    <div className={`mb-4 ${className}`}>
      {label && (
        <label className="block text-textSecondary text-sm font-medium mb-2">
          {label}
        </label>
      )}
      <input
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        disabled={disabled}
        className={`input-field w-full ${error ? 'border-danger' : ''}`}
        {...props}
      />
      {error && (
        <p className="text-danger text-sm mt-1">{error}</p>
      )}
    </div>
  );
};

export default Input;


