// src/components/UI/Button.jsx
import React from 'react';

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  disabled = false, 
  loading = false,
  className = '',
  onClick,
  type = 'button',
  ...props 
}) => {
  const baseClasses = 'font-medium rounded-card transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-highlight focus:ring-opacity-50';
  
  const variantClasses = {
    primary: 'primary-button',
    secondary: 'secondary-button',
    danger: 'bg-danger text-white hover:brightness-110',
    success: 'bg-success text-primary hover:brightness-110',
    outline: 'border border-border text-textPrimary hover:bg-secondary'
  };
  
  const sizeClasses = {
    xs: 'px-2 py-1 text-xs font-medium',
    sm: 'px-3 py-1.5 text-sm font-medium',
    md: 'px-4 py-2 text-sm font-medium',
    lg: 'px-5 py-2.5 text-base font-medium'
  };
  
  const disabledClasses = disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer';
  
  return (
    <button
      type={type}
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${disabledClasses} ${className}`}
      disabled={disabled || loading}
      onClick={onClick}
      {...props}
    >
      {loading ? (
        <div className="flex items-center justify-center space-x-2">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
          <span className="text-sm font-medium">Loading...</span>
        </div>
      ) : (
        children
      )}
    </button>
  );
};

export default Button;
