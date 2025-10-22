// src/components/UI/Card.jsx
import React from 'react';

const Card = ({ children, className = '', ...props }) => {
  return (
    <div 
      className={`custom-card ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

export default Card;


