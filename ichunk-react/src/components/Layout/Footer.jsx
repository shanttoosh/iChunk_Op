// src/components/Layout/Footer.jsx
import React from 'react';

const Footer = () => {
  return (
    <footer className="mt-12 pt-8 border-t border-border bg-primaryDark w-full">
      <div className="text-center text-textSecondary text-sm">
        <div className="flex items-center justify-center space-x-2">
          <span>© 2025 iChunk Optimizer</span>
          <span>|</span>
          <span>Powered by</span>
          <img 
            src="/elevate-logo.svg" 
            alt="Elevate" 
            className="h-4 w-auto opacity-70"
            onError={(e) => {
              // Fallback to text if logo not found
              e.target.style.display = 'none';
              e.target.nextSibling.style.display = 'block';
            }}
          />
          <div className="hidden h-4 flex items-center">
            <span className="text-highlight font-bold text-xs">ELEVATE</span>
          </div>
        </div>
        <p className="mt-2 text-xs text-textTertiary">
          Empowering enterprises with intelligent data processing and AI-driven insights
        </p>
      </div>
    </footer>
  );
};

export default Footer;
