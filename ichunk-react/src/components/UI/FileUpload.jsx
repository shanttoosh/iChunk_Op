// src/components/UI/FileUpload.jsx
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText } from 'lucide-react';
import { formatFileSize } from '../../utils/formatting';
import { validateCSVFile } from '../../utils/validation';

const FileUpload = ({ 
  onFileSelect, 
  file, 
  disabled = false,
  className = '' 
}) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const validation = validateCSVFile(file);
      
      if (validation.isValid) {
        onFileSelect(file);
      } else {
        // Handle validation errors
        console.error('File validation errors:', validation.errors);
      }
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    maxSize: 3 * 1024 * 1024 * 1024, // 3GB
    multiple: false,
    disabled
  });

  return (
    <div className={`${className}`}>
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-card p-8 text-center cursor-pointer transition-all duration-200
          ${isDragActive ? 'border-highlight bg-secondary' : 'border-border hover:border-highlight'}
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        {file ? (
          <div className="flex items-center justify-center space-x-3">
            <FileText className="h-8 w-8 text-highlight" />
            <div className="text-left">
              <p className="text-textPrimary font-medium">{file.name}</p>
              <p className="text-textSecondary text-sm">{formatFileSize(file.size)}</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center space-y-3">
            <Upload className="h-12 w-12 text-textSecondary" />
            <div>
              <p className="text-textPrimary font-medium">
                {isDragActive ? 'Drop the CSV file here' : 'Drag & drop CSV file here'}
              </p>
              <p className="text-textSecondary text-sm">
                or click to browse (max 3GB)
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;


