// src/components/Export/ExportSection.jsx
import React from 'react';
import { Download, FileText, Database, Archive } from 'lucide-react';
import useAppStore from '../../stores/appStore';
import Card from '../UI/Card';
import Button from '../UI/Button';
import exportService from '../../services/export.service';

const ExportSection = () => {
  const { currentMode, apiResults } = useAppStore();

  const handleExport = async (exportType) => {
    try {
      switch (currentMode) {
        case 'fast':
        case 'config1':
          switch (exportType) {
            case 'preprocessed':
              await exportService.exportPreprocessed();
              break;
            case 'chunks':
              await exportService.exportChunks();
              break;
            case 'embeddings':
              await exportService.exportEmbeddings();
              break;
          }
          break;
        
        case 'deep':
          switch (exportType) {
            case 'preprocessed':
              await exportService.exportDeepPreprocessed();
              break;
            case 'chunks':
              await exportService.exportDeepChunks();
              break;
            case 'embeddings':
              await exportService.exportDeepEmbeddings();
              break;
          }
          break;
        
        case 'campaign':
          switch (exportType) {
            case 'preprocessed':
              await exportService.exportCampaignPreprocessed();
              break;
            case 'chunks':
              await exportService.exportCampaignChunks();
              break;
            case 'embeddings':
              await exportService.exportCampaignEmbeddings();
              break;
          }
          break;
        
        default:
          alert('No processing mode selected');
      }
    } catch (error) {
      console.error('Export error:', error);
      alert('Export failed. Please try again.');
    }
  };

  const getExportInfo = () => {
    switch (currentMode) {
      case 'fast':
        return {
          title: 'Fast Mode Exports',
          description: 'Download processed data from Fast Mode',
          files: [
            { type: 'preprocessed', name: 'Preprocessed Data', format: 'CSV', icon: FileText },
            { type: 'chunks', name: 'Chunks', format: 'CSV', icon: Archive },
            { type: 'embeddings', name: 'Embeddings', format: 'JSON', icon: Database }
          ]
        };
      
      case 'config1':
        return {
          title: 'Config-1 Mode Exports',
          description: 'Download processed data from Config-1 Mode',
          files: [
            { type: 'preprocessed', name: 'Preprocessed Data', format: 'CSV', icon: FileText },
            { type: 'chunks', name: 'Chunks', format: 'CSV', icon: Archive },
            { type: 'embeddings', name: 'Embeddings', format: 'JSON', icon: Database }
          ]
        };
      
      case 'deep':
        return {
          title: 'Deep Config Mode Exports',
          description: 'Download processed data with metadata from Deep Config Mode',
          files: [
            { type: 'preprocessed', name: 'Preprocessed Data', format: 'CSV', icon: FileText },
            { type: 'chunks', name: 'Chunks (with Metadata)', format: 'CSV', icon: Archive },
            { type: 'embeddings', name: 'Embeddings (with Metadata)', format: 'JSON', icon: Database }
          ]
        };
      
      case 'campaign':
        return {
          title: 'Campaign Mode Exports',
          description: 'Download processed campaign data',
          files: [
            { type: 'preprocessed', name: 'Preprocessed Text', format: 'TXT', icon: FileText },
            { type: 'chunks', name: 'Chunks', format: 'CSV', icon: Archive },
            { type: 'embeddings', name: 'Embeddings', format: 'JSON', icon: Database }
          ]
        };
      
      default:
        return {
          title: 'Export Data',
          description: 'Select a processing mode to enable exports',
          files: []
        };
    }
  };

  const exportInfo = getExportInfo();

  if (!currentMode) {
    return (
      <Card>
        <div className="text-center py-8 text-textSecondary">
          <Download className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <h3 className="text-lg font-medium text-textPrimary mb-2">Export Data</h3>
          <p>Select a processing mode to enable data exports.</p>
        </div>
      </Card>
    );
  }

  return (
    <Card>
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-8 h-8 bg-highlight rounded flex items-center justify-center">
          <Download className="h-5 w-5 text-white" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-textPrimary">{exportInfo.title}</h2>
          <p className="text-textSecondary text-sm">{exportInfo.description}</p>
        </div>
      </div>

      {!apiResults ? (
        <div className="text-center py-8 text-textSecondary">
          <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No processed data available for export.</p>
          <p className="text-sm">Run a processing mode first to generate exportable data.</p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {exportInfo.files.map((file) => {
              const Icon = file.icon;
              return (
                <div key={file.type} className="p-4 bg-secondary rounded-lg border border-border">
                  <div className="flex items-center space-x-3 mb-3">
                    <Icon className="h-6 w-6 text-highlight" />
                    <div>
                      <h4 className="font-medium text-textPrimary">{file.name}</h4>
                      <p className="text-sm text-textSecondary">{file.format} format</p>
                    </div>
                  </div>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full flex items-center justify-center space-x-2"
                    onClick={() => handleExport(file.type)}
                  >
                    <Download className="h-4 w-4" />
                    <span className="text-sm font-medium">Download</span>
                  </Button>
                </div>
              );
            })}
          </div>

          <div className="mt-6 p-4 bg-secondary rounded-lg">
            <h4 className="font-medium text-textPrimary mb-2">Export Information</h4>
            <div className="text-sm text-textSecondary space-y-1">
              <p><strong>Processing Mode:</strong> {currentMode.charAt(0).toUpperCase() + currentMode.slice(1)} Mode</p>
              <p><strong>Rows Processed:</strong> {apiResults.rows || 'N/A'}</p>
              <p><strong>Chunks Generated:</strong> {apiResults.chunks || 'N/A'}</p>
              <p><strong>Storage Status:</strong> {apiResults.stored || 'N/A'}</p>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
};

export default ExportSection;