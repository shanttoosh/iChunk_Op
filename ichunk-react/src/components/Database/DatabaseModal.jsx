// src/components/Database/DatabaseModal.jsx
import React, { useState } from 'react';
import { TestTube, List, CheckCircle, XCircle } from 'lucide-react';
import Modal from '../UI/Modal';
import Button from '../UI/Button';
import Input from '../UI/Input';
import Select from '../UI/Select';
import databaseService from '../../services/database.service';
import { DATABASE_TYPES } from '../../utils/constants';

const DatabaseModal = ({ isOpen, onClose, onImport }) => {
  const [dbConfig, setDbConfig] = useState({
    dbType: 'mysql',
    host: 'localhost',
    port: 3306,
    username: '',
    password: '',
    database: ''
  });
  
  const [tables, setTables] = useState([]);
  const [selectedTable, setSelectedTable] = useState('');
  const [connectionStatus, setConnectionStatus] = useState(null);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [isLoadingTables, setIsLoadingTables] = useState(false);

  const handleInputChange = (field, value) => {
    setDbConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const testConnection = async () => {
    setIsTestingConnection(true);
    setConnectionStatus(null);
    
    try {
      const result = await databaseService.testConnection(dbConfig);
      setConnectionStatus(result.status === 'success' ? 'success' : 'error');
    } catch (error) {
      setConnectionStatus('error');
    } finally {
      setIsTestingConnection(false);
    }
  };

  const loadTables = async () => {
    if (connectionStatus !== 'success') {
      alert('Please test connection first');
      return;
    }

    setIsLoadingTables(true);
    
    try {
      const result = await databaseService.listTables(dbConfig);
      setTables(result.tables || []);
    } catch (error) {
      console.error('Error loading tables:', error);
    } finally {
      setIsLoadingTables(false);
    }
  };

  const handleImport = () => {
    if (!selectedTable) {
      alert('Please select a table');
      return;
    }

    onImport({
      ...dbConfig,
      tableName: selectedTable
    });
    onClose();
  };

  const databaseTypeOptions = [
    { value: 'mysql', label: 'MySQL' },
    { value: 'postgresql', label: 'PostgreSQL' }
  ];

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Database Import Configuration"
      size="lg"
    >
      <div className="space-y-6">
        {/* Database Type */}
        <Select
          label="Database Type"
          value={dbConfig.dbType}
          onChange={(e) => handleInputChange('dbType', e.target.value)}
          options={databaseTypeOptions}
        />

        {/* Connection Details */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Input
            label="Host"
            value={dbConfig.host}
            onChange={(e) => handleInputChange('host', e.target.value)}
            placeholder="localhost"
          />
          
          <Input
            label="Port"
            type="number"
            value={dbConfig.port}
            onChange={(e) => handleInputChange('port', parseInt(e.target.value))}
            placeholder={dbConfig.dbType === 'mysql' ? '3306' : '5432'}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Input
            label="Username"
            value={dbConfig.username}
            onChange={(e) => handleInputChange('username', e.target.value)}
            placeholder="Enter username"
          />
          
          <Input
            label="Password"
            type="password"
            value={dbConfig.password}
            onChange={(e) => handleInputChange('password', e.target.value)}
            placeholder="Enter password"
          />
        </div>

        <Input
          label="Database Name"
          value={dbConfig.database}
          onChange={(e) => handleInputChange('database', e.target.value)}
          placeholder="Enter database name"
        />

        {/* Connection Test */}
        <div className="flex items-center space-x-4">
          <Button
            variant="outline"
            size="sm"
            onClick={testConnection}
            disabled={isTestingConnection}
            loading={isTestingConnection}
            className="flex items-center space-x-1"
          >
            <TestTube className="h-3 w-3" />
            <span>Test</span>
          </Button>
          
          {connectionStatus && (
            <div className="flex items-center space-x-2">
              {connectionStatus === 'success' ? (
                <>
                  <CheckCircle className="h-5 w-5 text-success" />
                  <span className="text-success text-sm">Connection successful</span>
                </>
              ) : (
                <>
                  <XCircle className="h-5 w-5 text-danger" />
                  <span className="text-danger text-sm">Connection failed</span>
                </>
              )}
            </div>
          )}
        </div>

        {/* Table Selection */}
        {connectionStatus === 'success' && (
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <Button
                variant="outline"
                size="sm"
                onClick={loadTables}
                disabled={isLoadingTables}
                loading={isLoadingTables}
                className="flex items-center space-x-1"
              >
                <List className="h-3 w-3" />
                <span>Load</span>
              </Button>
            </div>

            {tables.length > 0 && (
              <Select
                label="Select Table"
                value={selectedTable}
                onChange={(e) => setSelectedTable(e.target.value)}
                options={tables.map(table => ({ value: table, label: table }))}
                placeholder="Choose a table to import"
              />
            )}
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-end space-x-3 pt-4 border-t border-border">
          <Button
            variant="outline"
            size="sm"
            onClick={onClose}
            className="flex items-center space-x-1"
          >
            <span>Cancel</span>
          </Button>
          
          <Button
            variant="primary"
            size="sm"
            onClick={handleImport}
            disabled={!selectedTable || connectionStatus !== 'success'}
            className="flex items-center space-x-1"
          >
            <span>Import</span>
          </Button>
        </div>
      </div>
    </Modal>
  );
};

export default DatabaseModal;
