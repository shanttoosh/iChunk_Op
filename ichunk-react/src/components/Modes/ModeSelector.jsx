// src/components/Modes/ModeSelector.jsx
import React from 'react';
import { Zap, Settings, Microscope, Target } from 'lucide-react';
import useAppStore from '../../stores/appStore';
import Button from '../UI/Button';

const ModeSelector = () => {
  const { currentMode, setCurrentMode } = useAppStore();

  const modes = [
    {
      id: 'fast',
      name: 'Fast Mode',
      description: 'Automatic processing with optimized settings',
      icon: <Zap className="h-6 w-6" />,
      color: 'text-highlight'
    },
    {
      id: 'config1',
      name: 'Config-1 Mode',
      description: 'Configurable processing with custom options',
      icon: <Settings className="h-6 w-6" />,
      color: 'text-highlight'
    },
    {
      id: 'deep',
      name: 'Deep Config Mode',
      description: 'Advanced 10-step pipeline with full control',
      icon: <Microscope className="h-6 w-6" />,
      color: 'text-highlight'
    },
    {
      id: 'campaign',
      name: 'Campaign Mode',
      description: 'Specialized for media campaigns and contacts',
      icon: <Target className="h-6 w-6" />,
      color: 'text-highlight'
    }
  ];

  return (
    <div className="mb-8">
      <h2 className="text-xl font-semibold text-textPrimary mb-4">
        Select Processing Mode
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {modes.map((mode) => (
          <div
            key={mode.id}
            className={`
              custom-card cursor-pointer transition-all duration-200
              ${currentMode === mode.id 
                ? 'border-highlight bg-secondary' 
                : 'hover:border-highlight/20'
              }
            `}
            onClick={() => setCurrentMode(mode.id)}
          >
            <div className="flex items-center space-x-3 mb-3">
              <div className={mode.color}>
                {mode.icon}
              </div>
              <h3 className="font-semibold text-textPrimary">
                {mode.name}
              </h3>
            </div>
            
            <p className="text-textSecondary text-sm mb-4">
              {mode.description}
            </p>
            
            <Button
              variant={currentMode === mode.id ? 'primary' : 'outline'}
              size="sm"
              className="w-full"
              onClick={() => setCurrentMode(mode.id)}
            >
              {currentMode === mode.id ? 'Selected' : 'Select'}
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModeSelector;
