// src/App.jsx
import React, { useEffect } from 'react';
import { Toaster } from 'react-hot-toast';
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import Footer from './components/Layout/Footer';
import ModeSelector from './components/Modes/ModeSelector';
import FastMode from './components/Modes/FastMode';
import Config1Mode from './components/Modes/Config1Mode';
import DeepConfigMode from './components/Modes/DeepConfigMode';
import CampaignMode from './components/Modes/CampaignMode';
import useAppStore from './stores/appStore';
import useUIStore from './stores/uiStore';
import systemService from './services/system.service';

function App() {
  const { currentMode, setSystemInfo } = useAppStore();
  const { sidebarCollapsed } = useUIStore();

  useEffect(() => {
    // Load system info on app start
    const loadSystemInfo = async () => {
      try {
        const systemInfo = await systemService.getSystemInfo();
        setSystemInfo(systemInfo);
      } catch (error) {
        console.error('Failed to load system info:', error);
      }
    };

    loadSystemInfo();
  }, [setSystemInfo]);

  const renderModeComponent = () => {
    switch (currentMode) {
      case 'fast':
        return <FastMode />;
      case 'config1':
        return <Config1Mode />;
      case 'deep':
        return <DeepConfigMode />;
      case 'campaign':
        return <CampaignMode />;
      default:
        return null;
    }
  };

         try {
           return (
             <div className="flex h-screen bg-primaryDark text-textPrimary">
               <Sidebar />
               
               <main className={`flex-1 overflow-y-auto p-6 bg-primaryDark transition-all duration-300 ${sidebarCollapsed ? 'ml-16' : 'ml-80'} space-y-8`}>
                 <Header />
                 
                 <ModeSelector />
                 
                 {renderModeComponent()}
                 
                 <Footer />
               </main>
        
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 5000,
            style: {
              background: '#282828',
              color: '#ffffff',
              border: '1px solid #ffffff1f',
            },
            success: {
              iconTheme: {
                primary: '#d8fc77',
                secondary: '#282828',
              },
            },
            error: {
              iconTheme: {
                primary: '#dc143c',
                secondary: '#282828',
              },
            },
          }}
        />
      </div>
    );
  } catch (error) {
    console.error('App render error:', error);
    return (
      <div className="flex h-screen bg-red-500 text-white items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">App Error</h1>
          <p className="text-lg">{error.message}</p>
          <p className="text-sm mt-2">Check console for details</p>
        </div>
      </div>
    );
  }
}

export default App;