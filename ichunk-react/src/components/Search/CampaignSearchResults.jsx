// src/components/Search/CampaignSearchResults.jsx
import React, { useState } from 'react';
import { ChevronDown, ChevronUp, User, Building, Phone, Mail, Calendar, Target } from 'lucide-react';
import Card from '../UI/Card';

const CampaignSearchResults = ({ searchResults, query }) => {
  const [expandedGroups, setExpandedGroups] = useState({});

  if (!searchResults || !searchResults.results) {
    return (
      <div className="text-center py-8">
        <p className="text-textSecondary">No search results found</p>
      </div>
    );
  }

  const toggleGroup = (groupIndex) => {
    setExpandedGroups(prev => ({
      ...prev,
      [groupIndex]: !prev[groupIndex]
    }));
  };

  const formatRecord = (record) => {
    const formattedRecord = {};
    for (const [key, value] of Object.entries(record)) {
      if (value !== null && value !== undefined && value !== '') {
        formattedRecord[key] = value;
      }
    }
    return formattedRecord;
  };

  const getFieldIcon = (fieldName) => {
    const field = fieldName.toLowerCase();
    if (field.includes('email')) return <Mail className="h-4 w-4" />;
    if (field.includes('phone') || field.includes('contact')) return <Phone className="h-4 w-4" />;
    if (field.includes('company') || field.includes('brand')) return <Building className="h-4 w-4" />;
    if (field.includes('date') || field.includes('created')) return <Calendar className="h-4 w-4" />;
    if (field.includes('campaign') || field.includes('lead')) return <Target className="h-4 w-4" />;
    return <User className="h-4 w-4" />;
  };

  const renderRecord = (record, recordIndex) => {
    const formattedRecord = formatRecord(record);
    
    return (
      <div key={recordIndex} className="mb-4 p-4 bg-secondary rounded-lg border border-border">
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-medium text-textPrimary">Record {recordIndex + 1}:</h4>
        </div>
        
        <div className="space-y-2">
          {Object.entries(formattedRecord).map(([key, value]) => (
            <div key={key} className="flex items-start space-x-2">
              <div className="flex items-center space-x-1 text-textSecondary min-w-0 flex-shrink-0">
                {getFieldIcon(key)}
                <span className="text-sm font-medium">{key}:</span>
              </div>
              <div className="text-sm text-textPrimary break-words">
                {typeof value === 'object' ? JSON.stringify(value) : String(value)}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderContactGroup = (result, groupIndex) => {
    const isExpanded = expandedGroups[groupIndex];
    const records = result.complete_record || [];
    
    return (
      <Card key={groupIndex} className="mb-4">
        <div 
          className="cursor-pointer p-4 hover:bg-primary/50 transition-colors"
          onClick={() => toggleGroup(groupIndex)}
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <User className="h-5 w-5 text-highlight" />
              <div>
                <h3 className="font-semibold text-textPrimary">
                  👤 Contact Group #{groupIndex + 1}
                </h3>
                <p className="text-sm text-textSecondary">
                  Similarity: {result.similarity ? result.similarity.toFixed(3) : 'N/A'}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-textSecondary">
                Rank: {result.rank || groupIndex + 1}
              </span>
              {isExpanded ? (
                <ChevronUp className="h-5 w-5 text-textSecondary" />
              ) : (
                <ChevronDown className="h-5 w-5 text-textSecondary" />
              )}
            </div>
          </div>
        </div>
        
        {isExpanded && (
          <div className="px-4 pb-4 border-t border-border">
            <div className="mt-4 space-y-4">
              {records.map((record, recordIndex) => renderRecord(record, recordIndex))}
            </div>
          </div>
        )}
      </Card>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-textPrimary">
            🎯 Campaign Data Retrieval
          </h2>
          <p className="text-sm text-textSecondary mt-1">
            Search Campaign Data: <span className="font-medium text-highlight">"{query}"</span>
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm text-textSecondary">
            Results: <span className="font-semibold text-textPrimary">{searchResults.results.length}</span>
          </p>
          {searchResults.total_matches && (
            <p className="text-xs text-textSecondary">
              Total matches: {searchResults.total_matches}
            </p>
          )}
        </div>
      </div>

      <div className="space-y-4">
        {searchResults.results.map((result, index) => renderContactGroup(result, index))}
      </div>

      {searchResults.results.length === 0 && (
        <div className="text-center py-8">
          <Target className="h-12 w-12 text-textSecondary mx-auto mb-4" />
          <p className="text-textSecondary">No campaign data found for "{query}"</p>
          <p className="text-sm text-textSecondary mt-2">
            Try different search terms or check your data
          </p>
        </div>
      )}
    </div>
  );
};

export default CampaignSearchResults;
