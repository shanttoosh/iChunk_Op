// src/components/Search/SearchInterface.jsx
import React, { useState } from 'react';
import { Search, Brain, FileText, Download } from 'lucide-react';
import useAppStore from '../../stores/appStore';
import useCampaignStore from '../../stores/campaignStore';
import Card from '../UI/Card';
import Button from '../UI/Button';
import Input from '../UI/Input';
import Select from '../UI/Select';
import retrievalService from '../../services/retrieval.service';
import campaignService from '../../services/campaign.service';
import { DEFAULT_VALUES } from '../../utils/constants';

const SearchInterface = () => {
  const [query, setQuery] = useState('');
  const [k, setK] = useState(DEFAULT_VALUES.RETRIEVAL_K);
  const [searchField, setSearchField] = useState('all');
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState(null);

  const {
    currentMode,
    llmMode,
    apiResults,
    setRetrievalResults
  } = useAppStore();

  const {
    useSmartRetrieval,
    campaignResults
  } = useCampaignStore();

  const handleSearch = async () => {
    if (!query.trim()) {
      alert('Please enter a search query');
      return;
    }

    setIsSearching(true);
    
    try {
      let results;
      
      if (currentMode === 'campaign') {
        // Campaign mode search
        if (llmMode === 'LLM Enhanced') {
          results = await campaignService.llmAnswer(query);
        } else if (useSmartRetrieval) {
          results = await campaignService.smartRetrieval(query, searchField, k, true);
        } else {
          results = await campaignService.retrieve(query, searchField, k, true);
        }
      } else {
        // Other modes search
        if (llmMode === 'LLM Enhanced') {
          results = await retrievalService.llmAnswer(query);
        } else {
          results = await retrievalService.retrieve(query, k);
        }
      }

      setSearchResults(results);
      setRetrievalResults(results);

    } catch (error) {
      console.error('Search error:', error);
      alert('Search failed. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  const renderResults = () => {
    if (!searchResults) return null;

    if (llmMode === 'LLM Enhanced') {
      return (
        <div className="space-y-4">
          <div className="p-4 bg-secondary rounded-lg">
            <h3 className="text-lg font-semibold text-textPrimary mb-3 flex items-center">
              <Brain className="h-5 w-5 mr-2 text-highlight" />
              AI Answer
            </h3>
            <div className="text-textPrimary whitespace-pre-wrap">
              {searchResults.answer || searchResults.response}
            </div>
          </div>

          {searchResults.sources && searchResults.sources.length > 0 && (
            <div>
              <h4 className="text-md font-semibold text-textPrimary mb-3">Sources:</h4>
              <div className="space-y-2">
                {searchResults.sources.map((source, index) => (
                  <div key={index} className="p-3 bg-primary rounded-lg border border-border">
                    <div className="text-sm text-textSecondary mb-2">
                      Source {index + 1} (Similarity: {source.similarity ? (source.similarity * 100).toFixed(1) + '%' : 'N/A'})
                    </div>
                    <div className="text-textPrimary scrollable-chunk">
                      {source.content || source.text}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      );
    }

    // Normal retrieval results
    if (currentMode === 'campaign') {
      return (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-textPrimary">Search Results</h3>
          {searchResults.results && searchResults.results.map((result, index) => (
            <div key={index} className="p-4 bg-primary rounded-lg border border-border">
              <div className="flex justify-between items-start mb-3">
                <div className="text-sm text-textSecondary">
                  Result {index + 1}
                  {result.match_type && (
                    <span className={`ml-2 px-2 py-1 rounded text-xs ${
                      result.match_type === 'exact' ? 'bg-success text-primary' :
                      result.match_type === 'partial' ? 'bg-warning text-primary' :
                      'bg-secondary text-textPrimary'
                    }`}>
                      {result.match_type}
                    </span>
                  )}
                </div>
                <div className="text-sm text-textSecondary">
                  Similarity: {result.similarity ? (result.similarity * 100).toFixed(1) + '%' : 'N/A'}
                </div>
              </div>
              
              <div className="text-textPrimary scrollable-chunk mb-3">
                {result.content || result.text}
              </div>

              {result.complete_record && (
                <div className="mt-3 p-3 bg-secondary rounded border border-border">
                  <h5 className="font-medium text-textPrimary mb-2">Complete Record:</h5>
                  <div className="text-sm text-textPrimary">
                    {typeof result.complete_record === 'object' ? 
                      JSON.stringify(result.complete_record, null, 2) :
                      result.complete_record
                    }
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      );
    }

    // Standard retrieval results
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-textPrimary">Search Results</h3>
        {searchResults.results && searchResults.results.map((result, index) => (
          <div key={index} className="p-4 bg-primary rounded-lg border border-border">
            <div className="flex justify-between items-start mb-3">
              <div className="text-sm text-textSecondary">
                Result {index + 1}
              </div>
              <div className="text-sm text-textSecondary">
                Similarity: {result.similarity ? (result.similarity * 100).toFixed(1) + '%' : 'N/A'}
              </div>
            </div>
            
            <div className="text-textPrimary scrollable-chunk">
              {result.content || result.text}
            </div>

            {result.metadata && (
              <div className="mt-3 p-3 bg-secondary rounded border border-border">
                <h5 className="font-medium text-textPrimary mb-2">Metadata:</h5>
                <div className="text-sm text-textPrimary">
                  {typeof result.metadata === 'object' ? 
                    JSON.stringify(result.metadata, null, 2) :
                    result.metadata
                  }
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    );
  };

  const searchFieldOptions = [
    { value: 'all', label: 'All Fields' },
    { value: 'company', label: 'Company' },
    { value: 'contact', label: 'Contact' },
    { value: 'email', label: 'Email' },
    { value: 'phone', label: 'Phone' },
    { value: 'auto', label: 'Auto-detect' }
  ];

  return (
    <Card>
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-8 h-8 bg-highlight rounded flex items-center justify-center">
          <Search className="h-5 w-5 text-white" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-textPrimary">Search Interface</h2>
          <p className="text-textSecondary text-sm">
            {llmMode === 'LLM Enhanced' ? 'AI-powered semantic search' : 'Vector similarity search'}
          </p>
        </div>
      </div>

      {/* Search Form */}
      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-textSecondary text-sm font-medium mb-2">
            Search Query
          </label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your search query..."
            className="input-field w-full h-24 resize-none"
            disabled={isSearching}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {currentMode === 'campaign' && (
            <Select
              label="Search Field"
              value={searchField}
              onChange={(e) => setSearchField(e.target.value)}
              options={searchFieldOptions}
            />
          )}

          <div>
            <label className="block text-textSecondary text-sm font-medium mb-2">
              Number of Results (K)
            </label>
            <input
              type="range"
              min="1"
              max={DEFAULT_VALUES.MAX_RETRIEVAL_K}
              value={k}
              onChange={(e) => setK(parseInt(e.target.value))}
              className="w-full"
              disabled={isSearching}
            />
            <div className="flex justify-between text-xs text-textSecondary mt-1">
              <span>1</span>
              <span className="text-highlight font-medium">{k}</span>
              <span>{DEFAULT_VALUES.MAX_RETRIEVAL_K}</span>
            </div>
          </div>
        </div>

        <div className="flex justify-start mt-6">
          <Button
            variant="primary"
            size="sm"
            onClick={handleSearch}
            disabled={!query.trim() || isSearching}
            loading={isSearching}
            className="flex items-center space-x-1"
          >
            <Search className="h-4 w-4" />
            <span>{isSearching ? 'Searching...' : 'Search'}</span>
          </Button>
        </div>
      </div>

      {/* Search Results */}
      {searchResults && (
        <div className="border-t border-border pt-6">
          {renderResults()}
        </div>
      )}

      {/* No Results Message */}
      {searchResults && (!searchResults.results || searchResults.results.length === 0) && !searchResults.answer && (
        <div className="text-center py-8 text-textSecondary">
          <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>No results found for your query.</p>
          <p className="text-sm">Try adjusting your search terms or increasing the number of results.</p>
        </div>
      )}
    </Card>
  );
};

export default SearchInterface;