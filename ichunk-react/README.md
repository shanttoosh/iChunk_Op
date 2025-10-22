# iChunk Optimizer - Frontend Documentation

## Overview

The frontend is a React-based web application built with Vite that provides an intuitive user interface for the iChunk Optimizer RAG system. It offers multiple processing modes, real-time status updates, and comprehensive data management capabilities.

## Architecture

- **Framework**: React 18 with Vite
- **Port**: 5173 (development)
- **State Management**: Zustand stores
- **UI Library**: Custom components with Tailwind CSS
- **HTTP Client**: Axios for API communication
- **Build Tool**: Vite for fast development and building

## File Structure

```
ichunk-react/
├── src/
│   ├── components/           # React components
│   │   ├── Layout/          # Layout components
│   │   ├── Modes/           # Processing mode components
│   │   ├── DeepConfig/      # Deep Config specific components
│   │   ├── Search/          # Search components
│   │   ├── Export/          # Export components
│   │   └── UI/              # Reusable UI components
│   ├── services/            # API service layer
│   ├── stores/              # State management (Zustand)
│   ├── utils/               # Utility functions
│   └── App.jsx              # Main React application
├── public/                  # Static assets
├── package.json             # Node.js dependencies
├── vite.config.js           # Vite configuration
├── .gitignore              # Git ignore rules
└── README.md               # This documentation
```

## Core Components

### Layout Components

#### Header.jsx
Application header component displaying the main title and branding.

**Features:**
- Application title display
- Responsive design
- Clean, minimal interface

#### Sidebar.jsx
Navigation sidebar with mode selection, status display, and session management.

**Features:**
- Mode selection display
- Processing status indicators
- File information display
- Session management (reset session)
- Collapsible design
- Real-time status updates

**Key Functions:**
- `handleResetSession()`: Resets both backend and frontend state
- Status display for all processing steps
- File information display
- Mode indicator

#### Footer.jsx
Application footer with additional information and links.

### Mode Components

#### ModeSelector.jsx
Main mode selection interface allowing users to choose processing modes.

**Features:**
- Visual mode cards with descriptions
- Mode selection with visual feedback
- Uniform button design
- Orange highlight for selected mode
- Responsive grid layout

**Modes Available:**
- **Fast Mode**: Quick processing with optimized settings
- **Config-1 Mode**: Balanced configuration with moderate customization
- **Deep Config Mode**: Advanced 10-step pipeline with full control
- **Campaign Mode**: Specialized for media campaigns and contacts

#### FastMode.jsx
Fast processing mode component with minimal configuration options.

**Features:**
- File upload interface
- Database import option
- Basic configuration options
- Processing status display
- Results display

#### Config1Mode.jsx
Config-1 processing mode with moderate customization options.

**Features:**
- File upload and database import
- Chunking configuration (size, overlap)
- Embedding model selection
- Storage type selection
- Processing pipeline execution
- Progressive UI flow

**Configuration Options:**
- Chunk size and overlap
- Embedding model (local or OpenAI)
- Storage type (FAISS or ChromaDB)
- Retrieval metric selection

#### DeepConfigMode.jsx
Advanced Deep Config mode with 9-step processing pipeline.

**Features:**
- Step-by-step configuration
- Detailed preprocessing options
- Advanced chunking strategies
- Embedding configuration
- Storage configuration
- Progress tracking
- Metadata column selection

**9-Step Pipeline:**
1. Data Preprocessing
2. Type Conversion
3. Null Handling
4. Duplicate Removal
5. Stopword Processing
6. Text Normalization
7. Chunking
8. Embedding
9. Storage

**Advanced Features:**
- Null analysis table
- Type conversion configuration
- Metadata column selector
- Chunking strategy selection
- Batch processing options

#### CampaignMode.jsx
Campaign mode specialized for marketing campaign data.

**Features:**
- Campaign-specific configuration
- Contact processing options
- Campaign metadata handling
- Specialized chunking strategies
- Campaign-aware search

### Deep Config Components

#### MetadataColumnSelector.jsx
Smart UI for selecting metadata columns based on backend limits.

**Features:**
- Column type detection
- Selection limits enforcement
- Visual column categorization
- Smart filtering options

#### NullAnalysisTable.jsx
Advanced null value analysis and handling configuration.

**Features:**
- Null value detection
- Strategy selection per column
- Custom value configuration
- Visual null analysis

#### TypeConversionConfig.jsx
Data type conversion configuration interface.

**Features:**
- Automatic type detection
- Manual type selection
- Bulk type conversion
- Type validation

#### ChunkingConfig.jsx
Advanced chunking strategy configuration.

**Features:**
- Multiple chunking strategies
- Parameter configuration
- Strategy-specific options
- Preview functionality

### Search Components

#### SearchInterface.jsx
Semantic search interface with advanced filtering options.

**Features:**
- Query input with suggestions
- Result count configuration
- Search field selection
- Real-time search results
- LLM-enhanced answers
- Metadata filtering
- Campaign-specific search

**Search Types:**
- Basic semantic search
- Metadata-filtered search
- LLM-enhanced search
- Campaign smart retrieval

### Export Components

#### ExportSection.jsx
Data export interface for downloading processed data.

**Features:**
- Multiple export formats
- Processed data download
- Chunk data export
- Embedding data export
- Preprocessed data export
- Export status tracking

**Export Options:**
- CSV format for processed data
- JSON format for chunks
- Binary format for embeddings
- Metadata export

### UI Components

#### Button.jsx
Reusable button component with multiple variants.

**Variants:**
- Primary: Main action buttons
- Secondary: Secondary actions
- Outline: Border-only buttons
- Ghost: Transparent buttons

**Sizes:**
- Small, medium, large
- Icon-only options
- Loading states

#### Input.jsx
Form input component with validation support.

**Features:**
- Type validation
- Error states
- Placeholder support
- Disabled states
- Required field indicators

#### Card.jsx
Container component for grouping related content.

**Features:**
- Consistent spacing
- Border and shadow options
- Header and footer support
- Responsive design

#### Modal.jsx
Modal dialog component for overlays and forms.

**Features:**
- Backdrop support
- Close on escape
- Focus management
- Size variants

## Services Layer

### API Services

#### api.js
Base API configuration with Axios setup.

**Features:**
- Base URL configuration
- Request/response interceptors
- Error handling
- Timeout configuration
- CORS handling

#### fastMode.service.js
API calls for Fast mode processing.

**Functions:**
- `runFastMode()`: Execute Fast mode processing
- File upload handling
- Database import support

#### config1.service.js
API calls for Config-1 mode processing.

**Functions:**
- `runConfig1Mode()`: Execute Config-1 processing
- Configuration parameter handling
- Progress tracking

#### deepConfig.service.js
API calls for Deep Config mode processing.

**Functions:**
- `preprocess()`: Data preprocessing
- `typeConvert()`: Type conversion
- `nullHandle()`: Null value handling
- `duplicates()`: Duplicate removal
- `stopwords()`: Stopword processing
- `normalize()`: Text normalization
- `chunk()`: Chunking
- `embed()`: Embedding generation
- `store()`: Vector storage
- `getMetadataColumns()`: Metadata column information

#### campaign.service.js
API calls for Campaign mode processing.

**Functions:**
- `runCampaign()`: Campaign processing
- `smartRetrieval()`: Campaign-specific search
- `llmAnswer()`: LLM-enhanced answers

#### retrieval.service.js
API calls for semantic search functionality.

**Functions:**
- `retrieve()`: Basic semantic search
- `retrieveWithMetadata()`: Metadata-filtered search
- Query parameter handling

#### export.service.js
API calls for data export functionality.

**Functions:**
- `exportChunks()`: Download chunk data
- `exportEmbeddings()`: Download embedding data
- `exportPreprocessed()`: Download preprocessed data

#### system.service.js
API calls for system management.

**Functions:**
- `getSystemInfo()`: System information
- `getFileInfo()`: File information
- `resetSession()`: Reset backend state
- `healthCheck()`: System health check

## State Management

### Zustand Stores

#### appStore.js
Main application state management.

**State Properties:**
- `currentMode`: Currently selected processing mode
- `processStatus`: Processing step status
- `fileInfo`: File information
- `llmMode`: LLM mode selection
- `systemInfo`: System information
- `apiResults`: Processing results
- `retrievalResults`: Search results

**Actions:**
- `setCurrentMode()`: Set processing mode
- `updateProcessStatus()`: Update processing status
- `setFileInfo()`: Set file information
- `toggleLLMMode()`: Toggle LLM mode
- `setSystemInfo()`: Set system information
- `setApiResults()`: Set processing results
- `setRetrievalResults()`: Set search results
- `resetSession()`: Reset application state

#### uiStore.js
UI state management for interface elements.

**State Properties:**
- `sidebarCollapsed`: Sidebar collapse state
- `showDatabaseModal`: Database modal visibility
- `showMetadataModal`: Metadata modal visibility

**Actions:**
- `toggleSidebar()`: Toggle sidebar
- `toggleDatabaseModal()`: Toggle database modal
- `toggleMetadataModal()`: Toggle metadata modal

#### campaignStore.js
Campaign-specific state management.

**State Properties:**
- `campaignResults`: Campaign processing results
- `useSmartRetrieval`: Smart retrieval toggle
- `campaignConfig`: Campaign configuration

**Actions:**
- `setCampaignResults()`: Set campaign results
- `toggleSmartRetrieval()`: Toggle smart retrieval
- `setCampaignConfig()`: Set campaign configuration

## Utility Functions

### constants.js
Application constants and configuration values.

**Constants:**
- `DEFAULT_VALUES`: Default configuration values
- `PROCESS_STEPS`: Processing step definitions
- `CHUNKING_METHODS`: Available chunking methods
- `EMBEDDING_MODELS`: Available embedding models
- `STORAGE_TYPES`: Available storage types

### formatting.js
Data formatting and display utilities.

**Functions:**
- `formatFileSize()`: Format file sizes
- `formatNumber()`: Format numbers
- `formatDate()`: Format dates
- `truncateText()`: Truncate long text

## Progressive UI Flow

The application implements a progressive UI flow where interface elements appear based on the current processing stage:

### Stage 1: Mode Selection
- Mode selector visible
- File upload section visible
- Database import option visible

### Stage 2: Configuration (After Data Upload)
- Configuration options appear
- Processing parameters visible
- Process button enabled

### Stage 3: Processing
- Progress indicators active
- Status updates in real-time
- Processing steps highlighted

### Stage 4: Results (After Processing)
- Search interface appears
- Export section becomes available
- Results display enabled

## Responsive Design

### Breakpoints
- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

### Responsive Features
- Collapsible sidebar
- Responsive grid layouts
- Mobile-friendly forms
- Touch-friendly buttons
- Adaptive typography

## Error Handling

### Error States
- Network errors
- Validation errors
- Processing errors
- File upload errors

### Error Display
- Toast notifications
- Inline error messages
- Error boundaries
- Fallback UI components

### Error Recovery
- Retry mechanisms
- Graceful degradation
- User guidance
- Error reporting

## Performance Optimization

### Code Splitting
- Route-based splitting
- Component lazy loading
- Dynamic imports
- Bundle optimization

### Caching
- API response caching
- Component memoization
- State persistence
- Local storage usage

### Optimization Techniques
- Virtual scrolling for large lists
- Debounced search inputs
- Optimized re-renders
- Efficient state updates

## Development

### Prerequisites
- Node.js 16 or higher
- npm or yarn package manager

### Setup
1. Navigate to frontend directory
2. Install dependencies: `npm install`
3. Start development server: `npm run dev`

### Build
- Development build: `npm run dev`
- Production build: `npm run build`
- Preview build: `npm run preview`

### Testing
- Unit tests: `npm run test`
- E2E tests: `npm run test:e2e`
- Coverage: `npm run test:coverage`

## Configuration

### Environment Variables
- `VITE_API_BASE_URL`: Backend API base URL
- `VITE_APP_TITLE`: Application title
- `VITE_APP_VERSION`: Application version

### Build Configuration
- Vite configuration in `vite.config.js`
- Tailwind configuration
- ESLint configuration
- TypeScript configuration (if used)

## Dependencies

### Core Dependencies
- **react**: React framework
- **react-dom**: React DOM rendering
- **zustand**: State management
- **axios**: HTTP client
- **lucide-react**: Icon library

### Development Dependencies
- **vite**: Build tool
- **@vitejs/plugin-react**: React plugin
- **tailwindcss**: CSS framework
- **autoprefixer**: CSS prefixing
- **postcss**: CSS processing

### UI Dependencies
- **clsx**: Class name utility
- **tailwind-merge**: Tailwind class merging

## Browser Support

### Supported Browsers
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Features
- Modern JavaScript (ES2020+)
- CSS Grid and Flexbox
- Web APIs (File API, Fetch API)
- Local Storage

## Accessibility

### Features
- Keyboard navigation
- Screen reader support
- ARIA labels and roles
- Focus management
- Color contrast compliance

### Guidelines
- WCAG 2.1 AA compliance
- Semantic HTML structure
- Accessible form controls
- Error message accessibility

## Security

### Features
- Input validation
- XSS prevention
- CSRF protection
- Secure API communication
- File upload validation

### Best Practices
- Sanitize user inputs
- Validate file types
- Use HTTPS in production
- Implement proper error handling
- Regular dependency updates