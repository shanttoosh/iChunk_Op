# Data Flow Architecture

##  Complete System Data Flow

```

                            DATA SOURCES                                      
         
    CSV Files    MySQL Database     PostgreSQL     Web Uploads     
     (3GB+)         Tables           Tables        (Streamlit)     
         

                                                                
                                               
                                                                 
          
                          
                          
    
                       INGESTION LAYER                             
        
        • Streaming I/O (No memory load for large files)        
        • Batch Processing (2K row batches)                     
        • Encoding Detection (UTF-8, chardet fallback)          
        • Size Validation (up to 3GB+)                          
        • Database Chunked Imports                              
        
    
                                
                                
    
                      PREPROCESSING LAYER                          
                                                                   
          
        Stage 1: Header Normalization                          
        • Lowercase conversion                                 
        • Special character removal                            
        • Underscore normalization                             
        • Null header handling                                 
          
                                                                  
          
        Stage 2: Text Cleaning (Default)                       
        • HTML tag removal                                     
        • Lowercase text                                       
        • Whitespace normalization                             
        • Excel formula safety                                 
          
                                                                  
          
        Stage 3: Advanced Processing (Deep Config Only)        
        • Type conversion                                      
        • Null handling (7 strategies)                         
        • Duplicate removal                                    
        • Stopword removal                                     
        • Lemmatization/Stemming                               
          
    
                                
                                
    
                        CHUNKING LAYER                             
                                                                   
             
        Fixed     Recursive     Semantic    Document    
       Chunking    Key-Value   Clustering     Based     
             
                                                              
                                                              
                      
                                                                 
                                                                 
        
        Chunk Output:                                           
        • Text chunks (string array)                            
        • Metadata (optional)                                   
          - Chunk IDs                                           
          - Source references                                   
          - Statistical metadata (numeric aggregations)        
          - Categorical metadata (mode values)                 
        
    
                                
                                
    
                       EMBEDDING LAYER                             
                                                                   
        
        Model Selection:                                        
                
         Local Models        OpenAI API (Optional)          
         • paraphrase-*      • text-embedding-ada-002       
         • all-MiniLM-*      • Custom base URL              
         384 dimensions      • 1536 dimensions              
                
        
                                                                  
                                            
                                                                   
        
        Processing Strategy:                                     
        • Parallel processing (6 workers)                        
        • Batch embedding (256 chunks/batch)                     
        • Memory management                                      
        • Progress tracking                                      
        
                                                                   
                                                                   
        
        Vector Output: numpy.ndarray                             
        Shape: (num_chunks, embedding_dim)                       
        dtype: float32                                           
        
    
                                
                                
    
                        STORAGE LAYER                              
                                                                   
                    
            FAISS Storage               ChromaDB Storage      
                                                              
          • IndexFlatL2                • PersistentClient     
          • Metadata indexing          • Collections          
          • Fast filtering             • Metadata queries     
          • Disk persistence           • Distance metrics     
                    
                                                                 
                                   
                                                                  
        
        Enhanced Features:                                       
        • Metadata indexing for fast filtering                  
        • Batch insertion (1K vectors/batch)                    
        • Multiple distance metrics (L2, cosine, IP)            
        • Persistent storage                                    
        
    
                                
                                
    
                       RETRIEVAL LAYER                             
                                                                   
        
        Query Processing:                                        
        1. Text query → Vector embedding (same model)            
        2. Vector search in database                             
        3. Metadata filtering (optional)                         
        4. Distance calculation                                  
        5. Ranking by similarity                                 
        
                                                                  
                                                                  
        
        Metadata Filtering Pipeline:                             
        • Parse filter conditions (JSON)                         
        • Apply metadata index lookup                            
        • Intersect matching indices                             
        • Filter vector search results                           
        • Return top-k filtered results                          
        
                                                                  
                                                                  
        
        Result Formatting:                                       
        {                                                        
          "rank": 1,                                             
          "content": "chunk text",                               
          "similarity": 0.89,                                    
          "distance": 0.11,                                      
          "metadata": {...}                                      
        }                                                        
        
    
                                
                                
    
                         EXPORT LAYER                             
                                                                  
                        
         Chunks      Embeddings   Preprocessed              
         (CSV)       (JSON)        Data (CSV)               
                        
    
```

##  Mode-Specific Workflows

### Fast Mode Flow

```
CSV Upload
    
    
Default Preprocessing
(Headers + Text Cleaning)
    
    
Semantic Clustering
(KMeans, n=20)
    
    
Paraphrase-MiniLM Embedding
(384-dim vectors)
    
    
FAISS Storage
    
    
Ready for Retrieval

Time: ~60s for 100K rows
```

### Config-1 Mode Flow

```
CSV/DB Upload
    
    
Default Preprocessing (optional)
    
    
User-Selected Chunking
(Fixed/Recursive/Semantic/Document)
    
    
User-Selected Model
(Local or OpenAI)
    
    
User-Selected Storage
(FAISS or ChromaDB)
    
    
User-Selected Metric
(Cosine/Euclidean/Dot)
    
    
Ready for Retrieval

Time: ~90s for 100K rows
```

### Deep Config Mode Flow

```
CSV/DB Upload
    
    
Step 1: Preprocess & Load
    
    
Step 2: Type Conversion
    
    
Step 3: Null Handling
    
    
Step 4: Duplicate Removal
    
    
Step 5: Stopword Removal
    
    
Step 6: Text Normalization
    
    
Step 7: Chunking + Metadata
    
    
Step 8: Embedding Generation
    
    
Step 9: Vector Storage
    
    
Ready for Filtered Retrieval

Time: ~120s for 100K rows (all steps)
```

##  State Management Flow

```

                 GLOBAL STATE VARIABLES                   
      
    current_df          : Processed DataFrame          
    current_chunks      : Text chunks array            
    current_embeddings  : Vector array                 
    current_metadata    : Metadata list                
    current_model       : Embedding model instance     
    current_store_info  : Vector DB connection         
    current_file_info   : Source information           
      

                         
                         

              PERSISTENCE MECHANISM                       
      
    current_state.pkl (Pickle serialization)           
    • Saved after each processing pipeline             
    • Loaded on API startup                            
    • Used for retrieval across sessions               
      
                                                          
      
    faiss_store/ (FAISS indices)                       
    • index.faiss (vector index)                       
    • data.pkl (documents + metadata)                  
      
                                                          
      
    chromadb_store/ (ChromaDB persistence)             
    • Collections with embeddings                      
    • Metadata stored with vectors                     
      

```

##  Retrieval Flow with Metadata

```
Query Text
    
    
Embed Query
(Same model as chunks)
    
    
Load Metadata Filter
(Optional JSON)
    
     No Filter 
                         
     With Filter 
                         
                         
Apply Metadata      Vector Search
Index Lookup        in Full DB
                         
                         
Get Matching IDs          
                         
                         
Vector Search 
(Filtered indices)
    
    
Calculate Similarity
    
    
Rank Results
    
    
Return Top-K
```

##  Memory Management Flow

```
Large File (>10MB)
    
    

 Stream to Temp File     
 (No memory load)        

           
           

 Batch Processing        
 (2K rows at a time)     

           
           

 Process Each Batch      
 • Preprocess            
 • Chunk                 
 • Collect results       

           
           

 Garbage Collection      
 (After each batch)      

           
           

 Aggregate All Batches   

           
           

 Batch Embedding         
 (256 chunks at a time)  

           
           

 Batch Storage           
 (1K vectors at a time)  

           
           
     Complete
```

##  Parallel Processing Flow

```
Input Chunks (N chunks)
    
    
Split into Batches
(N / num_workers batches)
    
    
                                  
Worker Worker Worker Worker Worker Worker
  #1     #2     #3     #4     #5     #6
                                  
       (Each worker embeds a batch)  
                                  
    
                  
                  
         Combine Results
         (numpy.vstack)
                  
                  
          Final Embeddings
```

##  Data Type Flow

```
Input Data (Mixed Types)
    
    

 Automatic Type Detection            
 • object → string                   
 • int64 → integer                   
 • float64 → float                   
 • datetime64 → datetime             

               
               

 User Type Conversion (Deep Config)  
 • String → Integer/Float/Boolean    
 • String → Datetime                 
 • Numeric → String                  
 • Any → Category                    

               
               

 Validation & Error Handling         
 • Coerce errors to NaN              
 • Handle conversion failures        
 • Log conversion results            

               
               
     Typed DataFrame
```

##  Performance Optimization Flow

```
Input Request
    
    

 Check Cache         
 (file hash lookup)  

             
  Found    Not Found
             
             
  Return  
  Cached   Check File Size     
  Result  
              Small    Large
                       
                       
          Standard  Optimized
         Processing Processing
                       
               
               
     
      Apply Turbo Mode?    
     
       Yes       No
                   
                   
    Parallel    Sequential
   Processing   Processing
                   
           
                
     
      Cache Result         
     
                
          Return Result
```

---

**Next**: See [Component Details](./components.md) for implementation specifics.

