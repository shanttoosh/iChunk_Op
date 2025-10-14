# Chunking Optimizer API - Visual Guide

##  Complete Documentation Structure

```
 CHUNKING OPTIMIZER API v2.0

  API_USAGE_GUIDE.md  (START HERE - Complete reference)
    Executive summary, workflows, examples, best practices

  API-DOCUMENTATION/
   
     README.md (Documentation overview)
   
     01-GETTING-STARTED/
       installation.md (Setup instructions)
       quickstart.md (5-minute tutorial)
       authentication.md (Security setup)
   
     02-ARCHITECTURE/
       system-overview.md (High-level design)
       data-flow.md  (Processing pipelines)
       components.md (Module details)
   
     03-API-REFERENCE/
       core-endpoints.md  (All 24+ endpoints)
       processing-modes.md (Mode comparison)
       retrieval.md (Search endpoints)
       database.md (DB integration)
       export.md (Export endpoints)
       system.md (System endpoints)
   
     04-WORKFLOWS/
       fast-mode-workflow.md (Simple workflow)
       config1-workflow.md (Custom workflow)
       deep-config-workflow.md (Advanced workflow)
       database-workflow.md (DB workflow)
   
     05-EXAMPLES/
       curl-examples.md (Command-line)
       python-examples.md  (Python code)
       javascript-examples.md (JS code)
       postman-collection.json (Postman)
   
     06-ADVANCED/
       metadata-filtering.md (Filtered search)
       large-files.md (3GB+ optimization)
       performance-tuning.md (Optimization)
       openai-compatibility.md (OpenAI API)
   
     07-REFERENCE/
       error-codes.md (Error reference)
       data-types.md (Type system)
       configuration.md (Config options)
   
     08-APPENDIX/
        changelog.md (Version history)
        faq.md (Common questions)
        troubleshooting.md (Problem solving)

  TEST_DATASETS_README.md (Sample data guide)

  requirements.txt (Dependencies)

  Source Code
     main.py (FastAPI server)
     backend.py (Processing engine)
     app.py (Streamlit UI)
```

---

##  Three Processing Modes - Visual Comparison

```
+-----------------------------------------------------------------------+
|                          FAST MODE                                    |
|  +----------+    +----------+    +----------+    +----------+        |
|  |   CSV    |--->|   Auto   |--->| Semantic |--->|  FAISS   |        |
|  |  Upload  |    | Process  |    |Clustering|    |  Store   |        |
|  +----------+    +----------+    +----------+    +----------+        |
|                                                                       |
|  Time: ~60s for 100K rows                                            |
|  Best For: Quick prototyping, demos, exploration                     |
|  API Calls: 1                                                        |
|  Configuration: None (automatic)                                     |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
|                         CONFIG-1 MODE                                 |
|  +----------+    +----------+    +----------+    +----------+        |
|  |   CSV    |--->|  Custom  |--->|   User   |--->|   User   |        |
|  |    or    |    | Process  |    | Chunking |    | Storage  |        |
|  |    DB    |    |(Optional)|    |  Method  |    |  Choice  |        |
|  +----------+    +----------+    +----------+    +----------+        |
|                                                                       |
|  Time: ~90s for 100K rows                                            |
|  Best For: Production apps, specific requirements                    |
|  API Calls: 1                                                        |
|  Configuration: High (10+ parameters)                                |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
|                       DEEP CONFIG MODE                                |
|  +-------+  +-------+  +-------+  +-------+  +-------+  +-------+    |
|  | Load  |->| Type  |->| Nulls |->| Dupes |->| Stop  |->| Norm  |    |
|  |       |  |Convert|  |       |  |       |  | Words |  |       |    |
|  +-------+  +-------+  +-------+  +-------+  +-------+  +-------+    |
|                                     |                                 |
|  +-------+  +-------+  +-------+    |                                 |
|  | Store |<-| Embed |<-| Chunk |<---+                                 |
|  |   +   |  |   +   |  |   +   |                                      |
|  | Index |  |Vector |  |  Meta |                                      |
|  +-------+  +-------+  +-------+                                      |
|                                                                       |
|  Time: ~120s for 100K rows                                           |
|  Best For: Enterprise, data quality, filtered search                 |
|  API Calls: 9 (step-by-step)                                         |
|  Configuration: Maximum (30+ parameters)                             |
+-----------------------------------------------------------------------+
```

---

##  Complete Data Flow (End-to-End)

```
+=========================================================================+
| STAGE 1: DATA INGESTION                                                |
|                                                                         |
|  +----------+  +----------+  +----------+  +----------+                |
|  |CSV File  |  |  MySQL   |  |PostgreSQL|  |   Web    |                |
|  | (3GB+)   |  | Database |  |          |  |  Upload  |                |
|  +----------+  +----------+  +----------+  +----------+                |
|       |             |             |             |                       |
|       +-------------+-------------+-------------+                       |
|                     |                                                   |
|                     v                                                   |
|          Streaming I/O + Encoding Detection                            |
|                     |                                                   |
|                     v                                                   |
|          Batch Processing (2K rows)                                    |
+=========================================================================+
                      |
                      v
+=========================================================================+
| STAGE 2: PREPROCESSING                                                 |
|                                                                         |
|  +------------------------------------------------------------------+  |
|  | Default (All Modes)                                              |  |
|  | • Header normalization (lowercase, underscore)                   |  |
|  | • HTML tag removal                                               |  |
|  | • Text lowercase                                                 |  |
|  | • Whitespace normalization                                       |  |
|  | • Excel formula safety                                           |  |
|  +------------------------------------------------------------------+  |
|                                                                         |
|  +------------------------------------------------------------------+  |
|  | Advanced (Deep Config Only)                                      |  |
|  | • Type conversion (7 types)                                      |  |
|  | • Null handling (7 strategies)                                   |  |
|  | • Duplicate removal (4 strategies)                               |  |
|  | • Stopword removal (spaCy/basic)                                 |  |
|  | • Text normalization (lemmatize/stem)                            |  |
|  +------------------------------------------------------------------+  |
+=========================================================================+
                      |
                      v
+=========================================================================+
| STAGE 3: CHUNKING                                                      |
|                                                                         |
|  +----------+  +----------+  +----------+  +----------+                |
|  |  Fixed   |  |Recursive |  | Semantic |  |Document  |                |
|  |          |  |Key-Value |  |  KMeans  |  |  Based   |                |
|  +----------+  +----------+  +----------+  +----------+                |
|  |Size: 400 |  | Format   |  |Clustering|  | By Key   |                |
|  |Overlap:50|  |          |  |          |  | Column   |                |
|  +----------+  +----------+  +----------+  +----------+                |
|  | Simple   |  | Struct   |  | Similar  |  | Group by |                |
|  | uniform  |  |   data   |  | content  |  | entity   |                |
|  +----------+  +----------+  +----------+  +----------+                |
|                                                                         |
|  Output: Text Chunks + Optional Metadata                               |
+=========================================================================+
                      |
                      v
+=========================================================================+
| STAGE 4: EMBEDDING GENERATION                                          |
|                                                                         |
|  +-----------------------------+  +-----------------------------+      |
|  | Local Models                |  | OpenAI API (Optional)       |      |
|  | +-------------------------+ |  | +-------------------------+ |      |
|  | | paraphrase-MiniLM       | |  | | text-embedding-ada-002  | |      |
|  | | • 384 dims              | |  | | • 1536 dims             | |      |
|  | | • Fast                  | |  | | • Best quality          | |      |
|  | +-------------------------+ |  | +-------------------------+ |      |
|  |                             |  | Requires: API Key           |      |
|  | +-------------------------+ |  | Cost: $0.0001/1K tokens     |      |
|  | | all-MiniLM-L6-v2        | |  |                             |      |
|  | | • 384 dims              | |  |                             |      |
|  | | • Balanced              | |  |                             |      |
|  | +-------------------------+ |  |                             |      |
|  +-----------------------------+  +-----------------------------+      |
|                                                                         |
|  Processing: Parallel (6 workers) + Batching (256 chunks)              |
|                                                                         |
|  Output: numpy.ndarray (N chunks x M dimensions), dtype: float32       |
+=========================================================================+
                      |
                      v
+=========================================================================+
| STAGE 5: VECTOR STORAGE                                                |
|                                                                         |
|  +-----------------------------+  +-----------------------------+      |
|  | FAISS (Facebook AI)         |  | ChromaDB                    |      |
|  | +-------------------------+ |  | +-------------------------+ |      |
|  | | IndexFlatL2             | |  | | PersistentClient        | |      |
|  | | • L2 distance           | |  | | • Collections           | |      |
|  | | • Exact search          | |  | | • Metadata support      | |      |
|  | | • Fast                  | |  | | • Multiple metrics      | |      |
|  | +-------------------------+ |  | +-------------------------+ |      |
|  |                             |  |                             |      |
|  | +-------------------------+ |  | +-------------------------+ |      |
|  | | Enhanced Features       | |  | | Distance Metrics        | |      |
|  | | • Metadata indexing     | |  | | • Cosine                | |      |
|  | | • Fast filtering        | |  | | • L2 (Euclidean)        | |      |
|  | | • Batch insertion       | |  | | • IP (Dot product)      | |      |
|  | +-------------------------+ |  | +-------------------------+ |      |
|  +-----------------------------+  +-----------------------------+      |
|                                                                         |
|  Storage: Persistent (disk) + In-memory index                          |
+=========================================================================+
                      |
                      v
+=========================================================================+
| STAGE 6: RETRIEVAL                                                     |
|                                                                         |
|  Query Text                                                             |
|       |                                                                 |
|       v                                                                 |
|  Embed Query (same model)                                               |
|       |                                                                 |
|       v                                                                 |
|  Optional: Parse Metadata Filter                                        |
|       |                                                                 |
|       v                                                                 |
|  Vector Search                                                          |
|    • No Filter: Search all vectors                                      |
|    • With Filter: Search filtered subset                                |
|       |                                                                 |
|       v                                                                 |
|  Calculate Similarity                                                   |
|    • Cosine: 1 / (1 + distance)                                         |
|    • Euclidean: Based on L2 norm                                        |
|    • Dot Product: Direct similarity                                     |
|       |                                                                 |
|       v                                                                 |
|  Return Top-K Results                                                   |
|  {                                                                      |
|    "rank": 1,                                                           |
|    "content": "chunk text...",                                          |
|    "similarity": 0.89,                                                  |
|    "distance": 0.11,                                                    |
|    "metadata": {...}                                                    |
|  }                                                                      |
+=========================================================================+
```

---

##  Endpoint Categories - Visual Map

```
+========================================================================+
|                     API ENDPOINT MAP                                   |
+========================================================================+
|                                                                        |
| +--------------------------------------------------------------------+ |
| | PROCESSING (13 endpoints)                                          | |
| +--------------------------------------------------------------------+ |
| |                                                                    | |
| |  Fast Mode                                                         | |
| |    POST /run_fast                                                  | |
| |                                                                    | |
| |  Config-1 Mode                                                     | |
| |    POST /run_config1                                               | |
| |                                                                    | |
| |  Deep Config Mode (Step-by-Step)                                   | |
| |    POST /deep_config/preprocess    [1/9]                           | |
| |    POST /deep_config/type_convert  [2/9]                           | |
| |    POST /deep_config/null_handle   [3/9]                           | |
| |    POST /deep_config/duplicates    [4/9]                           | |
| |    POST /deep_config/stopwords     [5/9]                           | |
| |    POST /deep_config/normalize     [6/9]                           | |
| |    POST /deep_config/chunk         [7/9]                           | |
| |    POST /deep_config/embed         [8/9]                           | |
| |    POST /deep_config/store         [9/9]                           | |
| +--------------------------------------------------------------------+ |
|                                                                        |
| +--------------------------------------------------------------------+ |
| | RETRIEVAL (3 endpoints)                                            | |
| +--------------------------------------------------------------------+ |
| |                                                                    | |
| | POST /retrieve                     [Basic search]                  | |
| | POST /retrieve_with_metadata       [Filtered search]              | |
| | POST /v1/retrieve                  [OpenAI-style]                  | |
| +--------------------------------------------------------------------+ |
|                                                                        |
| +--------------------------------------------------------------------+ |
| | DATABASE (3 endpoints)                                             | |
| +--------------------------------------------------------------------+ |
| |                                                                    | |
| | POST /db/test_connection           [Test DB]                       | |
| | POST /db/list_tables               [List tables]                   | |
| | POST /db/import_one                [Import & process]              | |
| +--------------------------------------------------------------------+ |
|                                                                        |
| +--------------------------------------------------------------------+ |
| | EXPORT (6 endpoints)                                               | |
| +--------------------------------------------------------------------+ |
| |                                                                    | |
| | GET /export/chunks                  [CSV]                          | |
| | GET /export/embeddings              [NumPy .npy]                   | |
| | GET /export/embeddings_text         [JSON]                         | |
| | GET /export/preprocessed            [Preprocessed CSV]             | |
| | GET /deep_config/export/chunks      [Deep Config CSV]             | |
| | GET /deep_config/export/embeddings  [Deep Config JSON]            | |
| +--------------------------------------------------------------------+ |
|                                                                        |
| +--------------------------------------------------------------------+ |
| | SYSTEM (4 endpoints)                                               | |
| +--------------------------------------------------------------------+ |
| |                                                                    | |
| | GET /                               [Root info]                    | |
| | GET /health                         [Health check]                 | |
| | GET /system_info                    [Resource info]                | |
| | GET /capabilities                   [Feature list]                 | |
| +--------------------------------------------------------------------+ |
|                                                                        |
| +--------------------------------------------------------------------+ |
| | UNIVERSAL (1 endpoint)                                             | |
| +--------------------------------------------------------------------+ |
| |                                                                    | |
| | POST /api/v1/process                [All operations consolidated]  | |
| +--------------------------------------------------------------------+ |
|                                                                        |
| +--------------------------------------------------------------------+ |
| | OPENAI COMPATIBLE (3 endpoints)                                    | |
| +--------------------------------------------------------------------+ |
| |                                                                    | |
| | POST /v1/embeddings                 [Generate embeddings]          | |
| | POST /v1/chat/completions           [Chat (requires OpenAI)]       | |
| | POST /v1/retrieve                   [Search (OpenAI style)]        | |
| +--------------------------------------------------------------------+ |
|                                                                        |
| TOTAL: 33 endpoints (24 dedicated + 9 deep config steps)              |
+========================================================================+
```

---

##  Decision Tree: Which Mode to Use?

```
                    START: Need to process data?
                            |
                            v
                     What's your goal?   
                            |
       +--------------------+--------------------+
       |                    |                    |
       v                    v                    v
  +----------+         +-----------+        +-----------+
  |  Quick   |         | Specific  |        |  Maximum  |
  |  Demo/   |         |Requirement|        |  Control  |
  |Prototype |         |  Storage  |        | +Metadata |
  +----------+         +-----------+        +-----------+
       |                    |                    |
       v                    v                    v
  +----------+         +-----------+        +-----------+
  |FAST MODE |         | CONFIG-1  |        |   DEEP    |
  +----------+         |   MODE    |        |  CONFIG   |
  | 1 API    |         +-----------+        |   MODE    |
  | call     |         | Custom    |        +-----------+
  | Auto     |         | chunking  |        | 9 Steps   |
  | 60s/100K |         | Model     |        | Full      |
  +----------+         | choice    |        | control   |
                       | 90s/100K  |        | Metadata  |
                       +-----------+        | 120s/100K |
                                            +-----------+

RECOMMENDATION MATRIX:

+------------------+----------+-----------+-------------+
| Need             | Fast     | Config-1  | Deep Config |
+------------------+----------+-----------+-------------+
| Quick start      | YES      | NO        | NO          |
| Custom chunking  | NO       | YES       | YES         |
| Model selection  | NO       | YES       | YES         |
| Data cleaning    | BASIC    | BASIC     | ADVANCED    |
| Type conversion  | NO       | NO        | YES         |
| Null handling    | NO       | NO        | YES         |
| Metadata extract | NO       | NO        | YES         |
| Filtered search  | NO       | NO        | YES         |
| Production ready | NO       | YES       | YES         |
| Enterprise       | NO       | PARTIAL   | YES         |
+------------------+----------+-----------+-------------+
```

---

##  Performance Visualization

```
Processing Time (100K rows)

  Fast Mode       [==========]     60s
  Config-1        [===============]  90s
  Deep Config     [====================] 120s

Memory Usage (1GB file)

  Fast Mode       [======]          2GB
  Config-1        [========]        2.5GB
  Deep Config     [=========]       3GB

Complexity

  Fast Mode       [===]             Simple
  Config-1        [======]          Moderate
  Deep Config     [=========]       Advanced

Features

  Fast Mode       [====]            Basic
  Config-1        [========]        Extended
  Deep Config     [============]    Complete
```

---

##  Quick Start Commands

```bash
# ============================================================
# SETUP (One-time)
# ============================================================

# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Start API server
python main.py
# → Server running on http://127.0.0.1:8001

# 3. Test health
curl http://127.0.0.1:8001/health
# → {"status": "healthy", ...}

# ============================================================
# FAST MODE (Simplest - 3 commands)
# ============================================================

# 1. Process file
curl -X POST http://127.0.0.1:8001/run_fast \
  -F "file=@data.csv" \
  -F "use_turbo=true"

# 2. Search
curl -X POST http://127.0.0.1:8001/retrieve \
  -F "query=your search query" \
  -F "k=5"

# 3. Export
curl http://127.0.0.1:8001/export/chunks --output results.csv

# ============================================================
# CONFIG-1 MODE (Custom - 3 commands)
# ============================================================

# 1. Process with custom settings
curl -X POST http://127.0.0.1:8001/run_config1 \
  -F "file=@data.csv" \
  -F "chunk_method=document" \
  -F "document_key_column=category" \
  -F "model_choice=all-MiniLM-L6-v2" \
  -F "storage_choice=faiss" \
  -F "use_turbo=true"

# 2. Search
curl -X POST http://127.0.0.1:8001/retrieve \
  -F "query=your query" \
  -F "k=5"

# 3. Export
curl http://127.0.0.1:8001/export/chunks --output results.csv

# ============================================================
# DEEP CONFIG MODE (Advanced - 11 commands)
# ============================================================

# 1. Preprocess
curl -X POST http://127.0.0.1:8001/deep_config/preprocess \
  -F "file=@data.csv"

# 2. Convert types
curl -X POST http://127.0.0.1:8001/deep_config/type_convert \
  -F 'type_conversions={"id":"integer","price":"float"}'

# 3. Handle nulls
curl -X POST http://127.0.0.1:8001/deep_config/null_handle \
  -F 'null_strategies={"price":"mean","name":"mode"}'

# 4. Remove duplicates
curl -X POST http://127.0.0.1:8001/deep_config/duplicates \
  -F "strategy=keep_first"

# 5. Remove stopwords
curl -X POST http://127.0.0.1:8001/deep_config/stopwords \
  -F "remove_stopwords=true"

# 6. Normalize text
curl -X POST http://127.0.0.1:8001/deep_config/normalize \
  -F "text_processing=lemmatize"

# 7. Chunk with metadata
curl -X POST http://127.0.0.1:8001/deep_config/chunk \
  -F "chunk_method=document" \
  -F "key_column=category" \
  -F "store_metadata=true" \
  -F 'selected_numeric_columns=["price"]' \
  -F 'selected_categorical_columns=["category"]'

# 8. Generate embeddings
curl -X POST http://127.0.0.1:8001/deep_config/embed \
  -F "model_name=paraphrase-MiniLM-L6-v2" \
  -F "use_parallel=true"

# 9. Store vectors
curl -X POST http://127.0.0.1:8001/deep_config/store \
  -F "storage_type=faiss"

# 10. Search with metadata filter
curl -X POST http://127.0.0.1:8001/retrieve_with_metadata \
  -F "query=your query" \
  -F "k=5" \
  -F 'metadata_filter={"category_mode":"Electronics"}'

# 11. Export
curl http://127.0.0.1:8001/export/chunks --output results.csv
```

---

##  Documentation Quick Links

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **API_USAGE_GUIDE.md**  | Complete reference | Start here |
| installation.md | Setup guide | First time setup |
| quickstart.md | 5-min tutorial | Learn basics |
| core-endpoints.md | API reference | Look up endpoints |
| python-examples.md | Python code | Integration |
| data-flow.md | Architecture | Understand system |

---

##  Learning Path

```
DAY 1: Basics
 1. Read API_USAGE_GUIDE.md (30 min)
 2. Install & start server (15 min)
 3. Try Fast Mode (15 min)
 4. Test retrieval (10 min)
   Total: ~70 minutes

DAY 2-3: Intermediate
 1. Try Config-1 Mode (30 min)
 2. Test database import (30 min)
 3. Experiment with chunking (45 min)
 4. Try metadata filtering (30 min)
   Total: ~2.5 hours

DAY 4-7: Advanced
 1. Deep Config Mode (1 hour)
 2. Optimize metadata (1 hour)
 3. Build integration (2 hours)
 4. Production deployment (2 hours)
   Total: ~6 hours


TOTAL LEARNING TIME: ~10 hours
```

---

##  Project Submission Checklist

```
DOCUMENTATION
 [] API_USAGE_GUIDE.md complete
 [] API-DOCUMENTATION/ folder structured
 [] All endpoints documented
 [] Examples provided (Python, cURL)
 [] Architecture diagrams included
 [] Visual guides created

FUNCTIONALITY
 [] Fast Mode working
 [] Config-1 Mode working
 [] Deep Config Mode (all 9 steps)
 [] Retrieval working
 [] Metadata filtering working
 [] Database import working
 [] Export functions working
 [] Error handling tested

TESTING
 [] Sample data processed
 [] All endpoints tested
 [] Performance measured
 [] Error cases handled
 [] Large files tested (3GB+)

CODE QUALITY
 [] Code documented
 [] Functions commented
 [] Error messages clear
 [] Logging implemented
 [] Best practices followed

SUBMISSION READY
 [] README.md complete
 [] Requirements.txt up to date
 [] Test datasets included
 [] Setup instructions clear
 [] Demo script prepared
```

---

** Documentation Complete!**

For company submission, provide:
1. **API_USAGE_GUIDE.md** (main reference)
2. **API_VISUAL_GUIDE.md** (this file - visual overview)
3. **API-DOCUMENTATION/** folder (detailed docs)
4. **Source code** (main.py, backend.py, app.py)
5. **Test datasets** (TEST_DATASETS_README.md)

**Total Documentation:** 10+ files, 50+ pages, fully comprehensive!

