# RAG-Enhanced Chunking Optimizer: Gemini LLM Integration Plan

## Execution TODOs (2-hour sprint)

- [in_progress] Install Gemini SDK and update requirements.txt
  - Command: `pip install google-generativeai`
  - Add to `requirements.txt`: `google-generativeai>=0.3.0`

- [pending] Add GeminiClient to `backend.py` with `get_gemini_client()`
  - Defaults (approved): model `gemini-2.0-flash-lite`, temperature `0.3`, max_output_tokens `1024`

- [pending] Add context packing utilities to `backend.py`
  - `pack_context_for_llm`, `build_prompt`, `detect_task_from_query`, `compute_campaign_facts`

- [pending] Implement `llm_answer(query, use_campaign)` in `backend.py`
  - Reuse existing retrieval, pack context, call Gemini, return answer + sources

- [pending] Expose `/llm/answer` and `/campaign/llm_answer` in `main.py`
  - Simple Form body: `query`

- [pending] Add `call_llm_answer_api()` and LLM mode UI to `app.py`
  - Toggle between "Search Only" and "LLM Answer (Gemini)"

- [pending] Set `GEMINI_API_KEY` environment variable and run smoke tests
  - PowerShell: `$env:GEMINI_API_KEY="<your_key>"`
  - `curl -X POST http://127.0.0.1:8001/llm/answer -F "query=test"`

- [pending] Document and verify fallback to retrieval-only
  - If LLM fails, show retrieval results with friendly message

Stop points for critical approvals (will ask only if needed):
- Model change from Flash Lite to another variant (requires approval)
- Enabling streaming endpoints (optional; approval required to add)
- Moving code into `backend/llm` package (post-sprint refactor)

## Project Overview
Add Gemini LLM-powered answer generation to existing chunking optimizer while preserving current search/retrieval functionality.

**Timeline**: 2 hours  
**API**: Google Gemini Flash 2.0/2.5  
**API Key**: `AIzaSyCG7AA2m2E8u8ByY-hWPYwyNb800tU2MNs`  
**Approach**: Minimal viable implementation, query-only UI, backend-managed config

---

## Architecture Summary

### Current System (Keep Intact)
```
Upload CSV/DB → Preprocess → Chunk → Embed → Store (FAISS/Chroma) → Retrieve (top-k)
```

### New LLM Path (Add Alongside)
```
Query → Retrieve (existing) → Pack Context → Gemini Generate → Answer + Sources
```

### Key Principles
- ✅ Keep all existing endpoints and functionality
- ✅ Add new `/llm/answer` and `/campaign/llm_answer` endpoints
- ✅ UI only sends query; backend handles all config
- ✅ Graceful fallback if LLM fails → return retrieval results

---

## Implementation Checklist (2 Hours)

### Hour 1: Backend Implementation

#### Task 1.1: Install Gemini SDK (5 min)
```bash
pip install google-generativeai
```

Add to `requirements.txt`:
```
google-generativeai>=0.3.0
```

#### Task 1.2: Add Gemini Client to `backend.py` (15 min)
**Location**: Add at end of `backend.py` before exports section

```python
# -----------------------------
# 🔹 GEMINI LLM CLIENT
# -----------------------------
import google.generativeai as genai

class GeminiClient:
    """Wrapper for Google Gemini API"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-lite", 
                 temperature: float = 0.3, max_output_tokens: int = 1024):
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Configure API
        genai.configure(api_key=api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model)
        
        # Safety settings
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
        }
    
    def generate(self, prompt: str) -> Tuple[str, Dict]:
        """Generate answer from prompt"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
                safety_settings=self.safety_settings
            )
            
            answer = response.text
            
            # Estimate usage (Gemini SDK doesn't always return exact counts)
            usage = {
                "prompt_tokens": len(prompt.split()) * 1.3,  # Rough estimate
                "output_tokens": len(answer.split()) * 1.3,
                "model": self.model_name
            }
            
            return answer, usage
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count"""
        return int(len(text.split()) * 1.3)

# Global Gemini client (initialized on first use)
_gemini_client = None

def get_gemini_client():
    """Get or create Gemini client"""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyCG7AA2m2E8u8ByY-hWPYwyNb800tU2MNs")
        _gemini_client = GeminiClient(api_key, model="gemini-2.0-flash-lite")
    return _gemini_client
```

#### Task 1.3: Add Context Packer to `backend.py` (15 min)
**Location**: Add after GeminiClient

```python
# -----------------------------
# 🔹 CONTEXT PACKER FOR LLM
# -----------------------------
def pack_context_for_llm(chunks_results: List[Dict], token_budget: int = 8000, 
                         add_facts: bool = False, facts: Dict = None) -> str:
    """
    Pack retrieval results into LLM context
    - Deduplicate similar chunks
    - Snippetize to save tokens
    - Enforce token budget
    """
    packed_parts = []
    current_tokens = 0
    
    # Add facts if provided
    if add_facts and facts:
        facts_text = "Dataset Statistics:\n"
        for key, value in facts.items():
            facts_text += f"- {key}: {value}\n"
        facts_text += "\n"
        
        facts_tokens = estimate_token_count(facts_text)
        if facts_tokens < token_budget * 0.2:  # Max 20% for facts
            packed_parts.append(facts_text)
            current_tokens += facts_tokens
    
    # Add chunks (snippetized)
    chunks_added = 0
    for result in chunks_results:
        if chunks_added >= 10:  # Max 10 chunks
            break
            
        content = result.get("content", "")
        similarity = result.get("similarity", 0)
        
        # Snippetize: take first 3 sentences or 200 chars
        sentences = content.split('. ')
        if len(sentences) > 3:
            snippet = '. '.join(sentences[:3]) + '...'
        else:
            snippet = content[:500] + ('...' if len(content) > 500 else '')
        
        snippet_tokens = estimate_token_count(snippet)
        
        # Check token budget
        if current_tokens + snippet_tokens > token_budget:
            break
        
        # Format as source
        source_text = f"\n[Source {chunks_added + 1}] (Similarity: {similarity:.2f})\n{snippet}\n"
        packed_parts.append(source_text)
        current_tokens += snippet_tokens
        chunks_added += 1
    
    return "".join(packed_parts)

def build_prompt(query: str, context: str, task: str = "qa") -> str:
    """Build prompt for Gemini based on task type"""
    
    if task == "qa":
        return f"""You are a helpful assistant. Answer the question based ONLY on the provided context.

IMPORTANT RULES:
- Use only information from the context below
- If the answer is not in the context, say "I don't have enough information to answer that."
- Cite specific sources when possible
- Be concise and factual

Context:
{context}

Question: {query}

Answer:"""
    
    elif task == "summarize":
        return f"""You are a data analyst. Provide a concise summary of the dataset based on the provided information.

Context:
{context}

Task: Summarize the key insights, trends, and important entities from this data.

Summary:"""
    
    elif task == "insights":
        return f"""You are a business strategist analyzing campaign data. Identify opportunities and provide actionable recommendations.

Context:
{context}

Task: Based on this campaign data, identify the top opportunities with reasons and recommended next actions.

Analysis:"""
    
    else:  # default qa
        return f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

def detect_task_from_query(query: str) -> str:
    """Auto-detect task type from query"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['summarize', 'summary', 'overview', 'what is this data']):
        return "summarize"
    elif any(word in query_lower for word in ['opportunities', 'insights', 'recommend', 'strategy', 'improve']):
        return "insights"
    else:
        return "qa"

def compute_campaign_facts(df: pd.DataFrame = None) -> Dict:
    """Compute quick stats for campaign data"""
    global current_df, current_media_campaign_data
    
    facts = {}
    
    # Use preprocessed campaign data if available
    if current_media_campaign_data:
        df = current_media_campaign_data.get('processed_df')
        field_mapping = current_media_campaign_data.get('field_mapping', {})
    elif current_df is not None:
        df = current_df
        field_mapping = {}
    else:
        return facts
    
    if df is None or df.empty:
        return facts
    
    try:
        facts['total_records'] = len(df)
        
        # Find status column
        status_cols = [col for col, ftype in field_mapping.items() if ftype == 'lead_status']
        if status_cols and status_cols[0] in df.columns:
            status_counts = df[status_cols[0]].value_counts().to_dict()
            for status, count in list(status_counts.items())[:5]:
                facts[f'{status}_count'] = int(count)
        
        # Find source column
        source_cols = [col for col, ftype in field_mapping.items() if ftype == 'lead_source']
        if source_cols and source_cols[0] in df.columns:
            source_counts = df[source_cols[0]].value_counts().to_dict()
            facts['top_source'] = str(list(source_counts.keys())[0]) if source_counts else 'N/A'
        
        # Find company column
        company_cols = [col for col, ftype in field_mapping.items() if ftype in ['company', 'company_name']]
        if company_cols and company_cols[0] in df.columns:
            facts['unique_companies'] = int(df[company_cols[0]].nunique())
    
    except Exception as e:
        logger.warning(f"Could not compute facts: {e}")
    
    return facts
```

#### Task 1.4: Add LLM Answer Function to `backend.py` (15 min)
**Location**: Add after context packer

```python
# -----------------------------
# 🔹 LLM-POWERED RETRIEVAL
# -----------------------------
def llm_answer(query: str, use_campaign: bool = False) -> Dict:
    """
    Generate LLM-powered answer using existing retrieval + Gemini
    
    Args:
        query: User question
        use_campaign: Use campaign smart retrieval if True
    
    Returns:
        {
            "answer": str,
            "sources": List[Dict],
            "facts": Dict (optional),
            "usage": Dict,
            "retrieval": Dict,
            "model_info": Dict
        }
    """
    start_time = time.time()
    
    try:
        # Step 1: Auto-detect task type
        task = detect_task_from_query(query)
        logger.info(f"LLM answer request - Query: '{query}', Task: {task}, Campaign: {use_campaign}")
        
        # Step 2: Retrieval (use existing functions)
        k = 12 if task in ["summarize", "insights"] else 7
        
        if use_campaign:
            # Use campaign retrieval from backend_campaign
            from backend_campaign import campaign_smart_retrieval
            retrieval_results = campaign_smart_retrieval(query, "auto", k, True)
        else:
            # Use standard retrieval
            retrieval_results = retrieve_similar(query, k)
        
        # Handle retrieval errors
        if "error" in retrieval_results:
            return {
                "error": retrieval_results["error"],
                "fallback_mode": "retrieval_only",
                "sources": []
            }
        
        # Step 3: Extract chunks from results
        chunks_results = retrieval_results.get("results", [])
        
        if not chunks_results:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": [],
                "retrieval": {"k": k, "found": 0},
                "model_info": {"name": "none", "reason": "no_relevant_chunks"}
            }
        
        # Step 4: Compute facts for summarize/insights
        facts = None
        add_facts = task in ["summarize", "insights"]
        if add_facts:
            facts = compute_campaign_facts()
        
        # Step 5: Pack context
        context = pack_context_for_llm(chunks_results, token_budget=8000, 
                                      add_facts=add_facts, facts=facts)
        
        # Step 6: Build prompt
        prompt = build_prompt(query, context, task)
        
        # Step 7: Generate with Gemini
        gemini = get_gemini_client()
        answer, usage = gemini.generate(prompt)
        
        # Step 8: Format response
        # Prepare sources (limit to top 5 for display)
        sources = []
        for i, result in enumerate(chunks_results[:5]):
            sources.append({
                "rank": i + 1,
                "similarity": float(result.get("similarity", 0)),
                "snippet": result.get("content", "")[:300] + "..." if len(result.get("content", "")) > 300 else result.get("content", ""),
                "metadata": result.get("metadata", {})
            })
        
        response = {
            "answer": answer,
            "sources": sources,
            "facts": facts if add_facts else None,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("output_tokens", 0)),
                "total_tokens": int(usage.get("prompt_tokens", 0) + usage.get("output_tokens", 0))
            },
            "retrieval": {
                "k": k,
                "found": len(chunks_results),
                "method": retrieval_results.get("retrieval_method", "semantic"),
                "store_type": current_store_info.get("type", "unknown") if current_store_info else "unknown"
            },
            "model_info": {
                "name": "gemini-2.0-flash-lite",
                "temperature": 0.3,
                "task_detected": task
            },
            "processing_time": f"{time.time() - start_time:.2f}s"
        }
        
        return serialize_data(response) if 'serialize_data' in dir() else response
    
    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}")
        # Fallback to retrieval-only
        return {
            "error": f"LLM generation failed: {str(e)}",
            "fallback_mode": "retrieval_only",
            "sources": chunks_results[:5] if 'chunks_results' in locals() else [],
            "message": "Showing search results instead of generated answer"
        }
```

#### Task 1.5: Add `serialize_data` to `backend.py` if missing (5 min)
**Check if exists**: Search for `serialize_data` in `backend.py`

If NOT found, add this at the top after imports:

```python
# -----------------------------
# 🔹 JSON Serialization Helper
# -----------------------------
def convert_to_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types"""
    if pd.isna(obj):
        return None
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    else:
        return obj

def serialize_data(data):
    """Recursively serialize data for JSON response"""
    if isinstance(data, dict):
        return {k: serialize_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_data(item) for item in data]
    elif isinstance(data, tuple):
        return [serialize_data(item) for item in data]
    else:
        return convert_to_serializable(data)
```

#### Task 1.6: Add LLM Endpoints to `main.py` (20 min)
**Location**: Add before the `if __name__ == "__main__":` line

```python
# ---------------------------
# 🔹 NEW: LLM-POWERED ANSWER GENERATION
# ---------------------------
@app.post("/llm/answer")
async def llm_answer_endpoint(
    query: str = Form(...)
):
    """
    Generate LLM-powered answer using Gemini
    - Uses existing retrieval
    - Packs context intelligently
    - Returns answer + sources
    """
    try:
        from backend import llm_answer
        result = llm_answer(query, use_campaign=False)
        return result
    except Exception as e:
        return {"error": f"LLM answer generation failed: {str(e)}"}

@app.post("/campaign/llm_answer")
async def campaign_llm_answer_endpoint(
    query: str = Form(...)
):
    """
    Campaign-specific LLM answer with smart company retrieval
    """
    try:
        from backend import llm_answer
        result = llm_answer(query, use_campaign=True)
        return result
    except Exception as e:
        return {"error": f"Campaign LLM answer failed: {str(e)}"}
```

### Hour 2: UI Integration & Testing

#### Task 2.1: Update `app.py` - Add LLM Mode Toggle (15 min)
**Location**: In the retrieval section (around line 1340)

Find this section:
```python
# Enhanced Retrieval Section with SMART COLUMN DISPLAY
st.markdown("---")
st.markdown("## 🎯 Campaign Data Retrieval")
```

Replace the retrieval section with:

```python
# Enhanced Retrieval Section with LLM MODE
st.markdown("---")
st.markdown("## 🎯 Campaign Data Retrieval")

# Add LLM mode toggle
st.markdown("### 🤖 Retrieval Mode")
retrieval_mode = st.radio(
    "Select retrieval mode:",
    ["🔍 Search Only (Show Chunks)", "🤖 LLM Answer (Gemini)"],
    help="Search Only: Returns matching chunks. LLM Answer: Generates natural language answer using Gemini AI."
)

col1, col2, col3 = st.columns([3, 2, 1])

with col1:
    query = st.text_area("Search Campaign Data:", 
                       placeholder="Ask a question or search for companies, leads, campaigns...",
                       height=100)

with col2:
    if retrieval_mode == "🔍 Search Only (Show Chunks)":
        search_field = st.selectbox("Search Field:", 
                                  ["all", "company", "lead_source", "lead_status", "campaign_source"],
                                  help="Search in specific fields or all fields")
        
        k = st.number_input("Number of results:", 1, 50, 5)
    else:
        # LLM mode - no params needed
        st.info("🤖 LLM will auto-configure retrieval")
        search_field = "all"
        k = 7

with col3:
    if retrieval_mode == "🔍 Search Only (Show Chunks)":
        include_complete = st.checkbox("Complete Records", value=True,
                                     help="Return full contact records")
    else:
        include_complete = True  # Always true for LLM
    
    st.markdown("")
    if st.button("🔍 Search Campaign" if retrieval_mode == "🔍 Search Only (Show Chunks)" else "🤖 Ask Gemini", 
                 use_container_width=True):
        if query:
            with st.spinner("Processing..." if retrieval_mode == "🔍 Search Only (Show Chunks)" else "Thinking with Gemini AI..."):
                try:
                    if retrieval_mode == "🔍 Search Only (Show Chunks)":
                        # Use existing search retrieval
                        if st.session_state.use_smart_retrieval:
                            result = call_smart_company_retrieval_api(
                                query, search_field, k, include_complete
                            )
                        else:
                            result = call_media_campaign_retrieval_api(
                                query, search_field, k, include_complete
                            )
                        
                        st.session_state.media_campaign_results = result
                        st.session_state.process_status["retrieval"] = "completed"
                        
                        if 'error' in result:
                            st.error(f"Retrieval error: {result['error']}")
                        else:
                            retrieval_method = result.get('retrieval_method', 'standard')
                            if retrieval_method == 'company_keyword_matching':
                                exact_matches = result.get('exact_matches', 0)
                                partial_matches = result.get('partial_matches', 0)
                                st.success(f"✅ SMART Retrieval found {exact_matches} exact + {partial_matches} partial company matches!")
                            else:
                                st.success(f"✅ Found {len(result.get('results', []))} campaign matches!")
                    
                    else:
                        # Use new LLM answer endpoint
                        result = call_llm_answer_api(query)
                        
                        st.session_state.llm_answer_results = result
                        st.session_state.process_status["retrieval"] = "completed"
                        
                        if 'error' in result:
                            st.error(f"LLM error: {result['error']}")
                            # Show fallback sources if available
                            if result.get('sources'):
                                st.warning("Showing retrieval results instead:")
                                st.session_state.media_campaign_results = {
                                    "results": result['sources'],
                                    "query": query
                                }
                        else:
                            st.success(f"✅ Gemini answered in {result.get('processing_time', 'N/A')}!")
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a query first")
    
    if st.button("🧹 Clear Results", use_container_width=True):
        st.session_state.media_campaign_results = None
        st.session_state.llm_answer_results = None
```

#### Task 2.2: Add LLM API Call Function to `app.py` (5 min)
**Location**: Add after existing API call functions (around line 482)

```python
def call_llm_answer_api(query: str):
    """Call LLM answer endpoint for Gemini-powered responses"""
    data = {
        "query": query
    }
    response = requests.post(f"{API_BASE_URL}/campaign/llm_answer", data=data)
    return response.json()
```

#### Task 2.3: Add LLM Results Display to `app.py` (15 min)
**Location**: After the retrieval results display section

Find this section:
```python
# Display media campaign retrieval results - USING FIXED FUNCTION
if st.session_state.media_campaign_results:
    results = st.session_state.media_campaign_results
    display_retrieval_results(results)
```

Replace with:

```python
# Display results based on mode
if st.session_state.get('llm_answer_results'):
    # Display LLM answer
    display_llm_answer_results(st.session_state.llm_answer_results)
elif st.session_state.media_campaign_results:
    # Display regular retrieval results
    results = st.session_state.media_campaign_results
    display_retrieval_results(results)
```

Add the new display function before the footer (around line 1400):

```python
def display_llm_answer_results(results):
    """Display LLM-generated answer with sources"""
    if 'error' in results:
        st.error(f"❌ {results['error']}")
        
        # Show fallback sources if available
        if results.get('fallback_mode') == 'retrieval_only' and results.get('sources'):
            st.warning("⚠️ Showing search results instead:")
            for i, source in enumerate(results['sources']):
                with st.expander(f"📄 Source {i+1} (Similarity: {source.get('similarity', 0):.3f})", expanded=False):
                    st.text_area("Content", value=source.get('content', ''), height=200, disabled=True)
        return
    
    # Display answer
    st.markdown("### 🤖 Gemini AI Answer")
    
    answer = results.get('answer', 'No answer generated')
    
    # Answer card with styling
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); 
                border-left: 6px solid #48bb78; 
                padding: 25px; 
                border-radius: 12px; 
                margin: 15px 0;
                box-shadow: 0 8px 25px rgba(72, 187, 120, 0.2);">
        <div style="color: #48bb78; font-weight: bold; margin-bottom: 10px; font-size: 1.1em;">
            ✨ AI-Generated Answer
        </div>
        <div style="color: #e2e8f0; line-height: 1.8; white-space: pre-wrap;">
{answer}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show metadata
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_info = results.get('model_info', {})
        st.metric("Model", model_info.get('name', 'N/A'))
        if model_info.get('task_detected'):
            st.caption(f"Task: {model_info['task_detected']}")
    
    with col2:
        usage = results.get('usage', {})
        total_tokens = usage.get('total_tokens', 0)
        st.metric("Tokens Used", f"{total_tokens:,}")
        st.caption(f"In: {usage.get('prompt_tokens', 0)} | Out: {usage.get('output_tokens', 0)}")
    
    with col3:
        retrieval = results.get('retrieval', {})
        st.metric("Sources Found", retrieval.get('found', 0))
        st.caption(f"Method: {retrieval.get('method', 'N/A')}")
    
    # Display sources
    st.markdown("### 📚 Sources Used")
    sources = results.get('sources', [])
    
    if sources:
        for source in sources:
            similarity_color = "#48bb78" if source.get('similarity', 0) > 0.7 else "#ed8936" if source.get('similarity', 0) > 0.4 else "#f56565"
            
            with st.expander(f"📄 Source {source.get('rank', 0)} (Similarity: {source.get('similarity', 0):.3f})", expanded=False):
                st.markdown(f"""
                <div style="border-left: 4px solid {similarity_color}; padding-left: 15px;">
                """, unsafe_allow_html=True)
                
                st.text_area(
                    "Content",
                    value=source.get('snippet', source.get('content', '')),
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )
                
                # Show metadata if available
                if source.get('metadata'):
                    st.json(source['metadata'])
                
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No sources available")
    
    # Show facts if available
    facts = results.get('facts')
    if facts:
        with st.expander("📊 Dataset Statistics Used", expanded=False):
            st.json(facts)
```

#### Task 2.4: Add Session State for LLM Results (2 min)
**Location**: In session state initialization (around line 726)

Add:
```python
if "llm_answer_results" not in st.session_state:
    st.session_state.llm_answer_results = None
```

#### Task 2.5: Set Environment Variable (1 min)
**Option A**: Set in terminal before running:
```bash
# Windows PowerShell
$env:GEMINI_API_KEY="AIzaSyCG7AA2m2E8u8ByY-hWPYwyNb800tU2MNs"

# Windows CMD
set GEMINI_API_KEY=AIzaSyCG7AA2m2E8u8ByY-hWPYwyNb800tU2MNs

# Linux/Mac
export GEMINI_API_KEY="AIzaSyCG7AA2m2E8u8ByY-hWPYwyNb800tU2MNs"
```

**Option B**: Hardcode in `backend.py` GeminiClient (for quick testing only):
```python
api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyCG7AA2m2E8u8ByY-hWPYwyNb800tU2MNs")
```

#### Task 2.6: Testing (10 min)

**Test Plan**:

1. **Start API**:
```bash
python main.py
```

2. **Test LLM endpoint directly**:
```bash
curl -X POST http://127.0.0.1:8001/llm/answer \
  -F "query=What is this data about?"
```

Expected response:
```json
{
  "answer": "Based on the provided context...",
  "sources": [...],
  "usage": {...},
  "model_info": {...}
}
```

3. **Start Streamlit**:
```bash
streamlit run app.py
```

4. **Upload a CSV and run campaign pipeline**
5. **Switch to "LLM Answer" mode**
6. **Test queries**:
   - "Summarize this campaign data"
   - "What are the top opportunities?"
   - "Show me leads from Company XYZ"

#### Task 2.7: Fix Import Issues (if any) (5 min)

If you get `ModuleNotFoundError` for `backend_campaign`:

In `backend.py`, add fallback for campaign retrieval:

```python
def llm_answer(query: str, use_campaign: bool = False) -> Dict:
    # ... existing code ...
    
    if use_campaign:
        # Try campaign retrieval
        try:
            from backend_campaign import campaign_smart_retrieval
            retrieval_results = campaign_smart_retrieval(query, "auto", k, True)
        except ImportError:
            # Fallback to standard retrieval
            logger.warning("Campaign backend not available, using standard retrieval")
            retrieval_results = retrieve_similar(query, k)
    else:
        retrieval_results = retrieve_similar(query, k)
```

---

## Implementation Steps (Execution Order)

### Step 1: Install Dependencies (2 min)
```bash
pip install google-generativeai
```

### Step 2: Backend Changes (40 min)
1. Open `backend.py`
2. Add `serialize_data` helper (if missing) - scroll to top after imports
3. Add `GeminiClient` class - scroll to end, before exports section
4. Add `ContextPacker` functions (pack_context_for_llm, build_prompt, etc.)
5. Add `llm_answer` function
6. Save file

### Step 3: API Endpoint Changes (15 min)
1. Open `main.py`
2. Scroll to end (before `if __name__ == "__main__"`)
3. Add `/llm/answer` and `/campaign/llm_answer` endpoints
4. Save file

### Step 4: UI Changes (40 min)
1. Open `app.py`
2. Add `call_llm_answer_api` function (after line 482)
3. Add `llm_answer_results` to session state (around line 726)
4. Replace retrieval section with LLM mode toggle (around line 1340)
5. Add `display_llm_answer_results` function (before footer, around line 1400)
6. Save file

### Step 5: Set API Key (1 min)
```bash
$env:GEMINI_API_KEY="AIzaSyCG7AA2m2E8u8ByY-hWPYwyNb800tU2MNs"
```

### Step 6: Test (22 min)
1. Start API: `python main.py`
2. Test endpoint: `curl -X POST http://127.0.0.1:8001/llm/answer -F "query=test"`
3. Start UI: `streamlit run app.py`
4. Upload CSV, run pipeline
5. Test LLM mode with queries
6. Verify answer + sources display

---

## Code Snippets Reference

### Full GeminiClient Implementation
```python
import google.generativeai as genai
from typing import Tuple, Dict

class GeminiClient:
    """Wrapper for Google Gemini API"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-lite", 
                 temperature: float = 0.3, max_output_tokens: int = 1024):
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
        }
    
    def generate(self, prompt: str) -> Tuple[str, Dict]:
        """Generate answer from prompt"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
                safety_settings=self.safety_settings
            )
            
            answer = response.text
            
            usage = {
                "prompt_tokens": int(len(prompt.split()) * 1.3),
                "output_tokens": int(len(answer.split()) * 1.3),
                "model": self.model_name
            }
            
            return answer, usage
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise

_gemini_client = None

def get_gemini_client():
    """Get or create Gemini client"""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyCG7AA2m2E8u8ByY-hWPYwyNb800tU2MNs")
        _gemini_client = GeminiClient(api_key, model="gemini-2.0-flash-lite")
    return _gemini_client
```

---

## Testing Checklist

### Backend Tests
- [ ] API starts without errors: `python main.py`
- [ ] Health check works: `curl http://127.0.0.1:8001/health`
- [ ] LLM endpoint exists: `curl -X POST http://127.0.0.1:8001/llm/answer -F "query=test"`
- [ ] Campaign LLM endpoint exists: `curl -X POST http://127.0.0.1:8001/campaign/llm_answer -F "query=test"`

### UI Tests
- [ ] Streamlit starts: `streamlit run app.py`
- [ ] File upload works
- [ ] Campaign pipeline runs (any mode)
- [ ] "LLM Answer" mode toggle appears
- [ ] Query submission works in LLM mode
- [ ] Answer displays correctly
- [ ] Sources display correctly
- [ ] Usage stats display correctly
- [ ] Fallback to retrieval works if LLM fails

### Functional Tests
- [ ] QA query: "What companies are in this data?"
- [ ] Summarize query: "Summarize this dataset"
- [ ] Insights query: "What are the top opportunities?"
- [ ] Campaign query: "Show leads from Company ABC"
- [ ] Smart retrieval: "Company XYZ" (should use exact match first)

---

## Troubleshooting

### Issue: Import error for google.generativeai
**Fix**: 
```bash
pip install --upgrade google-generativeai
```

### Issue: API key error
**Fix**: Verify environment variable
```bash
# Windows PowerShell
echo $env:GEMINI_API_KEY

# Linux/Mac
echo $GEMINI_API_KEY
```

Or hardcode temporarily in `backend.py` for testing.

### Issue: "No model available" error
**Fix**: Make sure you've run a processing pipeline first (Fast/Config/Campaign mode) before trying retrieval or LLM answers.

### Issue: LLM returns empty answer
**Fix**: 
- Check if retrieval returned results
- Check Gemini API quotas/limits
- Verify API key is valid
- Check logs for safety blocks

### Issue: UI doesn't show LLM toggle
**Fix**: 
- Clear browser cache
- Restart Streamlit
- Check if session state initialization is correct

### Issue: Campaign endpoints not found
**Fix**: Make sure you're calling `/campaign/llm_answer` not `/llm/answer` for campaign mode

---

## Expected Behavior

### Search Only Mode (Existing - Unchanged)
1. User enters query
2. Returns top-k chunks with similarity scores
3. Displays expandable chunks with metadata
4. Complete records shown for campaign data

### LLM Answer Mode (New)
1. User enters query
2. Backend auto-detects task (qa/summarize/insights)
3. Retrieves relevant chunks (7-12 based on task)
4. Packs context with optional facts
5. Sends to Gemini with task-specific prompt
6. Returns:
   - Natural language answer (highlighted card)
   - Top 5 sources used (expandable)
   - Token usage stats
   - Processing time

### Example Queries & Expected Outputs

**Query**: "Summarize this campaign data"
- **Task Detected**: summarize
- **Retrieval**: diverse (samples across companies/sources)
- **Facts Added**: Yes (total leads, status counts, top source)
- **Answer**: "This campaign dataset contains 1,500 leads from 342 companies. The primary lead source is 'Website' (45%), followed by 'Referral' (30%). Most leads are in 'New' status (60%), with 234 marked as 'Hot' opportunities..."
- **Sources**: 5-10 diverse chunks

**Query**: "What are the top opportunities?"
- **Task Detected**: insights
- **Retrieval**: smart (exact company match → semantic)
- **Facts Added**: Yes
- **Answer**: "Top opportunities:\n1. Company A - High engagement (85% open rate), recent activity within 2 days\n2. Company B - Budget confirmed, decision maker identified..."
- **Sources**: 5-7 relevant chunks

**Query**: "Show leads from Company XYZ"
- **Task Detected**: qa
- **Retrieval**: smart (exact company match first)
- **Facts Added**: No
- **Answer**: "Company XYZ has 3 leads in the dataset:\n1. John Doe (john@xyz.com) - Hot lead, Website source\n2. Jane Smith (jane@xyz.com) - Warm lead, Referral..."
- **Sources**: Exact company matches

---

## Configuration Reference

### Hardcoded Defaults (No Config File Needed for 2-Hour Timeline)

**In `backend.py` → `llm_answer` function**:
```python
# Defaults
k = 12 if task in ["summarize", "insights"] else 7
token_budget = 8000
temperature = 0.3
max_output_tokens = 1024
model = "gemini-2.0-flash-lite"
```

**Task Detection Logic**:
```python
def detect_task_from_query(query: str) -> str:
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['summarize', 'summary', 'overview', 'what is this data']):
        return "summarize"
    elif any(word in query_lower for word in ['opportunities', 'insights', 'recommend', 'strategy', 'improve', 'leads']):
        return "insights"
    else:
        return "qa"
```

**Retrieval Strategy Selection**:
```python
# In llm_answer function:
if use_campaign:
    # Campaign mode → always use smart retrieval
    retrieval_results = campaign_smart_retrieval(query, "auto", k, True)
else:
    # Standard mode → semantic retrieval
    retrieval_results = retrieve_similar(query, k)
```

---

## File Modifications Summary

### `backend.py` Changes
**Add at end, before export functions**:
1. `serialize_data` helper (if missing)
2. `GeminiClient` class
3. `get_gemini_client()` function
4. `pack_context_for_llm()` function
5. `build_prompt()` function
6. `detect_task_from_query()` function
7. `compute_campaign_facts()` function
8. `llm_answer()` function

**Estimated lines added**: ~300 lines

### `main.py` Changes
**Add before `if __name__ == "__main__"`**:
1. `/llm/answer` endpoint
2. `/campaign/llm_answer` endpoint

**Estimated lines added**: ~30 lines

### `app.py` Changes
**Changes**:
1. Add `call_llm_answer_api()` function (after line 482)
2. Add `llm_answer_results` session state (around line 726)
3. Replace retrieval section with mode toggle (around line 1340)
4. Add `display_llm_answer_results()` function (before footer)
5. Update result display logic to handle both modes

**Estimated lines added/modified**: ~150 lines

### `requirements.txt` Changes
**Add**:
```
google-generativeai>=0.3.0
```

---

## Quick Reference: Key Functions

### Backend Functions
```python
# Gemini client
get_gemini_client() → GeminiClient

# Context packing
pack_context_for_llm(chunks, budget, facts) → str

# Prompt building
build_prompt(query, context, task) → str

# Task detection
detect_task_from_query(query) → "qa"|"summarize"|"insights"

# Facts computation
compute_campaign_facts() → Dict

# Main LLM function
llm_answer(query, use_campaign) → Dict
```

### API Endpoints
```python
POST /llm/answer
  - Body: { "query": str }
  - Returns: { answer, sources, usage, ... }

POST /campaign/llm_answer
  - Body: { "query": str }
  - Returns: { answer, sources, usage, ... }
```

### UI Functions
```python
# API call
call_llm_answer_api(query) → Dict

# Display
display_llm_answer_results(results) → None
```

---

## Success Criteria

### Minimal Success (Must Have)
- ✅ LLM endpoint responds without errors
- ✅ UI toggle between Search/LLM modes works
- ✅ Simple QA queries return answers
- ✅ Sources display correctly
- ✅ Fallback works if LLM fails

### Full Success (Nice to Have)
- ✅ Task auto-detection works (qa/summarize/insights)
- ✅ Campaign smart retrieval integrates with LLM
- ✅ Facts/aggregates show for summarize queries
- ✅ Token usage displays accurately
- ✅ Multiple query types work (summarize, opportunities, company search)

---

## Post-Implementation Enhancements (Future)

### Phase 2 (After 2-Hour Deadline)
- Add streaming endpoint for progressive display
- Add YAML config file for profiles
- Add caching layer (Redis)
- Add PII redaction hooks
- Add conversation history (multi-turn chat)

### Phase 3 (Production Ready)
- Add rate limiting per user
- Add cost tracking dashboard
- Add A/B testing for prompts
- Add user feedback collection (thumbs up/down)
- Add monitoring with Prometheus

---

## Diagram: Complete System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERACTION                             │
│                                                                  │
│  Step 1: Upload CSV (media campaign data)                       │
│          ↓                                                       │
│  Step 2: Run Campaign Pipeline                                  │
│          (chunk → embed → store in FAISS/Chroma)                │
│          ↓                                                       │
│  Step 3: Enter Query + Select Mode                              │
│          ┌─────────────────┬─────────────────┐                 │
│          │  Search Only    │  LLM Answer     │                 │
│          └────────┬────────┴────────┬────────┘                 │
└───────────────────┼─────────────────┼──────────────────────────┘
                    │                 │
                    ▼                 ▼
        ┌───────────────────┐  ┌──────────────────────────┐
        │ /campaign/retrieve│  │ /campaign/llm_answer     │
        └───────┬───────────┘  └──────────┬───────────────┘
                │                         │
                ▼                         ▼
        ┌────────────────┐      ┌──────────────────────────┐
        │ Return Chunks  │      │ 1. Retrieve (smart)      │
        └────────────────┘      │ 2. Pack context          │
                                │ 3. Build prompt          │
                                │ 4. Call Gemini           │
                                │ 5. Return answer+sources │
                                └──────────┬───────────────┘
                                           │
                                           ▼
                    ┌─────────────────────────────────────────┐
                    │  DISPLAY IN UI                          │
                    │                                         │
                    │  Search Mode:                           │
                    │  → Expandable chunks                    │
                    │  → Similarity scores                    │
                    │  → Complete records                     │
                    │                                         │
                    │  LLM Mode:                              │
                    │  → AI-generated answer (card)           │
                    │  → Sources used (expandable)            │
                    │  → Token usage stats                    │
                    │  → Processing time                      │
                    └─────────────────────────────────────────┘
```

---

## Environment Setup

### API Key Configuration
**Windows PowerShell** (Recommended for your system):
```powershell
$env:GEMINI_API_KEY="AIzaSyCG7AA2m2E8u8ByY-hWPYwyNb800tU2MNs"
python main.py
```

To make it permanent:
```powershell
[System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'AIzaSyCG7AA2m2E8u8ByY-hWPYwyNb800tU2MNs', 'User')
```

**Alternative**: Create `.env` file
```
GEMINI_API_KEY=AIzaSyCG7AA2m2E8u8ByY-hWPYwyNb800tU2MNs
```

Then in `backend.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Quick Start Commands

### Complete Setup (Copy-Paste)
```powershell
# Install Gemini SDK
pip install google-generativeai

# Set API key
$env:GEMINI_API_KEY="AIzaSyCG7AA2m2E8u8ByY-hWPYwyNb800tU2MNs"

# Start API server
python main.py

# In new terminal, start UI
streamlit run app.py
```

### Test Commands
```bash
# Test health
curl http://127.0.0.1:8001/health

# Test LLM endpoint
curl -X POST http://127.0.0.1:8001/llm/answer -F "query=What is this data about?"

# Test campaign LLM endpoint
curl -X POST http://127.0.0.1:8001/campaign/llm_answer -F "query=Summarize the campaign data"
```

---

## Implementation Notes

### Why Gemini Flash Lite?
- **Fast**: ~1-2s response time
- **Cost-effective**: Cheaper than Pro models
- **Good quality**: Sufficient for RAG tasks with good context
- **High rate limits**: Suitable for production

### Why Query-Only UI?
- **Simplicity**: Users don't need to understand retrieval/LLM params
- **Consistency**: All users get same quality experience
- **Maintainability**: Change behavior centrally in backend
- **Safety**: Enforce guardrails and cost limits

### Why Keep Search Mode?
- **Debugging**: See raw chunks for quality checks
- **Transparency**: Users can inspect exact matches
- **Fallback**: Always available if LLM has issues
- **Use Cases**: Some users prefer raw data over synthesized answers

---

## Final Checklist Before Deployment

### Code Quality
- [ ] All imports working (no missing modules)
- [ ] No syntax errors
- [ ] API key set in environment
- [ ] Gemini SDK installed

### Functionality
- [ ] Existing retrieval still works
- [ ] LLM endpoints return valid JSON
- [ ] UI toggle switches modes correctly
- [ ] Error handling works (try invalid query)
- [ ] Sources display properly

### User Experience
- [ ] Answer is readable and well-formatted
- [ ] Loading spinner shows during generation
- [ ] Error messages are user-friendly
- [ ] Token usage displays correctly

### Performance
- [ ] Response time < 5s for typical queries
- [ ] No memory leaks (test multiple queries)
- [ ] Large files still process correctly

---

## Next Steps After 2-Hour Implementation

### Immediate (Day 1)
- Test with real campaign data
- Collect user feedback
- Monitor Gemini API usage/costs
- Fix any critical bugs

### Short-term (Week 1)
- Add prompt tuning based on feedback
- Implement streaming for better UX
- Add conversation history (multi-turn)
- Add response caching

### Medium-term (Month 1)
- Extract config to YAML files
- Add multiple task profiles
- Implement PII redaction
- Add monitoring dashboard

---

## Cost Estimation

### Gemini Flash Lite Pricing (Approximate)
- Input: $0.000075 per 1K tokens
- Output: $0.0003 per 1K tokens

**Example Query Cost**:
- Retrieval context: 4000 tokens
- System prompt: 200 tokens
- Query: 20 tokens
- Answer: 400 tokens

Cost = (4220 × 0.000075) + (400 × 0.0003) = $0.000437 (~$0.0004 per query)

**Monthly Estimate** (1000 queries/month):
- Total cost: ~$0.44/month
- Very affordable for testing and small-scale production

---

## Support & Resources

### Gemini Documentation
- [Gemini API Quickstart](https://ai.google.dev/tutorials/python_quickstart)
- [Safety Settings](https://ai.google.dev/api/python/google/generativeai/types/SafetySetting)
- [Generation Config](https://ai.google.dev/api/python/google/generativeai/types/GenerationConfig)

### Your Project Documentation
- `API-DOCUMENTATION/` - Complete API reference
- `README.md` - Project overview
- `requirements.txt` - All dependencies

### Debug Endpoints
- `GET /health` - Check API status
- `GET /system_info` - Memory and resource usage
- `GET /capabilities` - Feature list
- `GET /debug/storage` - Storage state inspection

---

## Success Metrics

### Technical Metrics
- API response time: < 5s (target: 2-3s)
- Success rate: > 95%
- Fallback rate: < 5%
- Token efficiency: < 5000 tokens per query

### User Metrics
- Answer relevance: Subjective (user feedback)
- Citation accuracy: Sources match answer claims
- Task detection accuracy: Correct task > 80% of time
- User preference: LLM vs Search (collect feedback)

---

## Implementation Timeline (2 Hours)

```
0:00 - 0:10  Install Gemini SDK, verify API key works
0:10 - 0:40  Add GeminiClient + helpers to backend.py
0:40 - 0:55  Add LLM endpoints to main.py
0:55 - 1:30  Update app.py (toggle, API call, display)
1:30 - 1:50  Test end-to-end with sample queries
1:50 - 2:00  Fix any critical bugs, final verification
```

---

## END OF PLAN

**Status**: Ready for implementation  
**Next Action**: Start with Task 1.1 (Install Gemini SDK)  
**Estimated Completion**: 2 hours from start  

Good luck! 🚀

