# main.py - COMPLETE UPDATED VERSION
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import numpy as np
import tempfile
import os
from typing import Optional
import json
import uvicorn
import shutil
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

from backend import (
    run_fast_pipeline, 
    run_config1_pipeline, 
    run_deep_config_pipeline,
    retrieve_similar,
    export_chunks,
    export_embeddings,
    get_system_info,
    get_file_info,
    connect_mysql,
    connect_postgresql,
    get_table_list,
    import_table_to_dataframe,
    process_large_file,
    can_load_file,
    LARGE_FILE_THRESHOLD,
    get_table_size,
    import_large_table_to_dataframe,
    process_file_direct,
    EMBEDDING_BATCH_SIZE,
    PARALLEL_WORKERS,
    set_file_info,
    save_state,
    current_df,
    current_chunks,
    current_embeddings,
    current_model,
    current_store_info,
    current_file_info,
    current_metadata
)
import backend

# Campaign Mode imports
from backend_campaign import (
    run_campaign_pipeline,
    campaign_retrieve,
    campaign_smart_retrieval,
    campaign_export_chunks,
    campaign_export_embeddings,
    campaign_export_preprocessed,
    campaign_state,
    campaign_process_large_file,
    campaign_process_file_direct,
    serialize_data,
    connect_mysql as campaign_connect_mysql,
    connect_postgresql as campaign_connect_postgresql,
    get_table_list as campaign_get_table_list,
    import_table_to_dataframe as campaign_import_table,
    can_load_file as campaign_can_load_file
)

app = FastAPI(title="Chunking Optimizer API", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# NEW: LLM-POWERED ANSWER GENERATION ENDPOINTS
# Place after FastAPI app initialization
# ---------------------------
@app.post("/llm/answer")
async def llm_answer_endpoint(
    query: str = Form(...)
):
    """Generate LLM answer using standard retrieval"""
    try:
        from backend import llm_answer
        result = llm_answer(query, use_campaign=False)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"LLM answer endpoint error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"LLM answer generation failed: {str(e)}"}
        )

@app.post("/campaign/llm_answer")
async def campaign_llm_answer_endpoint(
    query: str = Form(...)
):
    """Generate LLM answer using campaign retrieval"""
    try:
        from backend import llm_answer
        result = llm_answer(query, use_campaign=True)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Campaign LLM answer endpoint error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Campaign LLM answer failed: {str(e)}"}
        )

def clear_deep_config_state():
    """Clear Deep Config data variables (not model/store)"""
    
    # Only clear data variables that will be rebuilt
    backend.current_df = None
    backend.current_chunks = None
    backend.current_embeddings = None
    backend.current_metadata = None
    
    # DON'T clear current_model and current_store_info yet
    # They will be cleared when embedding/storage steps run
    
    print("Deep Config data state cleared (model/store preserved)")

def clear_all_backend_state():
    """Clear ALL backend state variables"""
    
    # Clear all global state variables
    backend.current_df = None
    backend.current_chunks = None
    backend.current_embeddings = None
    backend.current_metadata = None
    backend.current_model = None
    backend.current_store_info = None
    backend.current_file_info = None
    backend.current_media_campaign_data = None
    
    # Clear campaign state
    campaign_state.update({
        'model': None,
        'store_info': None,
        'chunks': None,
        'embeddings': None,
        'metadata': None,
        'df': None,
        'file_info': None,
        'preprocessed_df': None,
        'distance_metric': 'cosine',
        'media_campaign_data': None,
        'contact_records': None
    })
    
    # Clear all state files
    files_cleared = []
    errors = []
    
    try:
        # Remove main state file
        if os.path.exists("current_state.pkl"):
            os.remove("current_state.pkl")
            files_cleared.append("current_state.pkl")
            print("Removed current_state.pkl file")
        
        # Remove FAISS store files
        if os.path.exists("faiss_store/data.pkl"):
            os.remove("faiss_store/data.pkl")
            files_cleared.append("faiss_store/data.pkl")
            print("Removed faiss_store/data.pkl file")
        
        if os.path.exists("faiss_store/index.faiss"):
            os.remove("faiss_store/index.faiss")
            files_cleared.append("faiss_store/index.faiss")
            print("Removed faiss_store/index.faiss file")
        
        # Remove processing cache directory
        if os.path.exists("processing_cache"):
            shutil.rmtree("processing_cache")
            files_cleared.append("processing_cache")
            print("Removed processing_cache directory")
        
        # Remove ChromaDB data directory
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")
            files_cleared.append("chroma_db")
            print("Removed chroma_db directory")
            
    except Exception as e:
        error_msg = f"Error removing state files: {e}"
        errors.append(error_msg)
        print(error_msg)
    
    # Log summary
    print(f"Reset completed. Files cleared: {files_cleared}")
    if errors:
        print(f"Errors encountered: {errors}")
    
    print("All backend state cleared")

# ---------------------------
# OpenAI-compatible API Endpoints
# ---------------------------
@app.post("/v1/embeddings")
async def openai_embeddings(
    model: str = Form("text-embedding-ada-002"),
    input: str = Form(...),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None)
):
    """OpenAI-compatible embeddings endpoint"""
    try:
        from backend import OpenAIEmbeddingAPI
        
        embedding_api = OpenAIEmbeddingAPI(
            model_name=model,
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        
        # Handle both string and list of strings
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input
            
        embeddings = embedding_api.encode(texts)
        
        # Format response in OpenAI standard
        response_data = {
            "object": "list",
            "data": [],
            "model": model,
            "usage": {
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        }
        
        for i, embedding in enumerate(embeddings):
            response_data["data"].append({
                "object": "embedding",
                "embedding": embedding.tolist(),
                "index": i
            })
            
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

@app.post("/v1/chat/completions")
async def openai_chat_completions(
    model: str = Form("gpt-3.5-turbo"),
    messages: str = Form(...),
    max_tokens: Optional[int] = Form(1000),
    temperature: Optional[float] = Form(0.7),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None)
):
    """OpenAI-compatible chat completions endpoint (requires external OpenAI API)"""
    try:
        import openai
        
        if openai_api_key:
            openai.api_key = openai_api_key
        if openai_base_url:
            openai.base_url = openai_base_url
            
        # Parse messages from JSON string
        messages_list = json.loads(messages)
        
        response = openai.chat.completions.create(
            model=model,
            messages=messages_list,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return JSONResponse(content=response.dict())
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion error: {str(e)}")

# ---------------------------
# Enhanced DB IMPORT with Large File Support
# ---------------------------
@app.post("/db/test_connection")
async def db_test_connection(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"status": "error", "message": "Unsupported db_type"}
        conn.close()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/db/list_tables")
async def db_list_tables(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        tables = get_table_list(conn, db_type)
        conn.close()
        return {"tables": tables}
    except Exception as e:
        return {"error": str(e)}

@app.post("/db/import_one")
async def db_import_one(
    db_type: str = Form(...),
    host: str = Form(...),
    port: int = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...),
    table_name: str = Form(...),
    processing_mode: str = Form("fast"),
    use_turbo: bool = Form(False),
    batch_size: int = Form(256)
):
    try:
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        
        # Use chunked import for large tables
        file_size = get_table_size(conn, table_name)
        if file_size > LARGE_FILE_THRESHOLD:
            df = import_large_table_to_dataframe(conn, table_name)
        else:
            df = import_table_to_dataframe(conn, table_name)
            
        conn.close()
        
        file_info = {"source": f"db:{db_type}", "table": table_name, "size": file_size}
        
        # Route to appropriate pipeline based on processing mode
        if processing_mode == "fast":
            result = run_fast_pipeline(
                df, 
                file_info=file_info,
                use_turbo=use_turbo,
                batch_size=batch_size
            )
            return {"mode": "fast", "summary": result}
        elif processing_mode == "config1":
            result = run_config1_pipeline(
                df, 
                null_handling="keep",
                fill_value="Unknown",
                chunk_method="recursive",
                chunk_size=400,
                overlap=50,
                model_choice="paraphrase-MiniLM-L6-v2",
                storage_choice="faiss",
                file_info=file_info,
                use_turbo=use_turbo,
                batch_size=batch_size
            )
            return {"mode": "config1", "summary": result}
        elif processing_mode == "deep":
            # Create config dict for deep config pipeline
            config_dict = {
                "preprocessing": {"fill_null_strategy": None, "type_conversions": None, "remove_stopwords_flag": False},
                "chunking": {"method": "recursive", "chunk_size": 400, "overlap": 50},
                "embedding": {"model_name": "paraphrase-MiniLM-L6-v2", "batch_size": batch_size, "use_parallel": use_turbo},
                "storage": {"type": "faiss"}
            }
            result = run_deep_config_pipeline(df, config_dict, file_info)
            return {"mode": "deep", "summary": result}
        else:
            return {"error": f"Unknown processing mode: {processing_mode}"}
            
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/reset-session")
async def reset_session():
    """Reset all backend state"""
    try:
        clear_all_backend_state()
        return {
            "status": "success",
            "message": "All backend state cleared",
            "timestamp": datetime.now().isoformat(),
            "details": "Global variables cleared, state files removed, cache directories cleaned"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Reset session failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        )

# Enhanced FAST MODE with Large File & OpenAI Support
# ---------------------------
@app.post("/run_fast")
async def run_fast(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: str = Form(None),
    port: int = Form(None),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    table_name: str = Form(None),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(False),
    batch_size: int = Form(256)
):
    try:
        # Handle database input
        if db_type and host and table_name and db_type != "sqlite":
            if db_type == "mysql":
                conn = connect_mysql(host, port, username, password, database)
            elif db_type == "postgresql":
                conn = connect_postgresql(host, port, username, password, database)
            else:
                return {"error": "Unsupported db_type"}
            
            # Use chunked import for large tables
            file_size = get_table_size(conn, table_name)
            if file_size > LARGE_FILE_THRESHOLD:
                df = import_large_table_to_dataframe(conn, table_name)
            else:
                df = import_table_to_dataframe(conn, table_name)
                
            conn.close()
            file_info = {"source": f"db:{db_type}", "table": table_name, "size": file_size}
            
            result = run_fast_pipeline(
                df, 
                db_type=db_type,
                file_info=file_info,
                use_turbo=use_turbo,
                batch_size=batch_size
            )
            return {"mode": "fast", "summary": result}
        
        # Handle file input - UPDATED FOR LARGE FILES
        elif file:
            # Create temporary file and stream upload directly to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                # Stream the upload directly to disk (no memory loading)
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            try:
                file_size = os.path.getsize(tmp_path)
                
                # Process directly from filesystem
                if file_size > LARGE_FILE_THRESHOLD and process_large_files:
                    result = process_file_direct(
                        tmp_path, 
                        processing_mode="fast",
                        use_turbo=use_turbo,
                        batch_size=batch_size
                    )
                else:
                    # For smaller files, use existing pipeline
                    df = pd.read_csv(tmp_path)
                    file_info = {
                        "filename": file.filename,
                        "file_size": file_size,
                        "upload_time": pd.Timestamp.now().isoformat()
                    }
                    
                    result = run_fast_pipeline(
                        df, 
                        db_type=db_type,
                        file_info=file_info,
                        use_turbo=use_turbo,
                        batch_size=batch_size
                    )
                
                # Add file info to result
                if 'file_info' not in result:
                    result["file_info"] = {
                        "filename": file.filename,
                        "file_size": file_size,
                        "upload_time": pd.Timestamp.now().isoformat(),
                        "large_file_processed": file_size > LARGE_FILE_THRESHOLD,
                        "turbo_mode": use_turbo,
                        "batch_size": batch_size
                    }
                
                return {"mode": "fast", "summary": result}
                
            finally:
                # Always clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            return {"error": "Either file upload or database parameters required"}
    
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Enhanced CONFIG-1 MODE
# ---------------------------
@app.post("/run_config1")
async def run_config1(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: str = Form(None),
    port: int = Form(None),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    table_name: str = Form(None),
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    n_clusters: int = Form(10),
    document_key_column: str = Form(None),
    token_limit: int = Form(2000),
    retrieval_metric: str = Form("cosine"),
    model_choice: str = Form("paraphrase-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss"),
    apply_default_preprocessing: bool = Form(True),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(False),
    batch_size: int = Form(256)
):
    try:
        # Handle database input
        if db_type and host and table_name and db_type != "sqlite":
            if db_type == "mysql":
                conn = connect_mysql(host, port, username, password, database)
            elif db_type == "postgresql":
                conn = connect_postgresql(host, port, username, password, database)
            else:
                return {"error": "Unsupported db_type"}
            
            # Use chunked import for large tables
            file_size = get_table_size(conn, table_name)
            if file_size > LARGE_FILE_THRESHOLD:
                df = import_large_table_to_dataframe(conn, table_name)
            else:
                df = import_table_to_dataframe(conn, table_name)
                
            conn.close()
            file_info = {"source": f"db:{db_type}", "table": table_name, "size": file_size}
        
        # Handle file input
        elif file:
            # Create temporary file and stream upload directly to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                # Stream the upload directly to disk (no memory loading)
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            try:
                file_size = os.path.getsize(tmp_path)
                
                # Process directly from filesystem for large files
                if file_size > LARGE_FILE_THRESHOLD and process_large_files:
                    result = process_file_direct(
                        tmp_path, 
                        processing_mode="config1",
                        use_turbo=use_turbo,
                        batch_size=batch_size
                    )
                    
                    # Add file info to result
                    if 'file_info' not in result:
                        result["file_info"] = {
                            "filename": file.filename,
                            "file_size": file_size,
                            "upload_time": pd.Timestamp.now().isoformat(),
                            "large_file_processed": True,
                            "turbo_mode": use_turbo,
                            "batch_size": batch_size
                        }
                    
                    return {"mode": "config1", "summary": result}
                else:
                    # For smaller files, use existing pipeline
                    df = pd.read_csv(tmp_path)
                    file_info = {
                        "filename": file.filename,
                        "file_size": file_size,
                        "upload_time": pd.Timestamp.now().isoformat()
                    }
                    
                    result = run_config1_pipeline(
                        df, 
                        chunk_method=chunk_method,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        model_choice=model_choice,
                        storage_choice=storage_choice,
                        db_config=None,
                        file_info=file_info,
                        n_clusters=n_clusters,
                        use_turbo=use_turbo,
                        batch_size=batch_size,
                        document_key_column=document_key_column,
                        token_limit=token_limit,
                        retrieval_metric=retrieval_metric,
                        apply_default_preprocessing=apply_default_preprocessing
                    )
                    
                    return {"mode": "config1", "summary": result}
                
            finally:
                # Always clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
        else:
            return {"error": "Either file upload or database parameters required"}
        
        # For smaller files or database imports, use the original pipeline
        result = run_config1_pipeline(
            df, chunk_method,
            chunk_size, overlap, model_choice, storage_choice, 
            document_key_column=document_key_column,
            token_limit=token_limit,
            retrieval_metric=retrieval_metric,
            apply_default_preprocessing=apply_default_preprocessing,
            n_clusters=n_clusters,
            file_info=file_info,
            use_turbo=use_turbo,
            batch_size=batch_size
        )
        return {"mode": "config1", "summary": result}
    
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# DEEP CONFIG STEP-BY-STEP API ENDPOINTS
# ---------------------------

@app.post("/deep_config/preprocess")
async def deep_config_preprocess(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: str = Form(None),
    port: int = Form(None),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    table_name: str = Form(None)
):
    """Step 1: Preprocess data (load and basic cleaning)"""
    
    # Clear all Deep Config state when starting new preprocessing
    clear_deep_config_state()
    
    try:
        # Handle database input
        if db_type and host and table_name and db_type != "sqlite":
            if db_type == "mysql":
                conn = connect_mysql(host, port, username, password, database)
            elif db_type == "postgresql":
                conn = connect_postgresql(host, port, username, password, database)
            else:
                return {"error": "Unsupported db_type"}
            
            df = import_table_to_dataframe(conn, table_name)
            conn.close()
            file_info = {"source": f"db:{db_type}", "table": table_name}
            
        # Handle file input
        elif file:
            # Use proper CSV loading with encoding detection
            from backend import _load_csv
            df = _load_csv(file.file)
            file_info = {"source": "csv", "filename": file.filename}
        else:
            return {"error": "Either file upload or database parameters required"}
        
        # Complete preprocessing pipeline (applied to both DB and file inputs)
        from backend import clean_column_names, preprocess_default
        
        # Step 1: Normalize headers
        df = clean_column_names(df)
        
        # Step 2: Apply default preprocessing (HTML removal, lowercase, strip, whitespace normalization)
        df = preprocess_default(df)
        
        # Store in global state
        backend.current_df = df.copy()
        set_file_info(file_info)
        
        # Save state to disk for persistence
        try:
            from backend import save_state
            save_state()
        except Exception as e:
            print(f"Warning: Could not save state: {e}")
        
        # Generate sample values for each column
        sample_values = {}
        for col in df.columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                sample_values[col] = str(non_null_values.iloc[0])
            else:
                sample_values[col] = "N/A"
        
        # Generate data preview (first 5 rows, first 3 columns)
        preview_columns = list(df.columns)[:3]
        data_preview = df[preview_columns].head(5).to_dict('records')
        
        return {
            "status": "success",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_values": sample_values,
            "data_preview": data_preview,
            "file_info": file_info
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/deep_config/type_convert")
async def deep_config_type_convert(
    type_conversions: str = Form("{}")
):
    """Step 2: Convert data types"""
    try:
        import json
        conversions = json.loads(type_conversions) if type_conversions else {}
        
        if backend.current_df is None:
            return {"error": "No data available. Run preprocessing first."}
        
        from backend import apply_type_conversion_enhanced
        df_converted = apply_type_conversion_enhanced(backend.current_df, conversions)
        
        # Update global state
        backend.current_df = df_converted.copy()
        
        return {
            "status": "success",
            "rows": len(df_converted),
            "columns": len(df_converted.columns),
            "data_types": {col: str(dtype) for col, dtype in df_converted.dtypes.items()},
            "conversions_applied": conversions
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/deep_config/null_handle")
async def deep_config_null_handle(
    null_strategies: str = Form("{}")
):
    """Step 3: Handle null values"""
    try:
        import json
        strategies = json.loads(null_strategies) if null_strategies else {}
        
        # Debug logging
        print(f"Null handling debug:")
        print(f"  Strategies received: {strategies}")
        print(f"  Current df shape: {backend.current_df.shape if backend.current_df is not None else 'None'}")
        if backend.current_df is not None:
            print(f"  Nulls before: {backend.current_df.isnull().sum().sum()}")
        
        # Try to load state if current_df is None
        if backend.current_df is None:
            try:
                from backend import load_state
                load_state()
                if backend.current_df is None:
                    return {"error": "No data available. Run preprocessing first."}
            except Exception as e:
                print(f"Error loading state: {e}")
                return {"error": "No data available. Run preprocessing first."}
        
        # Normalize strategy column keys to match dataframe columns
        def norm(s: str) -> str:
            return str(s).strip().lower().replace(' ', '_').replace('-', '_')
        df_cols = list(backend.current_df.columns)
        norm_to_actual = {norm(c): c for c in df_cols}
        normalized_strategies = {}
        for key, value in strategies.items():
            nk = norm(key)
            actual = norm_to_actual.get(nk)
            if actual is None:
                # Try a looser match (remove non-alnum)
                import re
                nk2 = re.sub(r"[^a-z0-9_]", "", nk)
                found = None
                for nc, ac in norm_to_actual.items():
                    if re.sub(r"[^a-z0-9_]", "", nc) == nk2:
                        found = ac
                        break
                if found is not None:
                    normalized_strategies[found] = value
                else:
                    # Fallback: keep the original key if not found
                    normalized_strategies[key] = value
            else:
                normalized_strategies[actual] = value
        print(f"  Strategies normalized: {normalized_strategies}")
        
        from backend import apply_null_strategies_enhanced
        df_processed = apply_null_strategies_enhanced(backend.current_df, normalized_strategies, add_flags=False)
        
        # Debug logging after processing
        print(f"  Nulls after: {df_processed.isnull().sum().sum()}")
        print(f"  Processed df shape: {df_processed.shape}")
        
        # Update global state
        backend.current_df = df_processed.copy()
        
        # Save state to disk for persistence
        try:
            from backend import save_state
            save_state()
        except Exception as e:
            print(f"Warning: Could not save state: {e}")
        
        return {
            "status": "success",
            "rows": len(df_processed),
            "columns": len(df_processed.columns),
            "null_count": int(df_processed.isnull().sum().sum()),
            "strategies_applied": normalized_strategies
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/deep_config/analyze_nulls")
async def analyze_nulls():
    """Analyze null values in current data without processing"""
    try:
        if backend.current_df is None:
            return {"error": "No data available. Run preprocessing first."}
        
        from backend import profile_nulls_enhanced
        null_profile = profile_nulls_enhanced(backend.current_df)
        
        # Filter to only show columns with null values (matching Streamlit behavior)
        filtered_profile = null_profile[null_profile['null_count'] > 0].copy()
        
        # Debug logging
        print(f"Total columns analyzed: {len(null_profile)}")
        print(f"Columns with nulls: {len(filtered_profile)}")
        print(f"Filtered columns: {list(filtered_profile['column'])}")
        
        # Convert DataFrame to list of dictionaries
        null_profile_list = filtered_profile.to_dict('records')
        
        return {
            "status": "success",
            "null_profile": null_profile_list,
            "total_nulls": int(backend.current_df.isnull().sum().sum()),
            "has_nulls": bool(backend.current_df.isnull().sum().sum() > 0)
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/deep_config/analyze_duplicates")
async def analyze_duplicates():
    """Analyze duplicate rows in current data without processing"""
    try:
        if backend.current_df is None:
            return {"error": "No data available. Run preprocessing first."}
        
        from backend import analyze_duplicates_enhanced
        duplicate_analysis = analyze_duplicates_enhanced(backend.current_df)
        
        # Debug logging
        print(f"Duplicate analysis debug:")
        print(f"  Total rows: {duplicate_analysis['total_rows']}")
        print(f"  Duplicate pairs: {duplicate_analysis['duplicate_pairs_count']}")
        print(f"  Rows to remove: {duplicate_analysis['rows_to_remove']}")
        print(f"  Duplicate percentage: {duplicate_analysis['duplicate_percentage']}")
        print(f"  Has duplicates: {duplicate_analysis['has_duplicates']}")
        print(f"  Duplicate groups count: {len(duplicate_analysis['duplicate_groups'])}")
        
        return {
            "status": "success",
            "duplicate_analysis": duplicate_analysis
        }
    
    except Exception as e:
        return {"error": str(e)}
@app.post("/deep_config/duplicates")
async def deep_config_duplicates(
    strategy: str = Form("keep_first")
):
    """Step 3: Handle duplicate rows"""
    try:
        if backend.current_df is None:
            try:
                from backend import load_state
                load_state()
                if backend.current_df is None:
                    return {"error": "No data available. Run preprocessing first."}
            except Exception as e:
                print(f"Error loading state: {e}")
                return {"error": "No data available. Run preprocessing first."}
        
        from backend import analyze_duplicates_enhanced, remove_duplicates_enhanced
        
        # Analyze duplicates first
        duplicate_analysis = analyze_duplicates_enhanced(backend.current_df)
        
        # Apply duplicate removal strategy
        df_processed = remove_duplicates_enhanced(backend.current_df, strategy)
        
        # Update global state
        backend.current_df = df_processed.copy()
        
        # Save state to disk for persistence
        try:
            from backend import save_state
            save_state()
        except Exception as e:
            print(f"Warning: Could not save state: {e}")
        
        return {
            "status": "success",
            "rows_before": int(duplicate_analysis['total_rows']),
            "rows_after": int(len(df_processed)),
            "duplicates_removed": int(duplicate_analysis['total_rows'] - len(df_processed)),
            "strategy_applied": str(strategy),
            "has_duplicates": bool(duplicate_analysis['has_duplicates']),
            "duplicate_groups_count": int(len(duplicate_analysis['duplicate_groups']))
        }
    
    except Exception as e:
        print(f"Error in duplicates API: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/deep_config/stopwords")
async def deep_config_stopwords(
    remove_stopwords: bool = Form(False)
):
    """Step 4: Remove stop words"""
    try:
        # Try to load state if current_df is None
        if backend.current_df is None:
            try:
                from backend import load_state
                load_state()
                if backend.current_df is None:
                    return {"error": "No data available. Run preprocessing first."}
            except Exception as e:
                print(f"Error loading state: {e}")
                return {"error": "No data available. Run preprocessing first."}
        
        from backend import remove_stopwords_from_text_column_enhanced
        df_processed = remove_stopwords_from_text_column_enhanced(backend.current_df, remove_stopwords)
        
        # Update global state
        backend.current_df = df_processed.copy()
        
        # Save state to disk for persistence
        try:
            from backend import save_state
            save_state()
        except Exception as e:
            print(f"Warning: Could not save state: {e}")
        
        return {
            "status": "success",
            "rows": len(df_processed),
            "columns": len(df_processed.columns),
            "stopwords_removed": remove_stopwords
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/deep_config/normalize")
async def deep_config_normalize(
    text_processing: str = Form("none")
):
    """Step 5: Text normalization"""
    try:
        # Try to load state if current_df is None
        if backend.current_df is None:
            try:
                from backend import load_state
                load_state()
                if backend.current_df is None:
                    return {"error": "No data available. Run preprocessing first."}
            except Exception as e:
                print(f"Error loading state: {e}")
                return {"error": "No data available. Run preprocessing first."}
        
        from backend import process_text_enhanced
        df_processed = process_text_enhanced(backend.current_df, text_processing)
        
        # Update global state
        backend.current_df = df_processed.copy()
        
        # Save state to disk for persistence
        try:
            from backend import save_state
            save_state()
        except Exception as e:
            print(f"Warning: Could not save state: {e}")
        
        return {
            "status": "success",
            "rows": len(df_processed),
            "columns": len(df_processed.columns),
            "text_processing": text_processing
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/deep_config/metadata_columns")
async def get_metadata_columns():
    """Get available columns for metadata selection with smart limits"""
    try:
        if backend.current_df is None:
            return {"error": "No data available. Run preprocessing first."}
        
        from backend import get_metadata_limits
        
        # Get available columns
        numeric_cols = backend.current_df.select_dtypes(include=['number']).columns.tolist()
        
        # Get smart limits and filtered categorical columns
        max_numeric, max_categorical, categorical_cols = get_metadata_limits(backend.current_df)
        
        # Get sample values for each column
        numeric_samples = {}
        categorical_samples = {}
        
        for col in numeric_cols[:10]:  # Limit to first 10 for performance
            sample_val = backend.current_df[col].dropna().iloc[0] if not backend.current_df[col].dropna().empty else "N/A"
            numeric_samples[col] = str(sample_val)
        
        for col in categorical_cols[:10]:  # Limit to first 10 for performance
            sample_val = backend.current_df[col].dropna().iloc[0] if not backend.current_df[col].dropna().empty else "N/A"
            categorical_samples[col] = str(sample_val)
        
        # Get all categorical columns for comparison
        all_categorical_cols = backend.current_df.select_dtypes(include=['object']).columns.tolist()
        filtered_out_categorical = [col for col in all_categorical_cols if col not in categorical_cols]
        
        return {
            "status": "success",
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "max_numeric": max_numeric,
            "max_categorical": max_categorical,
            "numeric_samples": numeric_samples,
            "categorical_samples": categorical_samples,
            "total_columns": len(backend.current_df.columns),
            "filtered_out_categorical": filtered_out_categorical,
            "cardinality_threshold": 5
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/deep_config/chunk")
async def deep_config_chunk(
    chunk_method: str = Form("fixed"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    key_column: str = Form(None),
    token_limit: int = Form(2000),
    n_clusters: int = Form(10),
    store_metadata: bool = Form(False),
    selected_numeric_columns: str = Form("[]"),
    selected_categorical_columns: str = Form("[]")
):
    """Step 6: Chunk data"""
    try:
        if backend.current_df is None:
            return {"error": "No data available. Run preprocessing first."}
        
        from backend import (
            chunk_fixed_enhanced, chunk_recursive_keyvalue_enhanced,
            chunk_semantic_cluster_enhanced, document_based_chunking_enhanced
        )
        
        if chunk_method == "fixed":
            chunks = chunk_fixed_enhanced(backend.current_df, chunk_size, overlap)
            metadata = [{"chunk_id": f"fixed_{i:04d}", "method": "fixed"} for i in range(len(chunks))]
        elif chunk_method == "recursive":
            chunks = chunk_recursive_keyvalue_enhanced(backend.current_df, chunk_size, overlap)
            metadata = [{"chunk_id": f"kv_{i:04d}", "method": "recursive_kv"} for i in range(len(chunks))]
        elif chunk_method == "semantic":
            chunks = chunk_semantic_cluster_enhanced(backend.current_df, n_clusters)
            metadata = [{"chunk_id": f"sem_cluster_{i:04d}", "method": "semantic_cluster"} for i in range(len(chunks))]
        elif chunk_method == "document":
            chunks, metadata = document_based_chunking_enhanced(backend.current_df, key_column, token_limit)
        else:
            return {"error": f"Unknown chunking method: {chunk_method}"}
        
        # Enhance metadata with user-selected columns if enabled
        if store_metadata and metadata:
            import json
            try:
                selected_numeric = json.loads(selected_numeric_columns) if selected_numeric_columns else []
                selected_categorical = json.loads(selected_categorical_columns) if selected_categorical_columns else []
            except json.JSONDecodeError:
                selected_numeric = []
                selected_categorical = []
            
            enhanced_metadata = []
            for i, meta in enumerate(metadata):
                enhanced_meta = meta.copy()
                
                # Add selected numeric columns (min/mean/max per chunk)
                for col in selected_numeric:
                    if col in backend.current_df.columns:
                        # For document-based chunking, filter by key_value
                        if chunk_method == "document" and key_column and meta.get('key_value'):
                            col_data = backend.current_df[backend.current_df[key_column] == meta.get('key_value')][col]
                        else:
                            # For other methods, use all data or sample
                            col_data = backend.current_df[col].dropna()
                        
                        if not col_data.empty:
                            enhanced_meta[f'{col}_min'] = float(col_data.min())
                            enhanced_meta[f'{col}_mean'] = float(col_data.mean())
                            enhanced_meta[f'{col}_max'] = float(col_data.max())
                
                # Add selected categorical columns (mode per chunk)
                for col in selected_categorical:
                    if col in backend.current_df.columns:
                        # For document-based chunking, filter by key_value
                        if chunk_method == "document" and key_column and meta.get('key_value'):
                            col_data = backend.current_df[backend.current_df[key_column] == meta.get('key_value')][col]
                        else:
                            # For other methods, use all data or sample
                            col_data = backend.current_df[col].dropna()
                        
                        if not col_data.empty:
                            mode_value = col_data.mode()
                            enhanced_meta[f'{col}_mode'] = str(mode_value.iloc[0]) if not mode_value.empty else 'unknown'
                
                enhanced_metadata.append(enhanced_meta)
            metadata = enhanced_metadata
        
        # Store in global state
        backend.current_chunks = chunks
        backend.current_metadata = metadata
        
        # Save state to disk for persistence
        save_state()
        
        return {
            "status": "success",
            "total_chunks": len(chunks),
            "chunks": chunks,
            "chunk_method": chunk_method,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "metadata": metadata,
            "metadata_enabled": store_metadata
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/deep_config/embed")
async def deep_config_embed(
    model_name: str = Form("paraphrase-MiniLM-L6-v2"),
    batch_size: int = Form(64),
    use_parallel: bool = Form(True)
):
    """Step 7: Generate embeddings"""
    try:
        if backend.current_chunks is None:
            return {"error": "No chunks available. Run chunking first."}
        
        from backend import embed_texts_enhanced
        model, embeddings = embed_texts_enhanced(
            backend.current_chunks, model_name, None, None, batch_size, use_parallel
        )
        
        # Store in global state
        backend.current_model = model
        backend.current_embeddings = embeddings
        
        # Save state for persistence
        save_state()
        
        return {
            "status": "success",
            "total_chunks": len(embeddings),
            "vector_dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "embeddings": embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            "chunk_texts": backend.current_chunks,
            "model_name": model_name,
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/deep_config/store")
async def deep_config_store(
    storage_type: str = Form("chroma"),
    collection_name: str = Form("deep_config_collection")
):
    """Step 8: Store embeddings"""
    try:
        print(f"DEBUG: deep_config_store called with storage_type: {storage_type}, collection_name: {collection_name}")
        
        if backend.current_chunks is None or backend.current_embeddings is None:
            return {"error": "No chunks or embeddings available. Run chunking and embedding first."}
        
        from backend import store_chroma_enhanced, store_faiss_enhanced
        
        print(f"DEBUG: About to store with storage_type: {storage_type}")
        
        if storage_type == "chroma":
            print("DEBUG: Using ChromaDB storage")
            store = store_chroma_enhanced(backend.current_chunks, backend.current_embeddings, collection_name, backend.current_metadata)
        elif storage_type == "faiss":
            print("DEBUG: Using FAISS storage")
            store = store_faiss_enhanced(backend.current_chunks, backend.current_embeddings, backend.current_metadata)
        else:
            print(f"DEBUG: Unknown storage type: {storage_type}")
            return {"error": f"Unknown storage type: {storage_type}"}
        
        # Store in global state
        backend.current_store_info = store
        
        # Save state for retrieval
        save_state()
        
        print(f"Deep Config storage completed: {storage_type}, store keys: {list(store.keys())}")
        print(f"DEBUG: Final store type: {store.get('type', 'unknown')}")
        
        return {
            "status": "success",
            "storage_type": storage_type,
            "collection_name": collection_name if storage_type == "chroma" else "faiss_index",
            "total_vectors": len(backend.current_chunks)
        }
    
    except Exception as e:
        print(f"DEBUG: Exception in deep_config_store: {str(e)}")
        return {"error": str(e)}

@app.post("/retrieve_with_metadata")
async def retrieve_with_metadata(
    query: str = Form(...),
    k: int = Form(5),
    metadata_filter: str = Form("{}")
):
    """Enhanced retrieval with metadata filtering support"""
    
    try:
        import json
        
        # Parse metadata filter
        try:
            filter_dict = json.loads(metadata_filter) if metadata_filter else {}
        except json.JSONDecodeError:
            return {"error": "Invalid metadata filter JSON"}
        
        # Load state if needed
        if backend.current_model is None or backend.current_store_info is None:
            from backend import load_state
            load_state()
        
        if backend.current_model is None or backend.current_store_info is None:
            return {"error": "No model or store available. Run a pipeline first."}
        
        # Encode query
        if hasattr(backend.current_model, 'encode'):
            query_embedding = backend.current_model.encode([query])
        else:
            query_embedding = backend.current_model.encode([query])
        
        query_arr = np.array(query_embedding).astype("float32")
        
        # Enhanced retrieval based on store type
        if backend.current_store_info["type"] == "faiss":
            from backend import query_faiss_with_metadata
            faiss_data = backend.current_store_info.get("data", {})
            index = backend.current_store_info["index"]
            results = query_faiss_with_metadata(index, faiss_data, query_arr, k, filter_dict)
            
        elif backend.current_store_info["type"] == "chroma":
            collection = backend.current_store_info["collection"]
            
            # Build where clause for ChromaDB
            where_clause = {}
            for key, value in filter_dict.items():
                where_clause[key] = value
            
            chroma_results = collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=k,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            results = []
            for i, (doc, distance, meta) in enumerate(zip(
                chroma_results["documents"][0], 
                chroma_results["distances"][0],
                chroma_results.get("metadatas", [[]])[0] or [{}] * len(chroma_results["documents"][0])
            )):
                similarity = 1 / (1 + distance)
                results.append({
                    "rank": i + 1,
                    "content": doc,
                    "similarity": float(similarity),
                    "distance": float(distance),
                    "metadata": meta
                })
        else:
            return {"error": f"Unsupported store type: {backend.current_store_info['type']}"}
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "total_results": len(results),
            "metadata_filter_applied": metadata_filter != "{}" and len(filter_dict) > 0,
            "store_type": backend.current_store_info["type"]
        }
    
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Enhanced DEEP CONFIG ENDPOINT
# ---------------------------
@app.post("/run_deep_config")
async def run_deep_config(
    file: Optional[UploadFile] = File(None),
    db_type: str = Form("sqlite"),
    host: str = Form(None),
    port: int = Form(None),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    table_name: str = Form(None),
    preprocessing_config: str = Form("{}"),
    chunking_config: str = Form("{}"),
    embedding_config: str = Form("{}"),
    storage_config: str = Form("{}"),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(False),
    batch_size: int = Form(64)
):
    """Enhanced deep config pipeline with comprehensive preprocessing, chunking, embedding, and storage"""
    try:
        # Parse configuration dictionaries
        try:
            preprocessing_dict = json.loads(preprocessing_config) if preprocessing_config else {}
            chunking_dict = json.loads(chunking_config) if chunking_config else {}
            embedding_dict = json.loads(embedding_config) if embedding_config else {}
            storage_dict = json.loads(storage_config) if storage_config else {}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON configuration: {str(e)}"}
        
        # Handle database input
        if db_type and host and table_name and db_type != "sqlite":
            if db_type == "mysql":
                conn = connect_mysql(host, port, username, password, database)
            elif db_type == "postgresql":
                conn = connect_postgresql(host, port, username, password, database)
            else:
                return {"error": "Unsupported db_type"}
            
            # Use chunked import for large tables
            file_size = get_table_size(conn, table_name)
            if file_size > LARGE_FILE_THRESHOLD:
                df = import_large_table_to_dataframe(conn, table_name)
            else:
                df = import_table_to_dataframe(conn, table_name)
                
            conn.close()
            file_info = {"source": f"db:{db_type}", "table": table_name, "size": file_size}
        
        # Handle file input
        elif file:
            # Create temporary file and stream upload directly to disk
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                tmp_path = tmp_file.name
            
            try:
                file_size = os.path.getsize(tmp_path)
                
                # Process directly from filesystem for large files
                if file_size > LARGE_FILE_THRESHOLD and process_large_files:
                    result = process_large_file(
                        tmp_path, 
                        processing_mode="deep_config",
                        preprocessing_config=preprocessing_dict,
                        chunking_config=chunking_dict,
                        embedding_config=embedding_dict,
                        storage_config=storage_dict,
                        use_turbo=use_turbo,
                        batch_size=batch_size
                    )
                    
                    # Add file info to result
                    if 'file_info' not in result:
                        result["file_info"] = {
                            "filename": file.filename,
                            "file_size": file_size,
                            "upload_time": pd.Timestamp.now().isoformat(),
                            "large_file_processed": True,
                            "turbo_mode": use_turbo,
                            "batch_size": batch_size
                        }
                    
                    return {"mode": "deep_config", "summary": result}
                else:
                    # For smaller files, use existing pipeline
                    df = pd.read_csv(tmp_path)
                    file_info = {
                        "filename": file.filename,
                        "file_size": file_size,
                        "upload_time": pd.Timestamp.now().isoformat()
                    }
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            return {"error": "Either file upload or database parameters required for deep config mode"}
        
        # Combine all configurations
        config_dict = {
            "preprocessing": preprocessing_dict,
            "chunking": chunking_dict,
            "embedding": {
                **embedding_dict,
                "batch_size": batch_size,
                "use_parallel": use_turbo
            },
            "storage": storage_dict
        }
        
        # Run the enhanced deep config pipeline
        result = run_deep_config_pipeline(df, config_dict, file_info)
        return {"mode": "deep_config", "summary": result}
    
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# Enhanced RETRIEVAL ENDPOINTS
# ---------------------------
@app.post("/retrieve")
async def retrieve(query: str = Form(...), k: int = Form(5)):
    """Retrieve similar chunks after running any pipeline"""
    result = retrieve_similar(query, k)
    return result

@app.post("/v1/retrieve")
async def openai_style_retrieve(
    query: str = Form(...),
    model: str = Form("all-MiniLM-L6-v2"),
    n_results: int = Form(5)
):
    """OpenAI-style retrieval endpoint"""
    result = retrieve_similar(query, n_results)
    
    # Format in OpenAI style
    if "error" in result:
        return {"error": result["error"]}
    
    formatted_results = []
    for res in result["results"]:
        formatted_results.append({
            "object": "retrieval_result",
            "score": res["similarity"],
            "content": res["content"],
            "rank": res["rank"]
        })
    
    return {
        "object": "list",
        "data": formatted_results,
        "model": model,
        "query": query,
        "n_results": n_results
    }

# SYSTEM INFO ENDPOINTS
# ---------------------------
@app.get("/system_info")
async def system_info():
    """Get system information"""
    return get_system_info()

@app.get("/file_info")
async def file_info():
    """Get file information"""
    return get_file_info()

# EXPORT ENDPOINTS
# ---------------------------
@app.get("/export/chunks")
async def export_chunks_file():
    """Export chunks as CSV file"""
    chunks_csv = export_chunks()
    if not chunks_csv:
        return {"error": "No chunks available"}
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(chunks_csv)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="chunks.csv", media_type="text/csv")

@app.get("/export/embeddings")
async def export_embeddings_file():
    """Export embeddings as numpy file"""
    embeddings = export_embeddings()
    if embeddings is None:
        return {"error": "No embeddings available"}
    
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f, embeddings)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="embeddings.npy", media_type="application/octet-stream")

@app.get("/export/embeddings_text")
async def export_embeddings_text_file():
    """Export embeddings as JSON file"""
    from backend import export_embeddings_json
    embeddings_json = export_embeddings_json()
    if not embeddings_json or embeddings_json == "{}":
        return {"error": "No embeddings available"}
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(embeddings_json)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="embeddings.json", media_type="application/json")

# ---------------------------
# DOWNLOAD ENDPOINTS
# ---------------------------

@app.get("/api/preprocessed/preview")
async def get_preprocessed_preview():
    """Get preprocessed data preview for UI display"""
    try:
        # Try to get data from current_df first
        if backend.current_df is not None:
            preview_df = backend.current_df.head(100)
            return {
                "status": "success",
                "data": preview_df.to_dict('records'),
                "columns": list(preview_df.columns),
                "total_rows": len(backend.current_df),
                "preview_rows": len(preview_df)
            }
        
        # If current_df is None, try to load from saved state
        try:
            from backend import load_state
            if load_state() and backend.current_df is not None:
                preview_df = backend.current_df.head(100)
                return {
                    "status": "success",
                    "data": preview_df.to_dict('records'),
                    "columns": list(preview_df.columns),
                    "total_rows": len(backend.current_df),
                    "preview_rows": len(preview_df)
                }
        except:
            pass
        
        return {"error": "No preprocessed data available"}
        
    except Exception as e:
        return {"error": f"Preview generation failed: {str(e)}"}

@app.get("/export/preprocessed")
async def export_preprocessed():
    """Export preprocessed data as CSV (for Fast Mode and Config-1 Mode)"""
    if backend.current_df is None:
        return {"error": "No preprocessed data available"}
    
    csv_data = backend.current_df.to_csv(index=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_data)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="preprocessed_data.csv", media_type="text/csv")

# ---------------------------
# DEEP CONFIG STEP-BY-STEP DOWNLOAD ENDPOINTS
# ---------------------------

@app.get("/deep_config/export/preprocessed")
async def export_deep_config_preprocessed():
    """Export preprocessed data as CSV"""
    if backend.current_df is None:
        return {"error": "No preprocessed data available"}
    
    csv_data = backend.current_df.to_csv(index=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_data)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="preprocessed_data.csv", media_type="text/csv")

@app.get("/deep_config/export/chunks")
async def export_deep_config_chunks():
    """Export chunks as CSV"""
    if backend.current_chunks is None:
        return {"error": "No chunks available"}
    
    # Create DataFrame with chunks and metadata
    chunks_df = pd.DataFrame({
        'chunk_id': [f"chunk_{i:04d}" for i in range(len(backend.current_chunks))],
        'content': backend.current_chunks,
        'metadata': [str(meta) for meta in (backend.current_metadata or [{}] * len(backend.current_chunks))]
    })
    
    csv_data = chunks_df.to_csv(index=False)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_data)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="chunks.csv", media_type="text/csv")

@app.get("/deep_config/export/embeddings")
async def export_deep_config_embeddings():
    """Export embeddings as JSON"""
    if backend.current_embeddings is None:
        return {"error": "No embeddings available"}
    
    # Convert embeddings to JSON-serializable format
    embeddings_data = {
        "embeddings": backend.current_embeddings.tolist(),
        "shape": backend.current_embeddings.shape,
        "metadata": backend.current_metadata or [],
        "model_info": {
            "model_name": getattr(backend.current_model, 'model_name', 'unknown') if backend.current_model else 'unknown',
            "embedding_dimension": backend.current_embeddings.shape[1] if len(backend.current_embeddings.shape) > 1 else 0
        }
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(embeddings_data, f, indent=2)
        temp_path = f.name
    
    return FileResponse(temp_path, filename="embeddings.json", media_type="application/json")
# HEALTH CHECK & LARGE FILE SUPPORT INFO
# ---------------------------
@app.get("/")
async def root():
    system_info = get_system_info()
    return {
        "message": "Chunking Optimizer API is running", 
        "version": "1.0",
        "large_file_support": True,
        "max_recommended_file_size": system_info.get("max_recommended_file_size", "N/A"),
        "openai_compatible": True,
        "performance_optimized": True,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "parallel_workers": PARALLEL_WORKERS
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "large_file_support": True, "performance_optimized": True}

@app.get("/capabilities")
async def capabilities():
    """Return API capabilities"""
    return {
        "openai_compatible_endpoints": [
            "/v1/embeddings",
            "/v1/chat/completions", 
            "/v1/retrieve"
        ],
        "processing_modes": ["fast", "config1", "deep", "campaign"],
        "large_file_support": True,
        "max_file_size_recommendation": "3GB+",
        "supported_embedding_models": [
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2", 
            "text-embedding-ada-002"
        ],
        "chunking_methods": {
            "fast": ["semantic_clustering"],
            "config1": ["fixed_size", "recursive", "document_based", "semantic_clustering"],
            "deep": ["fixed_size", "recursive", "document_based", "semantic_clustering"],
            "campaign": ["record_based", "company_based", "source_based", "semantic_clustering", "document_based"]
        },
        "batch_processing": True,
        "memory_optimized": True,
        "database_large_table_support": True,
        "performance_features": {
            "turbo_mode": True,
            "parallel_processing": True,
            "optimized_batch_size": 256,
            "caching_system": True
        },
        "campaign_features": {
            "smart_company_retrieval": True,
            "field_detection": True,
            "contextual_display": True,
            "complete_records": True,
            "specialized_preprocessing": True
        }
    }

@app.get("/debug/storage")
async def debug_storage():
    """Debug current storage state"""
    from backend import debug_storage_state, current_store_info, current_chunks, current_embeddings
    import logging
    import os
    
    # Set up logging to capture debug output
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    debug_storage_state()
    
    return {
        "storage_info": current_store_info,
        "chunks_count": len(current_chunks) if current_chunks else 0,
        "embeddings_shape": current_embeddings.shape if current_embeddings is not None else None,
        "faiss_files_exist": {
            "index": os.path.exists("faiss_store/index.faiss"),
            "data": os.path.exists("faiss_store/data.pkl")
        } if current_store_info and current_store_info.get("type") == "faiss" else None
    }

# ---------------------------
# NEW: Large File Upload Endpoint
# ---------------------------
@app.post("/upload_large_file")
async def upload_large_file(
    file: UploadFile = File(...),
    processing_mode: str = Form("fast"),
    use_turbo: bool = Form(True),
    batch_size: int = Form(256)
):
    """Direct large file upload endpoint with disk streaming"""
    try:
        # Create temporary file and stream upload directly to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            # Stream the upload directly to disk (no memory loading)
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            file_size = os.path.getsize(tmp_path)
            
            # Process directly from filesystem
            result = process_file_direct(
                tmp_path, 
                processing_mode=processing_mode,
                use_turbo=use_turbo,
                batch_size=batch_size
            )
            
            # Add file info to result
            result["file_info"] = {
                "filename": file.filename,
                "file_size": file_size,
                "upload_time": pd.Timestamp.now().isoformat(),
                "large_file_processed": True,
                "processing_mode": processing_mode,
                "turbo_mode": use_turbo,
                "batch_size": batch_size
            }
            
            return {"mode": processing_mode, "summary": result}
            
        finally:
            # Always clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# UNIVERSAL ENDPOINT FOR COMPANY INTEGRATION
# ---------------------------
@app.post("/api/v1/process")
async def universal_processor(
    operation: str = Form(...),  # Required: operation type
    file: Optional[UploadFile] = File(None),
    # Database parameters
    db_type: str = Form("sqlite"),
    host: str = Form(None),
    port: int = Form(None),
    username: str = Form(None),
    password: str = Form(None),
    database: str = Form(None),
    table_name: str = Form(None),
    # Processing parameters
    chunk_method: str = Form("recursive"),
    chunk_size: int = Form(400),
    overlap: int = Form(50),
    document_key_column: str = Form(None),
    token_limit: int = Form(2000),
    retrieval_metric: str = Form("cosine"),
    model_choice: str = Form("paraphrase-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss"),
    apply_default_preprocessing: bool = Form(True),
    use_openai: bool = Form(False),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    process_large_files: bool = Form(True),
    use_turbo: bool = Form(False),
    batch_size: int = Form(256),
    # Deep config specific
    step: str = Form(None),
    preprocessing_config: str = Form("{}"),
    chunking_config: str = Form("{}"),
    embedding_config: str = Form("{}"),
    storage_config: str = Form("{}"),
    # Type conversion and null handling
    type_conversions: str = Form("{}"),
    null_strategies: str = Form("{}"),
    # Duplicate handling
    duplicate_strategy: str = Form("keep_first"),
    # Text processing
    remove_stopwords: bool = Form(False),
    text_processing: str = Form("none"),
    # Chunking parameters
    key_column: str = Form(None),
    n_clusters: int = Form(10),
    preserve_headers: bool = Form(True),
    store_metadata: bool = Form(False),
    numeric_columns: int = Form(0),
    categorical_columns: int = Form(0),
    # Retrieval parameters
    query: str = Form(None),
    k: int = Form(5),
    metadata_filter: str = Form("{}"),
    # Export parameters
    export_type: str = Form("chunks"),
    # System parameters
    system_action: str = Form("info")
):
    """
    Universal endpoint that consolidates all operations into a single endpoint.
    
    Operations:
    - "fast": Fast Mode processing
    - "config1": Config-1 Mode processing  
    - "deep_config": Deep Config Mode processing
    - "deep_config_step": Deep Config step-by-step operations
    - "retrieve": Semantic search
    - "export": Export operations
    - "system": System information
    - "db_test": Database connection test
    - "db_list": Database table listing
    - "db_import": Database table import
    """
    try:
        # Route to appropriate handler based on operation
        if operation == "fast":
            return await internal_run_fast(file, db_type, host, port, username, password,
                                         database, table_name, process_large_files, use_turbo, batch_size)
        
        elif operation == "config1":
            return await internal_run_config1(file, db_type, host, port, username, password,
                                            database, table_name, chunk_method, chunk_size, overlap,
                                            document_key_column, token_limit, retrieval_metric,
                                            model_choice, storage_choice, apply_default_preprocessing,
                                            n_clusters, process_large_files, use_turbo, batch_size)
        
        elif operation == "deep_config":
            return await internal_run_deep_config(file, db_type, host, port, username, password,
                                                database, table_name, preprocessing_config,
                                                chunking_config, embedding_config, storage_config,
                                                process_large_files, use_turbo, batch_size)
        
        elif operation == "deep_config_step":
            return await internal_deep_config_step(step, file, db_type, host, port, username, password,
                                                 database, table_name, type_conversions, null_strategies,
                                                 duplicate_strategy, remove_stopwords, text_processing,
                                                 chunk_method, chunk_size, overlap, key_column, token_limit,
                                                 n_clusters, preserve_headers, store_metadata,
                                                 numeric_columns, categorical_columns, model_choice,
                                                 storage_choice, batch_size, use_turbo)
        
        elif operation == "retrieve":
            return await internal_retrieve(query, k, metadata_filter)
        
        elif operation == "export":
            return await internal_export(export_type)
        
        elif operation == "system":
            return await internal_system(system_action)
        
        elif operation == "db_test":
            return await internal_db_test(db_type, host, port, username, password, database)
        
        elif operation == "db_list":
            return await internal_db_list(db_type, host, port, username, password, database)
        
        elif operation == "db_import":
            return await internal_db_import(db_type, host, port, username, password, database,
                                          table_name, chunk_method, chunk_size, overlap, model_choice,
                                          storage_choice, use_turbo, batch_size)
        
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    except Exception as e:
        return {"error": str(e)}

# Internal handler functions that wrap existing endpoint logic
async def internal_run_fast(file, db_type, host, port, username, password, database, table_name,
                           process_large_files, use_turbo, batch_size):
    """Internal handler for Fast Mode processing"""
    # Handle database input
    if db_type and host and table_name and db_type != "sqlite":
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        
        # Use chunked import for large tables
        file_size = get_table_size(conn, table_name)
        if file_size > LARGE_FILE_THRESHOLD:
            df = import_large_table_to_dataframe(conn, table_name)
        else:
            df = import_table_to_dataframe(conn, table_name)
            
        conn.close()
        file_info = {"source": f"db:{db_type}", "table": table_name, "size": file_size}
        
        result = run_fast_pipeline(
            df, 
            db_type=db_type,
            file_info=file_info,
            use_turbo=use_turbo,
            batch_size=batch_size
        )
        return {"mode": "fast", "summary": result}
    
    # Handle file input
    elif file:
        # Create temporary file and stream upload directly to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            file_size = os.path.getsize(tmp_path)
            
            # Process directly from filesystem
            if file_size > LARGE_FILE_THRESHOLD and process_large_files:
                result = process_file_direct(
                    tmp_path, 
                    processing_mode="fast",
                    use_turbo=use_turbo,
                    batch_size=batch_size
                )
            else:
                # For smaller files, use existing pipeline
                df = pd.read_csv(tmp_path)
                file_info = {
                    "filename": file.filename,
                    "file_size": file_size,
                    "upload_time": pd.Timestamp.now().isoformat()
                }
                
                result = run_fast_pipeline(
                    df, 
                    db_type=db_type,
                    file_info=file_info,
                    use_turbo=use_turbo,
                    batch_size=batch_size
                )
            
            # Add file info to result
            if 'file_info' not in result:
                result["file_info"] = {
                    "filename": file.filename,
                    "file_size": file_size,
                    "upload_time": pd.Timestamp.now().isoformat(),
                    "large_file_processed": file_size > LARGE_FILE_THRESHOLD,
                    "turbo_mode": use_turbo,
                    "batch_size": batch_size
                }
            
            return {"mode": "fast", "summary": result}
            
        finally:
            # Always clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    else:
        return {"error": "Either file upload or database parameters required"}

async def internal_run_config1(file, db_type, host, port, username, password, database, table_name,
                             chunk_method, chunk_size, overlap, document_key_column, token_limit,
                             retrieval_metric, model_choice, storage_choice, apply_default_preprocessing,
                             n_clusters, process_large_files, use_turbo, batch_size):
    """Internal handler for Config-1 Mode processing"""
    # Handle database input
    if db_type and host and table_name and db_type != "sqlite":
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        
        # Use chunked import for large tables
        file_size = get_table_size(conn, table_name)
        if file_size > LARGE_FILE_THRESHOLD:
            df = import_large_table_to_dataframe(conn, table_name)
        else:
            df = import_table_to_dataframe(conn, table_name)
            
        conn.close()
        file_info = {"source": f"db:{db_type}", "table": table_name, "size": file_size}
        
        result = run_config1_pipeline(
            df, 
            chunk_method=chunk_method,
            chunk_size=chunk_size,
            overlap=overlap,
            model_choice=model_choice,
            storage_choice=storage_choice,
            db_config=None,
            file_info=file_info,
            n_clusters=n_clusters,
            use_turbo=use_turbo,
            batch_size=batch_size,
            document_key_column=document_key_column,
            token_limit=token_limit,
            retrieval_metric=retrieval_metric,
            apply_default_preprocessing=apply_default_preprocessing
        )
        return {"mode": "config1", "summary": result}
    
    # Handle file input
    elif file:
        # Create temporary file and stream upload directly to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            file_size = os.path.getsize(tmp_path)
            
            # Process directly from filesystem for large files
            if file_size > LARGE_FILE_THRESHOLD and process_large_files:
                result = process_file_direct(
                    tmp_path, 
                    processing_mode="config1",
                    use_turbo=use_turbo,
                    batch_size=batch_size
                )
                
                # Add file info to result
                if 'file_info' not in result:
                    result["file_info"] = {
                        "filename": file.filename,
                        "file_size": file_size,
                        "upload_time": pd.Timestamp.now().isoformat(),
                        "large_file_processed": True,
                        "turbo_mode": use_turbo,
                        "batch_size": batch_size
                    }
                
                return {"mode": "config1", "summary": result}
            else:
                # For smaller files, use existing pipeline
                df = pd.read_csv(tmp_path)
                file_info = {
                    "filename": file.filename,
                    "file_size": file_size,
                    "upload_time": pd.Timestamp.now().isoformat()
                }
                
                result = run_config1_pipeline(
                    df, 
                    chunk_method=chunk_method,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    model_choice=model_choice,
                    storage_choice=storage_choice,
                    db_config=None,
                    file_info=file_info,
                    n_clusters=n_clusters,
                    use_turbo=use_turbo,
                    batch_size=batch_size,
                    document_key_column=document_key_column,
                    token_limit=token_limit,
                    retrieval_metric=retrieval_metric,
                    apply_default_preprocessing=apply_default_preprocessing
                )
                
                return {"mode": "config1", "summary": result}
                
        finally:
            # Always clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    else:
        return {"error": "Either file upload or database parameters required"}

async def internal_run_deep_config(file, db_type, host, port, username, password, database, table_name,
                                 preprocessing_config, chunking_config, embedding_config, storage_config,
                                 process_large_files, use_turbo, batch_size):
    """Internal handler for Deep Config Mode processing"""
    # Handle database input
    if db_type and host and table_name and db_type != "sqlite":
        if db_type == "mysql":
            conn = connect_mysql(host, port, username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, port, username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        
        # Use chunked import for large tables
        file_size = get_table_size(conn, table_name)
        if file_size > LARGE_FILE_THRESHOLD:
            df = import_large_table_to_dataframe(conn, table_name)
        else:
            df = import_table_to_dataframe(conn, table_name)
            
        conn.close()
        file_info = {"source": f"db:{db_type}", "table": table_name, "size": file_size}
        
        # Parse config dictionaries
        preprocessing_dict = json.loads(preprocessing_config) if preprocessing_config else {}
        chunking_dict = json.loads(chunking_config) if chunking_config else {}
        embedding_dict = json.loads(embedding_config) if embedding_config else {}
        storage_dict = json.loads(storage_config) if storage_config else {}
        
        result = run_deep_config_pipeline(df, {
            "preprocessing": preprocessing_dict,
            "chunking": chunking_dict,
            "embedding": embedding_dict,
            "storage": storage_dict
        }, file_info)
        return {"mode": "deep_config", "summary": result}
    
    # Handle file input
    elif file:
        # Create temporary file and stream upload directly to disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            file_size = os.path.getsize(tmp_path)
            
            # Process directly from filesystem for large files
            if file_size > LARGE_FILE_THRESHOLD and process_large_files:
                result = process_large_file(
                    tmp_path, 
                    processing_mode="deep_config",
                    preprocessing_config=preprocessing_dict,
                    chunking_config=chunking_dict,
                    embedding_config=embedding_dict,
                    storage_config=storage_dict,
                    use_turbo=use_turbo,
                    batch_size=batch_size
                )
                
                # Add file info to result
                if 'file_info' not in result:
                    result["file_info"] = {
                        "filename": file.filename,
                        "file_size": file_size,
                        "upload_time": pd.Timestamp.now().isoformat(),
                        "large_file_processed": True,
                        "turbo_mode": use_turbo,
                        "batch_size": batch_size
                    }
                
                return {"mode": "deep_config", "summary": result}
            else:
                # For smaller files, use existing pipeline
                df = pd.read_csv(tmp_path)
                file_info = {
                    "filename": file.filename,
                    "file_size": file_size,
                    "upload_time": pd.Timestamp.now().isoformat()
                }
                
                # Parse config dictionaries
                preprocessing_dict = json.loads(preprocessing_config) if preprocessing_config else {}
                chunking_dict = json.loads(chunking_config) if chunking_config else {}
                embedding_dict = json.loads(embedding_config) if embedding_config else {}
                storage_dict = json.loads(storage_config) if storage_config else {}
                
                result = run_deep_config_pipeline(df, {
                    "preprocessing": preprocessing_dict,
                    "chunking": chunking_dict,
                    "embedding": embedding_dict,
                    "storage": storage_dict
                }, file_info)
                
                return {"mode": "deep_config", "summary": result}
                
        finally:
            # Always clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    else:
        return {"error": "Either file upload or database parameters required"}

async def internal_deep_config_step(step, file, db_type, host, port, username, password, database, table_name,
                                   type_conversions, null_strategies, duplicate_strategy, remove_stopwords,
                                   text_processing, chunk_method, chunk_size, overlap, key_column, token_limit,
                                   n_clusters, preserve_headers, store_metadata, numeric_columns, categorical_columns,
                                   model_choice, storage_choice, batch_size, use_turbo):
    """Internal handler for Deep Config step-by-step operations"""
    # Route to specific deep config step
    if step == "preprocess":
        return await deep_config_preprocess(file, db_type, host, port, username, password, database, table_name)
    elif step == "type_convert":
        return await deep_config_type_convert(type_conversions)
    elif step == "null_handle":
        return await deep_config_null_handle(null_strategies)
    elif step == "duplicates":
        return await deep_config_duplicates(duplicate_strategy)
    elif step == "stopwords":
        return await deep_config_stopwords(remove_stopwords)
    elif step == "normalize":
        return await deep_config_normalize(text_processing)
    elif step == "chunk":
        return await deep_config_chunk(chunk_method, chunk_size, overlap, key_column, token_limit,
                                      n_clusters, store_metadata, numeric_columns, categorical_columns)
    elif step == "embed":
        return await deep_config_embed(model_choice, batch_size=batch_size, use_parallel=use_turbo)
    elif step == "store":
        return await deep_config_store(storage_choice, "deep_config_collection")
    else:
        return {"error": f"Unknown deep config step: {step}"}

async def internal_retrieve(query, k, metadata_filter):
    """Internal handler for retrieval operations"""
    if not query:
        return {"error": "Query parameter is required for retrieval"}
    
    try:
        metadata_filter_dict = json.loads(metadata_filter) if metadata_filter else {}
        
        # Use existing retrieve logic
        if metadata_filter_dict:
            return await retrieve_with_metadata(query, k, metadata_filter)
        else:
            return await retrieve(query, k)
    
    except Exception as e:
        return {"error": str(e)}

async def internal_export(export_type):
    """Internal handler for export operations"""
    try:
        if export_type == "chunks":
            return await export_chunks_file()
        elif export_type == "embeddings":
            return await export_embeddings_file()
        elif export_type == "embeddings_text":
            return await export_embeddings_text_file()
        elif export_type == "preprocessed":
            return await export_deep_config_preprocessed()
        elif export_type == "deep_chunks":
            return await export_deep_config_chunks()
        elif export_type == "deep_embeddings":
            return await export_deep_config_embeddings()
        else:
            return {"error": f"Unknown export type: {export_type}"}
    
    except Exception as e:
        return {"error": str(e)}

async def internal_system(system_action):
    """Internal handler for system operations"""
    try:
        if system_action == "info":
            return await system_info()
        elif system_action == "file_info":
            return await file_info()
        elif system_action == "health":
            return await health_check()
        elif system_action == "capabilities":
            return await capabilities()
        else:
            return {"error": f"Unknown system action: {system_action}"}
    
    except Exception as e:
        return {"error": str(e)}

async def internal_db_test(db_type, host, port, username, password, database):
    """Internal handler for database connection test"""
    try:
        return await db_test_connection(db_type, host, port, username, password, database)
    except Exception as e:
        return {"error": str(e)}

async def internal_db_list(db_type, host, port, username, password, database):
    """Internal handler for database table listing"""
    try:
        return await db_list_tables(db_type, host, port, username, password, database)
    except Exception as e:
        return {"error": str(e)}

async def internal_db_import(db_type, host, port, username, password, database, table_name,
                            chunk_method, chunk_size, overlap, model_choice, storage_choice,
                            use_turbo, batch_size):
    """Internal handler for database table import"""
    try:
        return await db_import_one(db_type, host, port, username, password, database, table_name,
                                 "config1", use_turbo, batch_size)
    except Exception as e:
        return {"error": str(e)}

# ---------------------------
# 🎯 CAMPAIGN MODE ENDPOINTS
# ---------------------------

@app.post("/campaign/run")
async def run_campaign_endpoint(
    file: Optional[UploadFile] = File(None),
    db_type: Optional[str] = Form(None),
    host: Optional[str] = Form(None),
    port: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    database: Optional[str] = Form(None),
    table_name: Optional[str] = Form(None),
    chunk_method: str = Form("record_based"),
    chunk_size: str = Form("5"),
    model_choice: str = Form("paraphrase-MiniLM-L6-v2"),
    storage_choice: str = Form("faiss"),
    use_openai: str = Form("false"),
    openai_api_key: Optional[str] = Form(None),
    openai_base_url: Optional[str] = Form(None),
    process_large_files: str = Form("true"),
    use_turbo: str = Form("true"),
    batch_size: str = Form("256"),
    preserve_record_structure: str = Form("true"),
    document_key_column: Optional[str] = Form(None)
):
    """Campaign mode pipeline - specialized for media campaign data"""
    try:
        use_openai_bool = use_openai.lower() == "true"
        process_large_bool = process_large_files.lower() == "true"
        use_turbo_bool = use_turbo.lower() == "true"
        preserve_bool = preserve_record_structure.lower() == "true"
        batch_size_int = int(batch_size)
        chunk_size_int = int(chunk_size)
        
        # Database import path
        if all([db_type, host, port, username, database, table_name]):
            if db_type == "mysql":
                conn = campaign_connect_mysql(host, int(port), username, password, database)
            elif db_type == "postgresql":
                conn = campaign_connect_postgresql(host, int(port), username, password, database)
            else:
                raise HTTPException(status_code=400, detail="Unsupported database type")
            
            df = campaign_import_table(conn, table_name)
            conn.close()
            
            file_info = {
                "filename": f"{table_name}_from_{database}",
                "file_size": len(df),
                "upload_time": "Database import",
                "location": "Database",
                "data_type": "campaign"
            }
            
            result = run_campaign_pipeline(
                df,
                chunk_method=chunk_method,
                chunk_size=chunk_size_int,
                model_choice=model_choice,
                storage_choice=storage_choice,
                file_info=file_info,
                use_openai=use_openai_bool,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                use_turbo=use_turbo_bool,
                batch_size=batch_size_int,
                preserve_record_structure=preserve_bool,
                document_key_column=document_key_column
            )
            
            return {"mode": "campaign", "summary": result}
        
        # File upload path
        elif file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_path = tmp_file.name
            
            file_info = {
                "filename": file.filename,
                "file_size": os.path.getsize(temp_path),
                "upload_time": "File upload",
                "location": "Temporary storage",
                "data_type": "campaign"
            }
            
            # Large file handling
            if process_large_bool and not campaign_can_load_file(file_info['file_size']):
                result = campaign_process_file_direct(
                    temp_path,
                    chunk_method=chunk_method,
                    chunk_size=chunk_size_int,
                    model_choice=model_choice,
                    storage_choice=storage_choice,
                    use_openai=use_openai_bool,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                    use_turbo=use_turbo_bool,
                    batch_size=batch_size_int,
                    document_key_column=document_key_column
                )
                os.unlink(temp_path)
                return {"mode": "campaign", "summary": result, "large_file_processed": True}
            else:
                df = pd.read_csv(temp_path)
                os.unlink(temp_path)
                
                result = run_campaign_pipeline(
                    df,
                    chunk_method=chunk_method,
                    chunk_size=chunk_size_int,
                    model_choice=model_choice,
                    storage_choice=storage_choice,
                    file_info=file_info,
                    use_openai=use_openai_bool,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                    use_turbo=use_turbo_bool,
                    batch_size=batch_size_int,
                    preserve_record_structure=preserve_bool,
                    document_key_column=document_key_column
                )
                
                return {"mode": "campaign", "summary": result}
        
        else:
            raise HTTPException(status_code=400, detail="No file or database config provided")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Campaign pipeline error: {str(e)}")

@app.post("/campaign/retrieve")
async def campaign_retrieve_endpoint(
    query: str = Form(...),
    search_field: str = Form("all"),
    k: int = Form(5),
    include_complete_records: str = Form("true")
):
    """Standard campaign retrieval"""
    try:
        include_complete_bool = include_complete_records.lower() == "true"
        result = campaign_retrieve(query, search_field, k, include_complete_bool)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Campaign retrieval error: {str(e)}")

@app.post("/campaign/smart_retrieval")
async def campaign_smart_retrieval_endpoint(
    query: str = Form(...),
    search_field: str = Form("auto"),
    k: int = Form(5),
    include_complete_records: str = Form("true")
):
    """SMART two-stage company retrieval"""
    try:
        include_complete_bool = include_complete_records.lower() == "true"
        result = campaign_smart_retrieval(query, search_field, k, include_complete_bool)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Smart retrieval error: {str(e)}")

@app.get("/campaign/export/chunks")
async def campaign_export_chunks_endpoint():
    """Export campaign chunks"""
    try:
        chunks_data = campaign_export_chunks()
        return JSONResponse(content={"data": chunks_data})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/campaign/export/embeddings")
async def campaign_export_embeddings_endpoint():
    """Export campaign embeddings"""
    try:
        embeddings_data = campaign_export_embeddings()
        return JSONResponse(content=embeddings_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/campaign/export/preprocessed")
async def campaign_export_preprocessed_endpoint():
    """Export campaign preprocessed data"""
    try:
        preprocessed_text = campaign_export_preprocessed()
        return JSONResponse(content={"preprocessed_data": preprocessed_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/campaign/info")
async def campaign_info_endpoint():
    """Get campaign processing info"""
    try:
        return {
            "mode": "campaign",
            "chunks_available": campaign_state['chunks'] is not None,
            "chunks_count": len(campaign_state['chunks']) if campaign_state['chunks'] else 0,
            "model_loaded": campaign_state['model'] is not None,
            "retrieval_ready": all([
                campaign_state['model'] is not None,
                campaign_state['store_info'] is not None,
                campaign_state['chunks'] is not None
            ])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Info error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)        