# main.py - MEDIA CAMPAIGN ONLY VERSION - FIXED RETRIEVAL STRUCTURE
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Union, Dict, Any
import pandas as pd
import io
import numpy as np
import tempfile
import os
import json
import uvicorn
import shutil
import logging

from backend1 import (
    run_media_campaign_pipeline,
    retrieve_media_campaign,
    smart_company_retrieval,
    export_chunks,
    export_embeddings,
    export_preprocessed_data,
    get_system_info,
    get_file_info,
    connect_mysql,
    connect_postgresql,
    get_table_list,
    import_table_to_dataframe,
    process_file_direct,
    can_load_file,
    EMBEDDING_BATCH_SIZE,
    PARALLEL_WORKERS,
    OpenAIEmbeddingAPI
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Media Campaign Processor API", version="3.0")

# ---------------------------
# Root Endpoint
# ---------------------------
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Media Campaign Processor API",
        "version": "3.0",
        "status": "running",
        "mode": "media_campaign_only",
        "endpoints": {
            "health": "/health",
            "capabilities": "/capabilities",
            "system_info": "/system_info",
            "file_info": "/file_info",
            "processing": {
                "run_media_campaign": "/run_media_campaign (POST)",
                "retrieve_media_campaign": "/retrieve_media_campaign (POST)",
                "smart_company_retrieval": "/smart_company_retrieval (POST)",
                "batch_retrieve": "/batch_retrieve (POST)"
            },
            "database": {
                "test_connection": "/db/test_connection (POST)",
                "list_tables": "/db/list_tables (POST)"
            },
            "export": {
                "chunks": "/export/chunks (GET)",
                "embeddings": "/export/embeddings_json (GET)",
                "preprocessed": "/export/preprocessed (GET)"
            },
            "openai_compatible": {
                "embeddings": "/v1/embeddings (POST)",
                "chat_completions": "/v1/chat/completions (POST)",
                "retrieve": "/v1/retrieve (POST)"
            }
        },
        "docs": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "message": "API is running successfully! Visit /docs for interactive documentation"
    }

# ---------------------------
# Request Models
# ---------------------------
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None

class MediaCampaignRetrievalRequest(BaseModel):
    query: str
    search_field: Optional[str] = "all"
    k: int = 5
    include_complete_records: bool = True

# ---------------------------
# Authentication & Model Mapping
# ---------------------------
OPENAI_MODEL_MAPPING = {
    "text-embedding-ada-002": "paraphrase-MiniLM-L6-v2",
    "text-embedding-3-small": "all-MiniLM-L6-v2", 
    "text-embedding-3-large": "all-MiniLM-L6-v2",
    "gpt-3.5-turbo": "local",
    "gpt-4": "local"
}

async def get_api_key(authorization: Optional[str] = Header(None)):
    """Extract and validate API key from header"""
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "")
        return api_key
    return None

def get_local_model(openai_model: str) -> str:
    """Map OpenAI model names to local models"""
    return OPENAI_MODEL_MAPPING.get(openai_model, "paraphrase-MiniLM-L6-v2")

# ---------------------------
# OpenAI-Compatible API Endpoints
# ---------------------------
@app.post("/v1/embeddings")
async def openai_embeddings_compatible(
    request: EmbeddingRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """OpenAI-compatible embeddings endpoint with JSON body"""
    try:
        local_model = get_local_model(request.model)
        
        if isinstance(request.input, str):
            texts = [request.input]
        else:
            texts = request.input
            
        embedding_api = OpenAIEmbeddingAPI(
            model_name=local_model,
            api_key=api_key,
            base_url=None
        )
        
        embeddings = embedding_api.encode(texts)
        total_tokens = sum(len(text.split()) for text in texts)
        
        response_data = {
            "object": "list",
            "data": [],
            "model": request.model,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
        
        for i, embedding in enumerate(embeddings):
            response_data["data"].append({
                "object": "embedding",
                "embedding": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                "index": i
            })
            
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"OpenAI embeddings error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": {
                    "message": f"Embedding error: {str(e)}",
                    "type": "internal_error",
                    "code": "embedding_failed"
                }
            }
        )

@app.post("/v1/chat/completions")
async def openai_chat_completions_compatible(
    request: ChatCompletionRequest,
    api_key: Optional[str] = Depends(get_api_key)
):
    """OpenAI-compatible chat completions endpoint"""
    try:
        assistant_message = "I'm a local AI assistant. This is a mock response since chat completion is not fully implemented yet."
        
        response_data = {
            "id": "chatcmpl-local-mock",
            "object": "chat.completion",
            "created": 1677858242,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_message,
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(msg["content"].split()) for msg in request.messages),
                "completion_tokens": len(assistant_message.split()),
                "total_tokens": sum(len(msg["content"].split()) for msg in request.messages) + len(assistant_message.split())
            }
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"OpenAI chat completions error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": {
                    "message": f"Chat completion error: {str(e)}",
                    "type": "internal_error", 
                    "code": "chat_completion_failed"
                }
            }
        )

# ---------------------------
# Media Campaign Endpoints
# ---------------------------
@app.post("/run_media_campaign")
async def run_media_campaign_endpoint(
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
    """Enhanced Media Campaign pipeline with all chunking methods"""
    try:
        logger.info("Starting Media Campaign pipeline...")
        
        use_openai_bool = use_openai.lower() == "true"
        process_large_files_bool = process_large_files.lower() == "true"
        use_turbo_bool = use_turbo.lower() == "true"
        preserve_structure_bool = preserve_record_structure.lower() == "true"
        batch_size_int = int(batch_size)
        chunk_size_int = int(chunk_size)
        
        logger.info(f"🔍 Database config check - db_type: {db_type}, host: {host}, port: {port}, username: {username}, database: {database}, table_name: {table_name}")
        
        has_complete_db_config = all([db_type, host, port, username, database, table_name])
        
        if has_complete_db_config:
            logger.info(f"🔄 Database import from {db_type}: {database}.{table_name}")
            
            try:
                if db_type == "mysql":
                    conn = connect_mysql(host, int(port), username, password, database)
                elif db_type == "postgresql":
                    conn = connect_postgresql(host, int(port), username, password, database)
                else:
                    raise HTTPException(status_code=400, detail="Unsupported database type")
                
                df = import_table_to_dataframe(conn, table_name)
                conn.close()
                
                file_info = {
                    "filename": f"{table_name}_from_{database}",
                    "file_size": len(df),
                    "upload_time": "Database import",
                    "location": "Database",
                    "data_type": "media_campaign"
                }
                
                logger.info(f"✅ Imported {len(df)} rows from database {database}.{table_name}")
                
                logger.info("Running Media Campaign pipeline processing with database data...")
                result = run_media_campaign_pipeline(
                    df,
                    chunk_method=chunk_method,
                    chunk_size=chunk_size_int,
                    model_choice=model_choice,
                    storage_choice=storage_choice,
                    db_config=None,
                    file_info=file_info,
                    use_openai=use_openai_bool,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                    use_turbo=use_turbo_bool,
                    batch_size=batch_size_int,
                    preserve_record_structure=preserve_structure_bool,
                    document_key_column=document_key_column
                )
                
                logger.info("✅ Media Campaign pipeline completed successfully with database import")
                return {
                    "mode": "media_campaign",
                    "summary": result
                }
                
            except Exception as db_error:
                logger.error(f"❌ Database import failed: {str(db_error)}")
                raise HTTPException(status_code=500, detail=f"Database import failed: {str(db_error)}")
                
        elif file:
            logger.info(f"🔄 File upload: {file.filename}")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_path = tmp_file.name
            
            file_info = {
                "filename": file.filename,
                "file_size": os.path.getsize(temp_path),
                "upload_time": "File upload",
                "location": "Temporary storage",
                "data_type": "media_campaign"
            }
            
            if process_large_files_bool and not can_load_file(file_info['file_size']):
                logger.info("Processing as large file with disk streaming")
                config = {
                    "chunk_method": chunk_method,
                    "chunk_size": chunk_size_int,
                    "model_choice": model_choice,
                    "storage_choice": storage_choice,
                    "preserve_record_structure": preserve_structure_bool,
                    "document_key_column": document_key_column
                }
                
                result = process_file_direct(
                    temp_path,
                    **config,
                    use_openai=use_openai_bool,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                    use_turbo=use_turbo_bool,
                    batch_size=batch_size_int
                )
                
                os.unlink(temp_path)
                
                logger.info("✅ Media Campaign pipeline completed (large file)")
                return {
                    "mode": "media_campaign",
                    "summary": result,
                    "large_file_processed": True
                }
            else:
                df = pd.read_csv(temp_path)
                os.unlink(temp_path)
                logger.info(f"✅ Loaded {len(df)} rows from file")
                
                logger.info("Running Media Campaign pipeline processing with file data...")
                result = run_media_campaign_pipeline(
                    df,
                    chunk_method=chunk_method,
                    chunk_size=chunk_size_int,
                    model_choice=model_choice,
                    storage_choice=storage_choice,
                    db_config=None,
                    file_info=file_info,
                    use_openai=use_openai_bool,
                    openai_api_key=openai_api_key,
                    openai_base_url=openai_base_url,
                    use_turbo=use_turbo_bool,
                    batch_size=batch_size_int,
                    preserve_record_structure=preserve_structure_bool,
                    document_key_column=document_key_column
                )
                
                logger.info("✅ Media Campaign pipeline completed successfully with file")
                return {
                    "mode": "media_campaign",
                    "summary": result
                }
            
        else:
            missing_params = []
            if not db_type: missing_params.append("db_type")
            if not host: missing_params.append("host") 
            if not port: missing_params.append("port")
            if not username: missing_params.append("username")
            if not database: missing_params.append("database")
            if not table_name: missing_params.append("table_name")
            
            error_msg = f"Incomplete database configuration. Missing parameters: {', '.join(missing_params)}. Also no file provided."
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
    except Exception as e:
        logger.error(f"❌ Media Campaign pipeline error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Media Campaign pipeline error: {str(e)}")

@app.post("/retrieve_media_campaign")
async def retrieve_media_campaign_endpoint(
    query: str = Form(...),
    search_field: str = Form("all"),
    k: int = Form(5),
    include_complete_records: str = Form("true")
):
    """Specialized retrieval for media campaign data"""
    try:
        include_complete_bool = include_complete_records.lower() == "true"
        result = retrieve_media_campaign(query, search_field, k, include_complete_bool)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Media campaign retrieval error: {str(e)}")

# SMART Company Retrieval Endpoint
@app.post("/smart_company_retrieval")
async def smart_company_retrieval_endpoint(
    query: str = Form(...),
    search_field: str = Form("auto"),
    k: int = Form(5),
    include_complete_records: str = Form("true")
):
    """SMART TWO-STAGE RETRIEVAL for company searches"""
    try:
        include_complete_bool = include_complete_records.lower() == "true"
        result = smart_company_retrieval(query, search_field, k, include_complete_bool)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Smart company retrieval error: {str(e)}")

# ---------------------------
# Retrieval Endpoints
# ---------------------------
@app.post("/v1/retrieve")
async def openai_retrieve_endpoint(
    query: str = Form(...),
    model: str = Form("all-MiniLM-L6-v2"),
    n_results: int = Form(5)
):
    try:
        result = retrieve_media_campaign(query, "all", n_results)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

# ---------------------------
# Export Endpoints
# ---------------------------
@app.get("/export/chunks")
async def export_chunks_endpoint():
    try:
        chunks_data = export_chunks()
        return JSONResponse(content=chunks_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/export/embeddings_json")
async def export_embeddings_json_endpoint():
    try:
        embeddings_data = export_embeddings()
        return JSONResponse(content=embeddings_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/export/preprocessed")
async def export_preprocessed_endpoint():
    try:
        preprocessed_text = export_preprocessed_data()
        return JSONResponse(content={"preprocessed_data": preprocessed_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

# ---------------------------
# Information Endpoints
# ---------------------------
@app.get("/system_info")
async def system_info_endpoint():
    try:
        info = get_system_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System info error: {str(e)}")

@app.get("/file_info")
async def file_info_endpoint():
    try:
        info = get_file_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File info error: {str(e)}")

@app.get("/capabilities")
async def capabilities_endpoint():
    return {
        "large_file_support": True,
        "max_file_size": "3GB+",
        "performance_features": {
            "turbo_mode": True,
            "batch_processing": True,
            "parallel_workers": PARALLEL_WORKERS,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE
        },
        "embedding_models": ["paraphrase-MiniLM-L6-v2", "all-MiniLM-L6-v2", "text-embedding-ada-002"],
        "vector_stores": ["faiss", "chroma"],
        "chunking_methods": ["record_based", "company_based", "source_based", "semantic_clustering", "document_based"],
        "processing_modes": ["media_campaign"],
        "openai_compatible": True,
        "media_campaign_support": True,
        "smart_company_retrieval": True,
        "smart_column_display": True,
        "document_chunking_columns": True,
        "semantic_clustering": True
    }

@app.get("/health")
async def health_endpoint():
    return {"status": "healthy", "version": "3.0", "mode": "media_campaign_only"}

# ---------------------------
# Database Endpoints
# ---------------------------
@app.post("/db/test_connection")
async def db_test_connection_endpoint(
    db_type: str = Form(...),
    host: str = Form(...),
    port: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    try:
        logger.info(f"Testing {db_type} connection to {host}:{port}")
        if db_type == "mysql":
            conn = connect_mysql(host, int(port), username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, int(port), username, password, database)
        else:
            return {"status": "error", "message": "Unsupported db_type"}
        conn.close()
        logger.info(f"✅ {db_type} connection successful to {host}:{port}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"❌ {db_type} connection failed: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/db/list_tables")
async def db_list_tables_endpoint(
    db_type: str = Form(...),
    host: str = Form(...),
    port: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    database: str = Form(...)
):
    try:
        logger.info(f"Listing tables for {db_type} at {host}:{port}")
        if db_type == "mysql":
            conn = connect_mysql(host, int(port), username, password, database)
        elif db_type == "postgresql":
            conn = connect_postgresql(host, int(port), username, password, database)
        else:
            return {"error": "Unsupported db_type"}
        tables = get_table_list(conn, db_type)
        conn.close()
        
        if db_type == "postgresql":
            system_tables = ['pg_', 'sql_', 'information_schema', 'system_']
            tables = [table for table in tables if not any(table.startswith(prefix) for prefix in system_tables)]
        elif db_type == "mysql":
            system_tables = ['mysql', 'information_schema', 'performance_schema', 'sys']
            tables = [table for table in tables if table not in system_tables]
        
        logger.info(f"✅ Found {len(tables)} tables in {database}")
        return {"tables": tables}
    except Exception as e:
        logger.error(f"❌ Failed to list tables: {str(e)}")
        return {"error": str(e)}

# ---------------------------
# Enhanced Retrieval Endpoint with Consistent Structure
# ---------------------------
@app.post("/v1/retrieve_enhanced")
async def enhanced_retrieve_endpoint(
    query: str = Form(...),
    search_field: str = Form("all"),
    k: int = Form(5),
    include_complete_records: str = Form("true"),
    use_smart_retrieval: str = Form("true")
):
    """Enhanced retrieval endpoint with consistent structure"""
    try:
        include_complete_bool = include_complete_records.lower() == "true"
        use_smart_bool = use_smart_retrieval.lower() == "true"
        
        if use_smart_bool:
            result = smart_company_retrieval(query, search_field, k, include_complete_bool)
        else:
            result = retrieve_media_campaign(query, search_field, k, include_complete_bool)
        
        # Ensure consistent structure
        if 'error' not in result:
            if 'results' in result:
                for res in result['results']:
                    # Ensure all result entries have consistent fields
                    if 'rank' not in res:
                        res['rank'] = result['results'].index(res) + 1
                    if 'similarity' not in res:
                        res['similarity'] = 0.0
                    if 'content' not in res:
                        res['content'] = ""
                    if 'complete_record' not in res:
                        res['complete_record'] = []
            
            # Ensure retrieval method is set
            if 'retrieval_method' not in result:
                result['retrieval_method'] = 'semantic_search'
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced retrieval error: {str(e)}")

# ---------------------------
# Status and Monitoring Endpoints
# ---------------------------
@app.get("/status")
async def status_endpoint():
    """Get current processing status"""
    try:
        from backend import (
            current_model, current_store_info, current_chunks, 
            current_embeddings, current_media_campaign_data
        )
        
        status = {
            "model_loaded": current_model is not None,
            "store_available": current_store_info is not None,
            "chunks_available": current_chunks is not None and len(current_chunks) > 0,
            "embeddings_available": current_embeddings is not None,
            "media_campaign_data_available": current_media_campaign_data is not None,
            "chunks_count": len(current_chunks) if current_chunks else 0,
            "retrieval_ready": all([
                current_model is not None,
                current_store_info is not None,
                current_chunks is not None,
                len(current_chunks) > 0
            ])
        }
        
        return status
    except Exception as e:
        return {"error": f"Status check failed: {str(e)}"}

@app.get("/current_config")
async def current_config_endpoint():
    """Get current pipeline configuration"""
    try:
        from backend import (
            current_media_campaign_data, current_file_info
        )
        
        config = {
            "file_info": current_file_info or {},
            "media_campaign_config": {
                "field_mapping": current_media_campaign_data.get('field_mapping', {}) if current_media_campaign_data else {},
                "processed_rows": current_media_campaign_data.get('processed_df', pd.DataFrame()).shape[0] if current_media_campaign_data else 0,
                "metadata_available": len(current_media_campaign_data.get('metadata', [])) if current_media_campaign_data else 0
            } if current_media_campaign_data else {}
        }
        
        return config
    except Exception as e:
        return {"error": f"Config check failed: {str(e)}"}

# ---------------------------
# Batch Processing Endpoints
# ---------------------------
@app.post("/batch_retrieve")
async def batch_retrieve_endpoint(
    queries: str = Form(...),  # JSON string of queries
    search_field: str = Form("all"),
    k: int = Form(5),
    include_complete_records: str = Form("true")
):
    """Batch retrieval for multiple queries"""
    try:
        include_complete_bool = include_complete_records.lower() == "true"
        
        # Parse queries JSON
        try:
            queries_list = json.loads(queries)
            if not isinstance(queries_list, list):
                queries_list = [queries_list]
        except json.JSONDecodeError:
            queries_list = [queries]
        
        results = []
        for query in queries_list:
            if isinstance(query, dict):
                query_text = query.get('query', '')
                current_k = query.get('k', k)
                current_field = query.get('search_field', search_field)
            else:
                query_text = str(query)
                current_k = k
                current_field = search_field
            
            result = retrieve_media_campaign(query_text, current_field, current_k, include_complete_bool)
            results.append({
                "query": query_text,
                "result": result
            })
        
        return {
            "batch_results": results,
            "total_queries": len(queries_list),
            "processing_time": "Batch processing completed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch retrieval error: {str(e)}")

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001
    )