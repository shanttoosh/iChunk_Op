# backend_campaign.py - CAMPAIGN MODE FOR ICHUNKOPTIMIZER (ISOLATED STATE)
# Specialized for media campaign & contact data with smart company retrieval
import pandas as pd
import numpy as np
import re
import time
import os
import psutil
import shutil
from typing import List, Dict, Any, Optional, Tuple, Union
import chromadb
import faiss
# Removed langchain dependency - using custom implementation from backend.py
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json
import gc
from pathlib import Path
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Custom RecursiveCharacterTextSplitter implementation (replaces langchain dependency)
class RecursiveCharacterTextSplitter:
    """Simple text splitter that recursively splits text using different separators."""
    
    def __init__(self, chunk_size=400, chunk_overlap=50, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using recursive separators."""
        if not text:
            return []
        
        chunks = []
        current_chunk = ""
        
        # Try to split by separators in order
        for separator in self.separators:
            if separator and separator in text:
                splits = text.split(separator)
                for split in splits:
                    if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                        if current_chunk:
                            current_chunk += separator + split
                        else:
                            current_chunk = split
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = split
                
                if current_chunk:
                    chunks.append(current_chunk)
                return chunks
        
        # Fallback: split by character count if no separator works
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk:
                chunks.append(chunk)
        
        return chunks

# -----------------------------
# 🔹 ISOLATED STATE for Campaign Mode
# -----------------------------
campaign_state = {
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
}

# -----------------------------
# 🔹 Performance Optimization Configuration
# -----------------------------
LARGE_FILE_THRESHOLD = 100 * 1024 * 1024  # 100MB
MAX_MEMORY_USAGE = 0.8  # 80% of available memory
BATCH_SIZE = 2000
EMBEDDING_BATCH_SIZE = 256
PARALLEL_WORKERS = 6
CACHE_DIR = "processing_cache"

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

# -----------------------------
# 🔹 Database Connection Helpers
# -----------------------------
def connect_mysql(host: str, port: int, username: str, password: str, database: str):
    import mysql.connector
    return mysql.connector.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        database=database
    )

def connect_postgresql(host: str, port: int, username: str, password: str, database: str):
    import psycopg2
    return psycopg2.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        dbname=database
    )

def get_table_list(conn, db_type: str):
    """Dynamic table discovery"""
    fallback_queries = {
        "mysql": [
            "SHOW TABLES",
            "SELECT table_name FROM information_schema.tables WHERE table_schema=DATABASE()",
        ],
        "postgresql": [
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'",
            "SELECT tablename FROM pg_tables WHERE schemaname='public'",
        ]
    }
    
    queries = fallback_queries.get(db_type, [])
    
    for query in queries:
        try:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            cur.close()
            if rows:
                tables = []
                for row in rows:
                    if isinstance(row, (list, tuple)) and len(row) > 0:
                        tables.append(str(row[0]))
                    elif isinstance(row, dict):
                        tables.append(str(list(row.values())[0]))
                    else:
                        tables.append(str(row))
                
                # Filter out system tables
                if db_type == "postgresql":
                    system_tables = ['pg_', 'sql_', 'information_schema', 'system_']
                    tables = [table for table in tables if not any(table.startswith(prefix) for prefix in system_tables)]
                elif db_type == "mysql":
                    system_tables = ['mysql', 'information_schema', 'performance_schema', 'sys']
                    tables = [table for table in tables if table not in system_tables]
                
                logger.info(f"✅ Found {len(tables)} tables: {tables}")
                return tables
        except Exception as e:
            logger.warning(f"Query failed: {query}, Error: {e}")
            continue
    
    logger.warning("No tables found or all queries failed")
    return []

def import_table_to_dataframe(conn, table_name: str) -> pd.DataFrame:
    """Import table to dataframe"""
    try:
        query = f"SELECT * FROM {table_name}"
        logger.info(f"Executing query: {query}")
        df = pd.read_sql(query, conn)
        logger.info(f"✅ Successfully imported {len(df)} rows from table '{table_name}'")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to import table '{table_name}': {str(e)}")
        raise e

# -----------------------------
# 🔹 File Handling Functions
# -----------------------------
def get_available_memory():
    """Get available system memory in bytes"""
    return psutil.virtual_memory().available

def can_load_file(file_size: int) -> bool:
    """Check if file can be safely loaded into memory"""
    available_memory = get_available_memory()
    return file_size < available_memory * MAX_MEMORY_USAGE

def campaign_set_file_info(file_info: Dict):
    """Store file information"""
    campaign_state['file_info'] = file_info

def campaign_get_file_info():
    """Get stored file information"""
    return campaign_state['file_info'] or {}

# -----------------------------
# 🔹 Text Processing Functions
# -----------------------------
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and assign default names if missing"""
    df_clean = df.copy()
    
    new_columns = []
    for i, col in enumerate(df_clean.columns):
        if pd.isna(col) or col == '' or col is None:
            new_columns.append(f"column_{i+1}")
        else:
            clean_col = re.sub(r'[^\w\s]', '', str(col))
            clean_col = re.sub(r'\s+', '_', clean_col.strip())
            new_columns.append(clean_col if clean_col else f"column_{i+1}")
    
    df_clean.columns = new_columns
    return df_clean

def clean_text_advanced(text_series: pd.Series, lowercase: bool = True, remove_delimiters: bool = True, 
                       remove_whitespace: bool = True, remove_stopwords: bool = True) -> pd.Series:
    """Advanced text cleaning for string columns"""
    cleaned_series = text_series.astype(str)
    
    if lowercase:
        cleaned_series = cleaned_series.str.lower()
    
    if remove_delimiters:
        cleaned_series = cleaned_series.str.replace(r'[^\w\s]', ' ', regex=True)
    
    if remove_whitespace:
        cleaned_series = cleaned_series.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    if remove_stopwords:
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                stop_words = set(stopwords.words('english'))
            except:
                nltk.download('stopwords')
                stop_words = set(stopwords.words('english'))
            
            cleaned_series = cleaned_series.apply(
                lambda text: ' '.join([word for word in text.split() if word not in stop_words])
            )
        except ImportError:
            logger.warning("NLTK not available for stopwords removal")
            basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            cleaned_series = cleaned_series.apply(
                lambda text: ' '.join([word for word in text.split() if word not in basic_stopwords])
            )
    
    return cleaned_series

# -----------------------------
# 🔹 Media Campaign Functions
# -----------------------------
def detect_media_campaign_fields(df: pd.DataFrame) -> Dict[str, str]:
    """Enhanced field detection with better company identification"""
    field_mapping = {}
    field_priority = {
        'company': ['company', 'organization', 'firm', 'employer', 'company_name', 'org_name'],
        'company_name': ['company_name', 'organization_name', 'firm_name'],
        'lead_id': ['lead_id', 'contact_id', 'id', 'record_id'],
        'customer_id': ['customer_id', 'client_id', 'account_id'],
        'email': ['email', 'email_address', 'mail', 'mail_id'],
        'phone': ['phone', 'telephone', 'mobile', 'contact_number', 'ph_no'],
        'lead_source': ['lead_source', 'source', 'acquisition_source'],
        'lead_status': ['lead_status', 'status', 'contact_status'],
        'campaign_source': ['campaign', 'campaign_source', 'marketing_campaign'],
        'timestamp': ['timestamp', 'date', 'time', 'created_at', 'updated_at'],
        'name': ['name', 'contact_name', 'first_name', 'last_name', 'full_name'],
        'address': ['address', 'location', 'city', 'state', 'country']
    }
    
    for col in df.columns:
        col_lower = col.lower().strip()
        mapped = False
        
        # Exact match priority
        for field_type, keywords in field_priority.items():
            for keyword in keywords:
                if col_lower == keyword:
                    field_mapping[col] = field_type
                    mapped = True
                    logger.info(f"🔍 Exact match: '{col}' -> '{field_type}'")
                    break
            if mapped:
                break
        
        # Partial match if no exact match
        if not mapped:
            if any(keyword in col_lower for keyword in ['company', 'org', 'firm', 'employer']):
                if 'name' in col_lower or 'title' in col_lower:
                    field_mapping[col] = 'company_name'
                else:
                    field_mapping[col] = 'company'
                logger.info(f"🔍 Partial match: '{col}' -> '{field_mapping[col]}'")
            elif any(keyword in col_lower for keyword in ['lead', 'contact']):
                if 'source' in col_lower:
                    field_mapping[col] = 'lead_source'
                elif 'status' in col_lower:
                    field_mapping[col] = 'lead_status'
                elif 'id' in col_lower:
                    field_mapping[col] = 'lead_id'
                else:
                    field_mapping[col] = 'other'
            elif any(keyword in col_lower for keyword in ['customer', 'client']):
                field_mapping[col] = 'customer_id'
            elif any(keyword in col_lower for keyword in ['campaign', 'marketing']):
                field_mapping[col] = 'campaign_source'
            elif any(keyword in col_lower for keyword in ['email', 'mail']):
                field_mapping[col] = 'email'
            elif any(keyword in col_lower for keyword in ['phone', 'mobile', 'telephone']):
                field_mapping[col] = 'phone'
            elif any(keyword in col_lower for keyword in ['name', 'contact']):
                field_mapping[col] = 'name'
            elif any(keyword in col_lower for keyword in ['address', 'location']):
                field_mapping[col] = 'address'
            else:
                field_mapping[col] = 'other'
    
    logger.info(f"🔍 Final field mapping: {field_mapping}")
    return field_mapping

def campaign_preprocess(df: pd.DataFrame, field_mapping: Dict[str, str] = None) -> pd.DataFrame:
    """Specialized preprocessing for media campaign data"""
    start_time = time.time()
    
    df_clean = clean_column_names(df)
    
    if field_mapping is None:
        field_mapping = detect_media_campaign_fields(df_clean)
    
    # Store original values for company fields before cleaning
    company_fields = [col for col, field_type in field_mapping.items() if field_type in ['company', 'company_name']]
    original_companies = {}
    for col in company_fields:
        if col in df_clean.columns:
            original_companies[col] = df_clean[col].copy()
    
    for col, field_type in field_mapping.items():
        if col in df_clean.columns:
            if field_type in ['lead_id', 'customer_id']:
                if df_clean[col].isna().any():
                    # Fix deprecated fillna method
                    df_clean[col] = df_clean[col].ffill().bfill()
                    null_count = df_clean[col].isna().sum()
                    if null_count > 0:
                        df_clean[col] = range(1, len(df_clean) + 1)
            elif field_type in ['company', 'company_name', 'lead_source', 'campaign_source']:
                df_clean[col] = df_clean[col].fillna('Unknown')
            elif field_type in ['lead_status']:
                df_clean[col] = df_clean[col].fillna('New')
            elif field_type in ['timestamp']:
                df_clean[col] = df_clean[col].fillna(pd.Timestamp.now())
            else:
                df_clean[col] = df_clean[col].fillna('')
    
    text_fields = [col for col, field_type in field_mapping.items() 
                  if field_type in ['company', 'company_name', 'lead_source', 'lead_status', 'campaign_source', 'other']]
    
    for col in text_fields:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
            # Don't lowercase company fields to preserve exact matching
            if field_mapping[col] not in ['company', 'company_name']:
                df_clean[col] = df_clean[col].str.lower()
    
    logger.info(f"Media campaign preprocessing completed in {time.time() - start_time:.2f}s")
    
    campaign_state['preprocessed_df'] = df_clean.copy()
    campaign_state['media_campaign_data'] = {
        'field_mapping': field_mapping,
        'processed_df': df_clean,
        'original_companies': original_companies
    }
    
    return df_clean

# -----------------------------
# 🔹 ENHANCED CHUNKING FOR MEDIA CAMPAIGN
# -----------------------------
def campaign_chunk_records(df: pd.DataFrame, records_per_chunk: int = 5, field_mapping: Dict[str, str] = None) -> Tuple[List[str], List[Dict]]:
    """Chunk media campaign data by grouping complete contact records"""
    start_time = time.time()
    
    chunks = []
    metadata = []
    
    structured_records = []
    for idx, row in df.iterrows():
        record_text = "CONTACT RECORD:\n"
        record_data = {}
        
        for col in df.columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                field_type = field_mapping.get(col, 'other') if field_mapping else 'other'
                record_text += f"{col}: {value}\n"
                record_data[col] = convert_to_serializable(value)
        
        structured_records.append({
            'text': record_text.strip(),
            'data': record_data,
            'index': int(idx)
        })
    
    for i in range(0, len(structured_records), records_per_chunk):
        end_idx = min(i + records_per_chunk, len(structured_records))
        chunk_group = structured_records[i:end_idx]
        
        chunk_text = f"CONTACT GROUP (Records {i+1}-{end_idx}):\n\n"
        chunk_data = []
        
        for record in chunk_group:
            chunk_text += record['text'] + "\n\n"
            chunk_data.append(record['data'])
        
        chunks.append(chunk_text.strip())
        metadata.append({
            'chunk_type': 'contact_records',
            'record_count': len(chunk_group),
            'start_index': i,
            'end_index': end_idx - 1,
            'records_per_chunk': records_per_chunk,
            'complete_records': chunk_data
        })
    
    logger.info(f"Media campaign record chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks, metadata

def campaign_chunk_by_company(df: pd.DataFrame, field_mapping: Dict[str, str] = None, max_records_per_company: int = 10) -> Tuple[List[str], List[Dict]]:
    """Chunk media campaign data by grouping contacts by company"""
    start_time = time.time()
    
    chunks = []
    metadata = []
    
    # Find the best company column
    company_col = None
    company_cols = [col for col, ftype in field_mapping.items() if ftype in ['company', 'company_name']]
    
    if company_cols:
        if any('name' in col.lower() for col in company_cols):
            company_col = [col for col in company_cols if 'name' in col.lower()][0]
        else:
            company_col = company_cols[0]
    
    if company_col is None:
        logger.warning("No company column found, falling back to record-based chunking")
        return campaign_chunk_records(df, 5, field_mapping)
    
    logger.info(f"Using company column: {company_col}")
    
    grouped = df.groupby(company_col)
    
    for company, group in grouped:
        company_records = []
        
        for idx, row in group.iterrows():
            record_text = f"COMPANY: {company}\n"
            record_data = {'company': company}
            
            for col in group.columns:
                value = row[col]
                if pd.notna(value) and str(value).strip() and col != company_col:
                    record_text += f"{col}: {value}\n"
                    record_data[col] = convert_to_serializable(value)
            
            company_records.append({
                'text': record_text.strip(),
                'data': record_data,
                'index': int(idx)
            })
        
        for i in range(0, len(company_records), max_records_per_company):
            end_idx = min(i + max_records_per_company, len(company_records))
            chunk_group = company_records[i:end_idx]
            
            chunk_text = f"COMPANY CONTACTS: {company} (Records {i+1}-{end_idx}):\n\n"
            chunk_data = []
            
            for record in chunk_group:
                chunk_text += record['text'] + "\n\n"
                chunk_data.append(record['data'])
            
            chunks.append(chunk_text.strip())
            metadata.append({
                'chunk_type': 'company_group',
                'company': company,
                'record_count': len(chunk_group),
                'start_index': i,
                'end_index': end_idx - 1,
                'max_records_per_company': max_records_per_company,
                'complete_records': chunk_data
            })
    
    logger.info(f"Media campaign company chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks, metadata

def campaign_chunk_by_source(df: pd.DataFrame, field_mapping: Dict[str, str] = None, max_records_per_source: int = 8) -> Tuple[List[str], List[Dict]]:
    """Chunk media campaign data by grouping contacts by lead source"""
    start_time = time.time()
    
    chunks = []
    metadata = []
    
    source_col = None
    source_cols = [col for col, ftype in field_mapping.items() if ftype in ['lead_source', 'campaign_source']]
    
    if source_cols:
        source_col = source_cols[0]
    
    if source_col is None:
        logger.warning("No source column found, falling back to record-based chunking")
        return campaign_chunk_records(df, 5, field_mapping)
    
    logger.info(f"Using source column: {source_col}")
    
    grouped = df.groupby(source_col)
    
    for source, group in grouped:
        source_records = []
        
        for idx, row in group.iterrows():
            record_text = f"SOURCE: {source}\n"
            record_data = {'source': source}
            
            for col in group.columns:
                value = row[col]
                if pd.notna(value) and str(value).strip() and col != source_col:
                    record_text += f"{col}: {value}\n"
                    record_data[col] = convert_to_serializable(value)
            
            source_records.append({
                'text': record_text.strip(),
                'data': record_data,
                'index': int(idx)
            })
        
        for i in range(0, len(source_records), max_records_per_source):
            end_idx = min(i + max_records_per_source, len(source_records))
            chunk_group = source_records[i:end_idx]
            
            chunk_text = f"SOURCE CONTACTS: {source} (Records {i+1}-{end_idx}):\n\n"
            chunk_data = []
            
            for record in chunk_group:
                chunk_text += record['text'] + "\n\n"
                chunk_data.append(record['data'])
            
            chunks.append(chunk_text.strip())
            metadata.append({
                'chunk_type': 'source_group',
                'source': source,
                'record_count': len(chunk_group),
                'start_index': i,
                'end_index': end_idx - 1,
                'max_records_per_source': max_records_per_source,
                'complete_records': chunk_data
            })
    
    logger.info(f"Media campaign source chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} chunks")
    return chunks, metadata

def campaign_chunk_semantic(df: pd.DataFrame, field_mapping: Dict[str, str] = None, n_clusters: int = 10) -> Tuple[List[str], List[Dict]]:
    """Apply semantic clustering to media campaign data for intelligent grouping"""
    start_time = time.time()
    from sentence_transformers import SentenceTransformer
    
    contact_texts = []
    for idx, row in df.iterrows():
        text_parts = []
        for col, field_type in field_mapping.items():
            if col in df.columns and pd.notna(row[col]):
                if field_type in ['company', 'company_name']:
                    text_parts.append(f"Company: {row[col]}")
                elif field_type == 'lead_source':
                    text_parts.append(f"Source: {row[col]}")
                elif field_type == 'lead_status':
                    text_parts.append(f"Status: {row[col]}")
                elif field_type == 'email':
                    text_parts.append(f"Email: {row[col]}")
                elif field_type == 'phone':
                    text_parts.append(f"Phone: {row[col]}")
                elif field_type == 'name':
                    text_parts.append(f"Name: {row[col]}")
        contact_texts.append(" | ".join(text_parts))
    
    n_clusters = min(n_clusters, max(2, len(df) // 3))
    
    if len(contact_texts) < n_clusters:
        logger.warning(f"Not enough records for {n_clusters} clusters, using record-based chunking")
        return campaign_chunk_records(df, 5, field_mapping)
    
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(contact_texts)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    chunks = []
    metadata = []
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) > 0:
            chunk_text = f"SEMANTIC CLUSTER {cluster_id} (Similar Contacts):\n\n"
            cluster_records = []
            
            for idx in cluster_indices:
                record = df.iloc[idx]
                chunk_text += f"Contact {idx}:\n"
                record_data = {}
                
                for col, field_type in field_mapping.items():
                    if col in df.columns and pd.notna(record[col]):
                        chunk_text += f"  {col}: {record[col]}\n"
                        record_data[col] = convert_to_serializable(record[col])
                
                chunk_text += "\n"
                cluster_records.append(record_data)
            
            chunks.append(chunk_text.strip())
            metadata.append({
                'chunk_type': 'semantic_cluster',
                'cluster_id': int(cluster_id),
                'record_count': len(cluster_indices),
                'cluster_size': len(cluster_indices),
                'complete_records': cluster_records
            })
    
    logger.info(f"Media campaign semantic clustering completed in {time.time() - start_time:.2f}s, created {len(chunks)} clusters")
    return chunks, metadata

def campaign_chunk_document(df: pd.DataFrame, key_column: str, field_mapping: Dict[str, str] = None, token_limit: int = 2000) -> Tuple[List[str], List[Dict]]:
    """Document-based chunking using a key column for media campaign data"""
    start_time = time.time()
    
    chunks = []
    metadata = []
    
    if key_column not in df.columns:
        possible_keys = [col for col, ftype in field_mapping.items() if ftype in ['company', 'company_name', 'lead_source', 'campaign_source']]
        if possible_keys:
            key_column = possible_keys[0]
            logger.warning(f"Key column not found, using: {key_column}")
        else:
            key_column = df.columns[0] if len(df.columns) > 0 else "id"
            logger.warning(f"No suitable key column found, using first column: {key_column}")
    
    grouped = df.groupby(key_column)
    
    for key, group in grouped:
        chunk_text = f"DOCUMENT GROUP: {key}\n\n"
        group_records = []
        
        for idx, row in group.iterrows():
            chunk_text += f"Record {idx}:\n"
            record_data = {key_column: key}
            
            for col in group.columns:
                value = row[col]
                if pd.notna(value) and str(value).strip() and col != key_column:
                    chunk_text += f"  {col}: {value}\n"
                    record_data[col] = convert_to_serializable(value)
            
            chunk_text += "\n"
            group_records.append(record_data)
        
        chunks.append(chunk_text.strip())
        metadata.append({
            'chunk_type': 'document_group',
            'grouping_column': key_column,
            'group_value': key,
            'record_count': len(group),
            'complete_records': group_records
        })
    
    logger.info(f"Media campaign document chunking completed in {time.time() - start_time:.2f}s, created {len(chunks)} document groups using column '{key_column}'")
    return chunks, metadata

# -----------------------------
# 🔹 SMART COMPANY RETRIEVAL SYSTEM
# -----------------------------
def campaign_smart_retrieval(query: str, search_field: str = "auto", k: int = 5, include_complete_records: bool = True):
    """
    SMART TWO-STAGE RETRIEVAL:
    Stage 1: Exact/Partial company matching
    Stage 2: Semantic fallback
    """
    logger.info(f"🎯 SMART Company Retrieval - Query: '{query}', Field: {search_field}")
    
    start_time = time.time()
    
    if not campaign_state['media_campaign_data']:
        logger.warning("No media campaign data found, using regular retrieval")
        return campaign_retrieve(query, search_field, k, include_complete_records)
    
    field_mapping = campaign_state['media_campaign_data'].get('field_mapping', {})
    processed_df = campaign_state['media_campaign_data'].get('processed_df')
    
    if processed_df is None:
        return {"error": "No processed data available"}
    
    # STAGE 1: COMPANY FIELD MATCHING
    company_matches = find_company_matches(processed_df, field_mapping, query)
    
    if company_matches:
        logger.info(f"✅ Found {len(company_matches)} company matches for '{query}'")
        results = format_company_matches(company_matches, query, k, include_complete_records)
        result_dict = {
            "query": query,
            "search_field": "company_auto",
            "results": results,
            "total_matches": len(company_matches),
            "exact_matches": len([m for m in company_matches if m['match_type'] == 'exact']),
            "partial_matches": len([m for m in company_matches if m['match_type'] == 'partial']),
            "retrieval_method": 'company_keyword_matching',
            "processing_time": f"{time.time() - start_time:.2f}s"
        }
        return serialize_data(result_dict)
    
    # STAGE 2: SEMANTIC FALLBACK
    logger.info(f"🔍 No company matches found, using semantic search for '{query}'")
    semantic_results = campaign_retrieve(query, search_field, k, include_complete_records)
    
    if 'error' not in semantic_results:
        semantic_results['retrieval_method'] = 'semantic_fallback'
        semantic_results['company_matches_found'] = 0
        semantic_results['processing_time'] = f"{time.time() - start_time:.2f}s"
    
    return serialize_data(semantic_results)

def find_company_matches(df: pd.DataFrame, field_mapping: Dict[str, str], query: str) -> List[Dict]:
    """Find exact and partial company matches"""
    company_cols = [col for col, ftype in field_mapping.items() if ftype in ['company', 'company_name']]
    query_lower = query.lower().strip()
    
    exact_matches = []
    partial_matches = []
    
    for company_col in company_cols:
        if company_col not in df.columns:
            continue
            
        # Exact matches (case-insensitive)
        exact_mask = df[company_col].astype(str).str.lower() == query_lower
        exact_indices = df[exact_mask].index.tolist()
        
        for idx in exact_indices:
            record_dict = df.loc[idx].to_dict()
            record = {k: convert_to_serializable(v) for k, v in record_dict.items()}
            
            exact_matches.append({
                'record': record,
                'company_column': company_col,
                'company_value': record[company_col],
                'match_type': 'exact',
                'similarity': 1.0,
                'record_index': int(idx)
            })
        
        # Partial matches (contains)
        if not exact_mask.any():
            partial_mask = df[company_col].astype(str).str.lower().str.contains(query_lower, na=False)
            partial_indices = df[partial_mask].index.tolist()
            
            for idx in partial_indices:
                record_dict = df.loc[idx].to_dict()
                record = {k: convert_to_serializable(v) for k, v in record_dict.items()}
                company_value = record[company_col]
                similarity = calculate_text_similarity(query_lower, company_value.lower())
                
                partial_matches.append({
                    'record': record,
                    'company_column': company_col,
                    'company_value': company_value,
                    'match_type': 'partial',
                    'similarity': float(similarity),
                    'record_index': int(idx)
                })
    
    all_matches = exact_matches + partial_matches
    all_matches.sort(key=lambda x: (x['match_type'] == 'exact', x['similarity']), reverse=True)
    
    return all_matches

def calculate_text_similarity(query: str, text: str) -> float:
    """Calculate similarity between query and text"""
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    
    if not query_words or not text_words:
        return 0.0
    
    intersection = query_words.intersection(text_words)
    union = query_words.union(text_words)
    
    return len(intersection) / len(union) if union else 0.0

def format_company_matches(company_matches: List[Dict], query: str, k: int, include_complete_records: bool) -> List[Dict]:
    """Format company matches into results format"""
    results = []
    seen_records = set()
    
    for i, match in enumerate(company_matches[:k]):
        record = match['record']
        record_key = str(record.get('lead_id', '') + str(record.get('customer_id', '')) + str(match['record_index']))
        
        if record_key in seen_records:
            continue
        seen_records.add(record_key)
        
        content = f"COMPANY MATCH: {match['company_value']} ({match['match_type'].upper()})\n"
        content += f"MATCHED FIELD: {match['company_column']}\n"
        content += f"SIMILARITY: {match['similarity']:.3f}\n\n"
        
        for field, value in record.items():
            if pd.notna(value) and str(value).strip():
                content += f"{field}: {value}\n"
        
        result_data = {
            'rank': len(results) + 1,
            'content': content.strip(),
            'similarity': float(match['similarity']),
            'match_type': match['match_type'],
            'company_matched': match['company_value'],
            'matched_field': match['company_column']
        }
        
        if include_complete_records:
            result_data['complete_record'] = [record]
        
        results.append(result_data)
    
    return results

# -----------------------------
# 🔹 Embedding Functions
# -----------------------------
def embed_texts(chunks, model_name="paraphrase-MiniLM-L6-v2", openai_api_key=None, openai_base_url=None, batch_size=EMBEDDING_BATCH_SIZE, use_parallel=True):
    start_time = time.time()
    
    if use_parallel and len(chunks) > 500 and not openai_api_key and "text-embedding" not in model_name.lower():
        logger.info(f"Using parallel processing for {len(chunks)} chunks")
        model, embeddings = parallel_embed_texts(chunks, model_name, batch_size)
        logger.info(f"Parallel embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
        return model, embeddings
    
    if openai_api_key or "text-embedding" in model_name.lower():
        openai_model = OpenAIEmbeddingAPI(
            model_name=model_name if "text-embedding" in model_name.lower() else "text-embedding-ada-002",
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        embeddings = openai_model.encode(chunks, batch_size=batch_size)
        model = openai_model
        logger.info(f"OpenAI embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        embeddings = model.encode(chunks, batch_size=batch_size, show_progress_bar=True)
        logger.info(f"Local embedding completed in {time.time() - start_time:.2f}s, embedded {len(chunks)} chunks")
    
    return model, np.array(embeddings).astype("float32")

def parallel_embed_texts(chunks, model_name="paraphrase-MiniLM-L6-v2", batch_size=EMBEDDING_BATCH_SIZE, num_workers=PARALLEL_WORKERS):
    """Parallel embedding for faster processing"""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    
    def embed_batch(batch_chunks):
        return model.encode(batch_chunks, batch_size=batch_size)
    
    chunk_batches = [chunks[i:i + len(chunks)//num_workers] for i in range(0, len(chunks), len(chunks)//num_workers)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(embed_batch, chunk_batches))
    
    embeddings = np.vstack(results)
    return model, embeddings

class OpenAIEmbeddingAPI:
    """OpenAI-compatible embedding API"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None, base_url: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.is_local = not api_key
    
    def encode(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
        """Encode texts using OpenAI API or local fallback"""
        if self.is_local:
            from sentence_transformers import SentenceTransformer
            local_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            embeddings = local_model.encode(texts, batch_size=batch_size)
            return np.array(embeddings).astype("float32")
        else:
            import openai
            openai.api_key = self.api_key
            if self.base_url:
                openai.base_url = self.base_url
            
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                try:
                    response = openai.embeddings.create(
                        model=self.model_name,
                        input=batch_texts
                    )
                    batch_embeddings = [data.embedding for data in response.data]
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"OpenAI API error: {e}")
                    from sentence_transformers import SentenceTransformer
                    local_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                    fallback_embeddings = local_model.encode(batch_texts)
                    embeddings.extend(fallback_embeddings)
            
            return np.array(embeddings).astype("float32")

# -----------------------------
# 🔹 Storage Functions
# -----------------------------
def store_chroma(chunks, embeddings, collection_name="chunks_collection"):
    start_time = time.time()
    client = chromadb.PersistentClient(path="chromadb_store_campaign")
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name)
    
    batch_size = 1000
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        batch_chunks = chunks[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_ids = [str(j) for j in range(i, end_idx)]
        
        col.add(
            ids=batch_ids,
            documents=batch_chunks,
            embeddings=batch_embeddings.tolist()
        )
    
    logger.info(f"Chroma storage completed in {time.time() - start_time:.2f}s, stored {len(chunks)} vectors")
    return {"type": "chroma", "collection": col, "collection_name": collection_name}

def store_faiss(embeddings):
    start_time = time.time()
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    
    faiss.normalize_L2(embeddings)
    
    batch_size = 10000
    for i in range(0, embeddings.shape[0], batch_size):
        end_idx = min(i + batch_size, embeddings.shape[0])
        batch_embeddings = embeddings[i:end_idx]
        index.add(batch_embeddings)
    
    logger.info(f"FAISS storage completed in {time.time() - start_time:.2f}s, stored {embeddings.shape[0]} vectors")
    return {"type": "faiss", "index": index}

# -----------------------------
# 🔹 Retrieval Functions
# -----------------------------
def campaign_retrieve(query: str, search_field: str = "all", k: int = 5, include_complete_records: bool = True):
    """Specialized retrieval for media campaign data with consistent result structure"""
    start_time = time.time()
    
    logger.info(f"🎯 Media campaign retrieval - Query: '{query}', Field: {search_field}, K: {k}")
    
    if campaign_state['model'] is None:
        return {"error": "No model available. Run a pipeline first."}
    
    if campaign_state['store_info'] is None:
        return {"error": "No store available. Run a pipeline first."}
    
    if campaign_state['chunks'] is None or len(campaign_state['chunks']) == 0:
        return {"error": "No chunks available. Run a pipeline first."}
    
    try:
        if hasattr(campaign_state['model'], 'encode'):
            query_embedding = campaign_state['model'].encode([query])
        else:
            query_embedding = campaign_state['model'].encode([query])
        
        query_arr = np.array(query_embedding).astype("float32")
        
    except Exception as e:
        return {"error": f"Query encoding failed: {str(e)}"}
    
    results = []
    
    try:
        if campaign_state['store_info']["type"] == "faiss":
            index = campaign_state['store_info']["index"]
            
            faiss.normalize_L2(query_arr)
            similarities, indices = index.search(query_arr, k)
            
            for i in range(len(indices[0])):
                idx = indices[0][i]
                similarity_val = similarities[0][i]
                
                if idx < len(campaign_state['chunks']):
                    result_data = {
                        "rank": i + 1,
                        "content": campaign_state['chunks'][idx],
                        "similarity": float(similarity_val),
                        "distance": float(1 - similarity_val),
                        "metric": "cosine"
                    }
                    
                    if include_complete_records and campaign_state['media_campaign_data']:
                        complete_records = get_complete_records_for_chunk(idx)
                        if complete_records:
                            result_data["complete_record"] = complete_records
                    
                    results.append(result_data)
        
        elif campaign_state['store_info']["type"] == "chroma":
            collection = campaign_state['store_info']["collection"]
            chroma_results = collection.query(
                query_embeddings=query_arr.tolist(),
                n_results=k,
                include=["documents", "distances"]
            )
            
            for i, (doc, distance) in enumerate(zip(
                chroma_results["documents"][0], 
                chroma_results["distances"][0]
            )):
                similarity = 1 - distance
                
                result_data = {
                    "rank": i + 1,
                    "content": doc,
                    "similarity": float(similarity),
                    "distance": float(distance),
                    "metric": "cosine"
                }
                
                if include_complete_records and campaign_state['media_campaign_data']:
                    for chunk_idx, chunk in enumerate(campaign_state['chunks']):
                        if chunk == doc:
                            complete_records = get_complete_records_for_chunk(chunk_idx)
                            if complete_records:
                                result_data["complete_record"] = complete_records
                            break
                
                results.append(result_data)
        
        logger.info(f"✅ Media campaign retrieval completed in {time.time() - start_time:.2f}s, found {len(results)} results")
        
        result_dict = {
            "query": query, 
            "search_field": search_field,
            "k": k, 
            "include_complete_records": include_complete_records,
            "results": results,
            "retrieval_method": "semantic_search",
            "processing_time": f"{time.time() - start_time:.2f}s"
        }
        
        return serialize_data(result_dict)
        
    except Exception as e:
        logger.error(f"❌ Media campaign retrieval failed: {str(e)}")
        return {"error": f"Media campaign retrieval failed: {str(e)}"}

def get_complete_records_for_chunk(chunk_idx: int) -> List[Dict]:
    """Helper function to get complete records for a chunk index"""
    if not campaign_state['media_campaign_data']:
        return []
    
    metadata = campaign_state['media_campaign_data'].get('metadata', [])
    if chunk_idx < len(metadata):
        chunk_metadata = metadata[chunk_idx]
        return chunk_metadata.get('complete_records', [])
    
    return []

# -----------------------------
# 🔹 Pipeline Functions - MAIN
# -----------------------------
def run_campaign_pipeline(df, chunk_method="record_based", chunk_size=5, model_choice="paraphrase-MiniLM-L6-v2", 
                          storage_choice="faiss", db_config=None, file_info=None, use_openai=False, 
                          openai_api_key=None, openai_base_url=None, use_turbo=False, batch_size=EMBEDDING_BATCH_SIZE,
                          preserve_record_structure=True, document_key_column=None,
                          agentic_strategy=None, user_context=None):
    """Enhanced Media Campaign pipeline with all chunking methods (including agentic)"""
    logger.info(f"🎯 Starting Campaign pipeline with {len(df)} records")
    campaign_state['df'] = df.copy()
    campaign_set_file_info(file_info)
    
    field_mapping = detect_media_campaign_fields(df)
    df1 = campaign_preprocess(df, field_mapping)
    
    chunks = []
    metadata = []
    
    # Enhanced chunking methods
    if chunk_method == "agentic":
        # Agentic chunking for campaign data
        import os
        import sys
        sys.path.append(os.path.dirname(__file__))
        from backend_agentic import get_agentic_orchestrator
        
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("Agentic chunking requires GEMINI_API_KEY environment variable")
        
        logger.info(f"Using agentic chunking for Campaign with strategy: {agentic_strategy or 'auto'}")
        orchestrator = get_agentic_orchestrator(gemini_key)
        chunks, metadata = orchestrator.analyze_and_chunk(
            df=df1,
            strategy=agentic_strategy or "auto",
            user_context=user_context or "Marketing campaign data analysis",
            max_chunk_size=chunk_size * 10  # Campaign chunks are typically smaller
        )
    elif chunk_method == "record_based":
        chunks, metadata = campaign_chunk_records(df1, chunk_size, field_mapping)
    elif chunk_method == "company_based":
        chunks, metadata = campaign_chunk_by_company(df1, field_mapping, chunk_size)
    elif chunk_method == "source_based":
        chunks, metadata = campaign_chunk_by_source(df1, field_mapping, chunk_size)
    elif chunk_method == "semantic_clustering":
        chunks, metadata = campaign_chunk_semantic(df1, field_mapping, chunk_size)
    elif chunk_method == "document_based":
        if document_key_column:
            chunks, metadata = campaign_chunk_document(df1, document_key_column, field_mapping)
        else:
            possible_keys = [col for col, ftype in field_mapping.items() if ftype in ['company', 'company_name']]
            doc_key = possible_keys[0] if possible_keys else df1.columns[0]
            chunks, metadata = campaign_chunk_document(df1, doc_key, field_mapping)
            logger.info(f"Auto-selected document key: {doc_key}")
    else:
        chunks, metadata = campaign_chunk_records(df1, 5, field_mapping)
    
    campaign_state['media_campaign_data'] = {
        'field_mapping': field_mapping,
        'processed_df': df1,
        'metadata': metadata,
        'complete_records': [record for meta in metadata for record in meta.get('complete_records', [])]
    }
    
    actual_openai = use_openai or "text-embedding" in model_choice.lower()
    model, embs = embed_texts(chunks, model_choice, openai_api_key if actual_openai else None, openai_base_url, batch_size=batch_size, use_parallel=use_turbo)
    
    if storage_choice == "faiss":
        store = store_faiss(embs)
    else:
        store = store_chroma(chunks, embs, f"campaign_{int(time.time())}")
    
    campaign_state['model'] = model
    campaign_state['store_info'] = store
    campaign_state['chunks'] = chunks
    campaign_state['embeddings'] = embs
    
    # Also update global state for consistency with other modes
    from backend import current_model, current_store_info, current_chunks, current_embeddings, current_df, save_state
    import backend
    
    backend.current_model = model
    backend.current_store_info = store
    backend.current_chunks = chunks
    backend.current_embeddings = embs
    backend.current_df = df1.copy()
    
    # Save state to disk for persistence
    save_state()
    
    logger.info(f"✅ Campaign pipeline completed: {len(chunks)} chunks created")
    
    result = {
        "rows": len(df1), 
        "chunks": len(chunks), 
        "stored": store["type"],
        "embedding_model": model_choice,
        "retrieval_ready": True,
        "turbo_mode": use_turbo,
        "media_campaign_processed": True,
        "field_mapping": field_mapping,
        "chunk_method": chunk_method,
        "document_key_used": document_key_column if chunk_method == "document_based" else None
    }
    
    return serialize_data(result)

# -----------------------------
# 🔹 Large File Processing
# -----------------------------
def process_large_file_in_batches(file_path: str, processing_callback, batch_size: int = BATCH_SIZE):
    """Process large CSV file in batches to avoid memory issues"""
    chunks_processed = []
    
    for chunk in pd.read_csv(file_path, chunksize=batch_size):
        processed_chunk = processing_callback(chunk)
        chunks_processed.extend(processed_chunk)
        gc.collect()
    
    return chunks_processed

def campaign_process_large_file(file_path: str, **kwargs):
    """Process large media campaign file in batches"""
    def process_batch(batch_df):
        field_mapping = detect_media_campaign_fields(batch_df)
        processed_df = campaign_preprocess(batch_df, field_mapping)
        
        chunk_method = kwargs.get('chunk_method', 'record_based')
        chunk_size = kwargs.get('chunk_size', 5)
        document_key_column = kwargs.get('document_key_column')
        
        if chunk_method == "record_based":
            chunks, _ = campaign_chunk_records(processed_df, chunk_size, field_mapping)
        elif chunk_method == "company_based":
            chunks, _ = campaign_chunk_by_company(processed_df, field_mapping, chunk_size)
        elif chunk_method == "source_based":
            chunks, _ = campaign_chunk_by_source(processed_df, field_mapping, chunk_size)
        elif chunk_method == "semantic_clustering":
            chunks, _ = campaign_chunk_semantic(processed_df, field_mapping, chunk_size)
        elif chunk_method == "document_based":
            if document_key_column:
                chunks, _ = campaign_chunk_document(processed_df, document_key_column, field_mapping)
            else:
                possible_keys = [col for col, ftype in field_mapping.items() if ftype in ['company', 'company_name']]
                doc_key = possible_keys[0] if possible_keys else processed_df.columns[0]
                chunks, _ = campaign_chunk_document(processed_df, doc_key, field_mapping)
        else:
            chunks, _ = campaign_chunk_records(processed_df, 5, field_mapping)
        
        return chunks
    
    all_chunks = process_large_file_in_batches(file_path, process_batch)
    
    model_choice = kwargs.get('model_choice', 'paraphrase-MiniLM-L6-v2')
    storage_choice = kwargs.get('storage_choice', 'faiss')
    use_openai = kwargs.get('use_openai', False)
    openai_api_key = kwargs.get('openai_api_key')
    openai_base_url = kwargs.get('openai_base_url')
    batch_size = kwargs.get('batch_size', EMBEDDING_BATCH_SIZE)
    use_turbo = kwargs.get('use_turbo', True)
    
    model, embs = embed_texts(all_chunks, model_choice, openai_api_key, openai_base_url, batch_size=batch_size, use_parallel=use_turbo)
    
    if storage_choice == "faiss":
        store = store_faiss(embs)
    else:
        store = store_chroma(all_chunks, embs, f"campaign_large_{int(time.time())}")
    
    campaign_state['model'] = model
    campaign_state['store_info'] = store
    campaign_state['chunks'] = all_chunks
    campaign_state['embeddings'] = embs
    
    # Also update global state for consistency with other modes
    from backend import current_model, current_store_info, current_chunks, current_embeddings, current_df, save_state
    import backend
    
    backend.current_model = model
    backend.current_store_info = store
    backend.current_chunks = all_chunks
    backend.current_embeddings = embs
    backend.current_df = None  # Large file processing doesn't keep the full dataframe
    
    # Save state to disk for persistence
    save_state()
    
    logger.info(f"✅ Large file Campaign pipeline completed: {len(all_chunks)} chunks created")
    
    result = {
        "rows": "Large file processed in batches", 
        "chunks": len(all_chunks), 
        "stored": store["type"],
        "embedding_model": model_choice,
        "retrieval_ready": True,
        "large_file_processed": True,
        "turbo_mode": use_turbo,
        "media_campaign_processed": True
    }
    
    return serialize_data(result)

def campaign_process_file_direct(file_path: str, **kwargs):
    """Process file directly from filesystem path (no memory loading)"""
    file_size = os.path.getsize(file_path)
    
    logger.info(f"Processing campaign file directly from filesystem: {file_path} ({file_size/(1024**3):.2f}GB)")
    
    if not can_load_file(file_size):
        logger.warning(f"File size {file_size/(1024**3):.2f}GB may exceed safe memory limits")
    
    return campaign_process_large_file(file_path, **kwargs)

# -----------------------------
# 🔹 Export Functions
# -----------------------------
def campaign_export_chunks():
    """Export current chunks as CSV"""
    if campaign_state['chunks']:
        chunks_df = pd.DataFrame({
            'chunk_id': range(len(campaign_state['chunks'])),
            'chunk_content': campaign_state['chunks']
        })
        return chunks_df.to_csv(index=False)
    return ""

def campaign_export_embeddings():
    """Export current embeddings as JSON"""
    if campaign_state['embeddings'] is not None:
        embeddings_list = campaign_state['embeddings'].tolist()
        return {
            "embeddings": embeddings_list,
            "shape": list(campaign_state['embeddings'].shape),
            "total_chunks": len(embeddings_list)
        }
    return {}

def campaign_export_preprocessed():
    """Export preprocessed data as text"""
    if campaign_state['preprocessed_df'] is not None:
        text_output = "CAMPAIGN PREPROCESSED DATA SUMMARY\n"
        text_output += "=" * 50 + "\n\n"
        
        text_output += f"Shape: {campaign_state['preprocessed_df'].shape[0]} rows, {campaign_state['preprocessed_df'].shape[1]} columns\n\n"
        
        text_output += "COLUMNS:\n"
        text_output += "-" * 20 + "\n"
        for col in campaign_state['preprocessed_df'].columns:
            dtype = campaign_state['preprocessed_df'][col].dtype
            text_output += f"{col} ({dtype})\n"
        
        text_output += "\nDATA PREVIEW (First 10 rows):\n"
        text_output += "-" * 30 + "\n"
        
        preview_df = campaign_state['preprocessed_df'].head(10)
        for idx, row in preview_df.iterrows():
            text_output += f"\nRow {idx}:\n"
            for col in preview_df.columns:
                text_output += f"  {col}: {row[col]}\n"
        
        text_output += "\nNULL VALUES SUMMARY:\n"
        text_output += "-" * 25 + "\n"
        null_counts = campaign_state['preprocessed_df'].isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                text_output += f"{col}: {count} null values\n"
        
        if null_counts.sum() == 0:
            text_output += "No null values found\n"
        
        return text_output
    return ""

# -----------------------------
# 🔹 System Info
# -----------------------------
def campaign_get_system_info():
    """Get system information including memory usage"""
    memory = psutil.virtual_memory()
    return {
        "memory_usage": f"{memory.percent}%",
        "available_memory": f"{memory.available / (1024**3):.2f} GB",
        "total_memory": f"{memory.total / (1024**3):.2f} GB",
        "large_file_support": True,
        "max_recommended_file_size": "3GB+",
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "parallel_workers": PARALLEL_WORKERS,
        "campaign_mode": True,
        "smart_company_retrieval": True
    }

