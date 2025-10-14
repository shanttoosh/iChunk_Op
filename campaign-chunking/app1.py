# app.py - MEDIA CAMPAIGN ONLY VERSION - FIXED RETRIEVAL ERROR
import streamlit as st
import pandas as pd
import requests
import io
import time
import base64
import os
from datetime import datetime
import json
import tempfile
import shutil

# FastAPI backend URL
API_BASE_URL = "http://127.0.0.1:8001"

# ---------- Enhanced Black Theme with Animations ----------
def load_css():
    """Load custom CSS styles with enhanced animations and black theme"""
    st.markdown(f"""
        <style>
            /* Root variables with black color scheme */
            :root {{
                --primary-gradient: linear-gradient(135deg, #f26f21 0%, #ffa800 100%);
                --success-gradient: linear-gradient(135deg, #48bb78, #38a169);
                --warning-gradient: linear-gradient(135deg, #ed8936, #dd6b20);
                --error-gradient: linear-gradient(135deg, #f56565, #e53e3e);
                --glass-bg: #1a1a1a99;
                --glass-border: rgba(255, 255, 255, 0.15);
                --text-primary: #ffffff;
                --text-secondary: #e2e8f0;
                --text-muted: #a0aec0;
                --border-light: rgba(255, 255, 255, 0.1);
                --bg-light: #2d2d2d;
                --background-color: #000000;
                --card-background: #1a1a1a;
                --dark-grey-bg: #1a1a1a;
                --darker-grey-bg: #000000;
            }}
            
            /* Main styling with black theme */
            .main {{
                background: var(--darker-grey-bg);
                color: var(--text-primary);
            }}
            
            .stApp {{
                background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #2d2d2d 100%);
                min-height: 100vh;
            }}
            
            /* Enhanced Header animations */
            @keyframes slideIn {{
                0% {{ 
                    transform: translateX(-100%); 
                    opacity: 0; 
                    filter: blur(10px);
                }}
                100% {{ 
                    transform: translateX(0); 
                    opacity: 1;
                    filter: blur(0);
                }}
            }}
            
            @keyframes fadeInUp {{
                0% {{ 
                    opacity: 0; 
                    transform: translateY(30px) scale(0.95);
                    filter: blur(5px);
                }}
                100% {{ 
                    opacity: 1; 
                    transform: translateY(0) scale(1);
                    filter: blur(0);
                }}
            }}
            
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.05); }}
                100% {{ transform: scale(1); }}
            }}
            
            @keyframes shimmer {{
                0% {{ background-position: -200px 0; }}
                100% {{ background-position: 200px 0; }}
            }}
            
            .slide-in {{
                animation: slideIn 0.8s ease-out;
            }}
            
            .fade-in-up {{
                animation: fadeInUp 0.6s ease-out;
            }}
            
            .pulse-animation {{
                animation: pulse 2s ease-in-out infinite;
            }}
            
            .shimmer-effect {{
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
                background-size: 200px 100%;
                animation: shimmer 2s infinite;
            }}
            
            /* Enhanced Card styling */
            .feature-card {{
                background: var(--card-background);
                backdrop-filter: blur(15px);
                border: 1px solid var(--glass-border);
                border-radius: 20px;
                padding: 25px;
                margin: 15px 0;
                transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
                position: relative;
                overflow: hidden;
            }}
            
            .feature-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
                transition: left 0.5s;
            }}
            
            .feature-card:hover::before {{
                left: 100%;
            }}
            
            .feature-card:hover {{
                transform: translateY(-12px) scale(1.03);
                box-shadow: 0 20px 60px rgba(242, 111, 33, 0.4);
                border-color: #f26f21;
            }}
            
            .chart-card {{
                background: var(--card-background);
                border: 1px solid var(--glass-border);
                border-radius: 16px;
                padding: 20px;
                margin: 15px 0;
                transition: all 0.4s ease;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
            }}
            
            .chart-card:hover {{
                transform: translateY(-8px) scale(1.02);
                box-shadow: 0 15px 35px rgba(242, 111, 33, 0.3);
                border-color: #ffa800;
            }}
            
            /* Enhanced Button styling */
            .stButton button {{
                background: var(--primary-gradient);
                color: white;
                border: none;
                border-radius: 15px;
                padding: 14px 28px;
                font-weight: 700;
                font-size: 16px;
                transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
                box-shadow: 0 6px 20px rgba(242, 111, 33, 0.4);
                position: relative;
                overflow: hidden;
            }}
            
            .stButton button::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }}
            
            .stButton button:hover::before {{
                left: 100%;
            }}
            
            .stButton button:hover {{
                transform: translateY(-4px) scale(1.05);
                box-shadow: 0 12px 30px rgba(242, 111, 33, 0.6);
                background: var(--primary-gradient);
            }}
            
            .stButton button:active {{
                transform: translateY(-2px) scale(1.02);
            }}
            
            /* Enhanced Metric card styling */
            .metric-card {{
                background: var(--card-background);
                border-radius: 16px;
                padding: 25px;
                margin: 12px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
                border-left: 6px solid #f26f21;
                transition: all 0.4s ease;
                position: relative;
                overflow: hidden;
            }}
            
            .metric-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 3px;
                background: var(--primary-gradient);
                transform: scaleX(0);
                transition: transform 0.3s ease;
            }}
            
            .metric-card:hover::before {{
                transform: scaleX(1);
            }}
            
            .metric-card:hover {{
                transform: translateY(-6px) scale(1.03);
                box-shadow: 0 15px 35px rgba(242, 111, 33, 0.3);
            }}
            
            /* Campaign-specific styles */
            .campaign-card {{
                background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
                border: 2px solid #f26f21;
                border-radius: 20px;
                padding: 25px;
                margin: 15px 0;
                box-shadow: 0 15px 40px rgba(242, 111, 33, 0.2);
                position: relative;
                overflow: hidden;
            }}
            
            .campaign-card::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: var(--primary-gradient);
            }}
            
            .contact-record {{
                background: var(--card-background);
                border: 1px solid var(--glass-border);
                border-radius: 12px;
                padding: 20px;
                margin: 10px 0;
                transition: all 0.3s ease;
                border-left: 4px solid #48bb78;
            }}
            
            .contact-record:hover {{
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(72, 187, 120, 0.2);
            }}
            
            .field-badge {{
                background: linear-gradient(135deg, #4299e1, #3182ce);
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: 600;
                margin: 2px;
                display: inline-block;
            }}
            
            /* Process step styling */
            .process-step {{
                padding: 12px;
                margin: 8px 0;
                border-radius: 8px;
                border-left: 4px solid #f26f21;
                background: var(--card-background);
            }}
            
            .process-step.completed {{
                border-left-color: #48bb78;
                background: rgba(72, 187, 120, 0.1);
            }}
            
            .process-step.running {{
                border-left-color: #ed8936;
                background: rgba(237, 137, 54, 0.1);
            }}
            
            .process-step.pending {{
                border-left-color: #a0aec0;
                background: rgba(160, 174, 192, 0.1);
            }}
            
            /* Large file warning */
            .large-file-warning {{
                background: linear-gradient(135deg, #ed8936, #dd6b20);
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
                text-align: center;
            }}
            
            /* Chunk header */
            .chunk-header {{
                background: var(--primary-gradient);
                color: white;
                padding: 10px 15px;
                border-radius: 8px;
                margin-bottom: 10px;
                font-weight: bold;
            }}
            
            /* Smart retrieval badge */
            .smart-badge {{
                background: linear-gradient(135deg, #48bb78, #38a169);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: 600;
                margin: 5px;
                display: inline-block;
                box-shadow: 0 4px 15px rgba(72, 187, 120, 0.3);
            }}
            
            /* Card title and content */
            .card-title {{
                font-size: 1.3em;
                font-weight: bold;
                margin-bottom: 15px;
                color: #f26f21;
            }}
            
            .card-content {{
                line-height: 1.6;
                color: var(--text-secondary);
            }}
            
            /* Rest of the styling */
            .stTextInput>div>div>input {{
                background: var(--card-background);
                color: var(--text-primary);
                border: 1px solid var(--glass-border);
            }}
            
            .stSelectbox>div>div {{
                background: var(--card-background);
                color: var(--text-primary);
            }}
            
            .stSlider>div>div>div {{
                background: var(--primary-gradient);
            }}
            
            .stExpander {{
                background: var(--card-background);
                border: 1px solid var(--glass-border);
                border-radius: 10px;
            }}
            
            .stDataFrame {{
                background: var(--card-background);
            }}
        </style>
    """, unsafe_allow_html=True)

# Load CSS when app starts
load_css()

# ---------- API Client Functions ----------
def call_media_campaign_api(file_path: str, filename: str, config: dict, db_config: dict = None,
                           use_openai: bool = False, openai_api_key: str = None, openai_base_url: str = None,
                           process_large_files: bool = True, use_turbo: bool = False, batch_size: int = 256):
    """Send media campaign file directly from filesystem path"""
    try:
        if db_config and db_config.get('use_db'):
            st.info(f"🔍 Sending database config to backend...")
            data = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in config.items()}
            data.update({
                "db_type": db_config["db_type"],
                "host": db_config["host"],
                "port": str(db_config["port"]),
                "username": db_config["username"],
                "password": db_config["password"],
                "database": db_config["database"],
                "table_name": db_config["table_name"],
                "use_openai": str(use_openai).lower(),
                "openai_api_key": openai_api_key or "",
                "openai_base_url": openai_base_url or "",
                "process_large_files": str(process_large_files).lower(),
                "use_turbo": str(use_turbo).lower(),
                "batch_size": str(batch_size)
            })
            debug_data = {k: '***' if 'password' in k else v for k, v in data.items()}
            st.info(f"🔍 API Request Data: {debug_data}")
            data = {k: v for k, v in data.items() if v is not None}
            response = requests.post(f"{API_BASE_URL}/run_media_campaign", data=data)
            return response.json()
        else:
            with open(file_path, 'rb') as f:
                files = {"file": (filename, f, "text/csv")}
                data = {k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in config.items()}
                data.update({
                    "use_openai": str(use_openai).lower(),
                    "openai_api_key": openai_api_key or "",
                    "openai_base_url": openai_base_url or "",
                    "process_large_files": str(process_large_files).lower(),
                    "use_turbo": str(use_turbo).lower(),
                    "batch_size": str(batch_size)
                })
                data = {k: v for k, v in data.items() if v is not None}
                response = requests.post(f"{API_BASE_URL}/run_media_campaign", files=files, data=data)
            return response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

def call_media_campaign_retrieval_api(query: str, search_field: str = "all", k: int = 5, include_complete_records: bool = True):
    """Specialized retrieval for media campaign data"""
    data = {
        "query": query,
        "search_field": search_field,
        "k": k,
        "include_complete_records": str(include_complete_records).lower()
    }
    response = requests.post(f"{API_BASE_URL}/retrieve_media_campaign", data=data)
    return response.json()

def call_smart_company_retrieval_api(query: str, search_field: str = "auto", k: int = 5, include_complete_records: bool = True):
    """SMART TWO-STAGE RETRIEVAL for company searches"""
    data = {
        "query": query,
        "search_field": search_field,
        "k": k,
        "include_complete_records": str(include_complete_records).lower()
    }
    response = requests.post(f"{API_BASE_URL}/smart_company_retrieval", data=data)
    return response.json()

def get_system_info_api():
    response = requests.get(f"{API_BASE_URL}/system_info")
    return response.json()

def get_file_info_api():
    response = requests.get(f"{API_BASE_URL}/file_info")
    return response.json()

def get_capabilities_api():
    response = requests.get(f"{API_BASE_URL}/capabilities")
    return response.json()

def download_chunks_csv():
    """Download chunks in CSV format"""
    response = requests.get(f"{API_BASE_URL}/export/chunks")
    return response.content

def download_embeddings_json():
    """Download embeddings in JSON format"""
    response = requests.get(f"{API_BASE_URL}/export/embeddings_json")
    return response.content

def download_preprocessed_data():
    """Download preprocessed data in text format"""
    response = requests.get(f"{API_BASE_URL}/export/preprocessed")
    return response.content

def db_test_connection_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/test_connection", data=payload).json()

def db_list_tables_api(payload: dict):
    return requests.post(f"{API_BASE_URL}/db/list_tables", data=payload).json()

# ---------- Large File Helper Functions ----------
def is_large_file(file_size: int, threshold_mb: int = 100) -> bool:
    """Check if file is considered large"""
    return file_size > threshold_mb * 1024 * 1024

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def handle_file_upload(uploaded_file):
    """
    Safely handle file uploads by streaming to disk (no memory loading)
    Returns temporary file path and file info
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        shutil.copyfileobj(uploaded_file, tmp_file)
        temp_path = tmp_file.name
    
    file_size = os.path.getsize(temp_path)
    file_size_str = format_file_size(file_size)
    
    file_info = {
        "name": uploaded_file.name,
        "size": file_size_str,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": "Temporary storage",
        "temp_path": temp_path
    }
    
    return temp_path, file_info

# ---------- SMART COLUMN DISPLAY FUNCTIONS ----------
def get_contextual_columns(record, query=""):
    """Return columns based on query context - SMART COLUMN SELECTION"""
    if not isinstance(record, dict):
        return {}
    
    query_lower = query.lower() if query else ""
    
    # Core columns to always show
    core_columns = ['mail_id', 'company_name', 'lead_status', 'name']
    
    # Contextual columns based on query
    if any(word in query_lower for word in ['mail', 'email', '@', 'gmail', 'outlook', 'yahoo']):
        contextual_columns = ['mail_id', 'company_name', 'lead_source', 'timestamp', 'ph_no']
    elif any(word in query_lower for word in ['company', 'organization', 'firm', 'business']):
        contextual_columns = ['company_name', 'industry', 'address', 'employee_size', 'revenue']
    elif any(word in query_lower for word in ['campaign', 'performance', 'ctr', 'conversion']):
        contextual_columns = ['campaign_name', 'impressions', 'clicks', 'conversions', 'ctr', 'roi']
    elif any(word in query_lower for word in ['phone', 'contact', 'number', 'call']):
        contextual_columns = ['ph_no', 'name', 'company_name', 'mail_id', 'role']
    else:
        # Default minimal set
        contextual_columns = ['mail_id', 'company_name', 'lead_status', 'timestamp']
    
    # Combine core and contextual, remove duplicates
    display_columns = list(dict.fromkeys(core_columns + contextual_columns))
    
    # Filter to only include columns that exist in record and have values
    result = {}
    for col in display_columns:
        if col in record and pd.notna(record[col]) and str(record[col]).strip():
            result[col] = record[col]
    
    return result

def get_field_icon(field_name):
    """Get appropriate icon for field type"""
    field_icons = {
        'mail_id': '📧', 'email': '📧', 'email_address': '📧',
        'ph_no': '📞', 'phone': '📞', 'mobile': '📞', 'contact_number': '📞',
        'company_name': '🏢', 'company': '🏢', 'organization': '🏢',
        'name': '👤', 'contact_name': '👤', 'first_name': '👤', 'last_name': '👤',
        'address': '📍', 'location': '📍', 'city': '📍', 'country': '📍',
        'lead_status': '📊', 'status': '📊', 'lead_source': '🎯',
        'campaign_name': '📢', 'campaign_source': '📢',
        'timestamp': '⏰', 'date': '⏰', 'time': '⏰',
        'impressions': '👀', 'clicks': '🖱️', 'conversions': '✅', 'ctr': '📈', 'roi': '💰'
    }
    return field_icons.get(field_name, '📋')

def display_contact_record(records, rank, similarity, query="", match_type=None, company_matched=None):
    """Display contact records with SMART column selection"""
    similarity_color = "#48bb78" if similarity > 0.7 else "#ed8936" if similarity > 0.4 else "#f56565"
    
    expander_title = f"👤 Contact Group #{rank} (Similarity: {similarity:.3f})"
    if match_type:
        expander_title += f" | {match_type.upper()}"
    
    with st.expander(expander_title, expanded=False):
        # Header with match information
        header_html = f"""
        <div style="background: #1a1a1a; padding: 12px; border-radius: 8px; margin-bottom: 12px; border-left: 6px solid {similarity_color};">
            <strong>Rank:</strong> {rank} | 
            <strong>Similarity:</strong> {similarity:.3f} |
            <strong>Records in Group:</strong> {len(records) if isinstance(records, list) else 1}
        """
        
        if match_type:
            header_html += f""" | <strong>Match Type:</strong> {match_type}"""
        
        if company_matched:
            header_html += f""" | <strong>Company:</strong> {company_matched}"""
        
        header_html += "</div>"
        
        st.markdown(header_html, unsafe_allow_html=True)
        
        # Smart retrieval badge
        if match_type in ['exact', 'partial']:
            st.markdown(f'<div class="smart-badge">🎯 SMART RETRIEVAL: {match_type.upper()} COMPANY MATCH</div>', unsafe_allow_html=True)
        
        if isinstance(records, list):
            for record_idx, record in enumerate(records):
                st.markdown(f"**Record {record_idx + 1}:**")
                display_single_contact_record(record, query)
                if record_idx < len(records) - 1:
                    st.markdown("---")
        else:
            display_single_contact_record(records, query)

def display_single_contact_record(record, query=""):
    """Display a single contact record with SMART column selection"""
    st.markdown("""
    <div class="contact-record">
    """, unsafe_allow_html=True)
    
    if isinstance(record, dict):
        # Get SMART selected columns based on query context
        display_data = get_contextual_columns(record, query)
        
        # Count total available fields
        total_fields = len([v for v in record.values() if pd.notna(v) and str(v).strip()])
        displayed_fields = len(display_data)
        
        # Display selected columns
        for field, value in display_data.items():
            icon = get_field_icon(field)
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{icon} {field}:**")
            with col2:
                st.write(str(value))
        
        # Show expandable section for all fields if there are more
        if total_fields > displayed_fields:
            with st.expander(f"📋 Show all {total_fields} fields"):
                for field, value in record.items():
                    if pd.notna(value) and str(value).strip() and field not in display_data:
                        icon = get_field_icon(field)
                        st.write(f"**{icon} {field}:** {value}")
    else:
        st.write("Raw record data:")
        st.write(record)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- FIXED RETRIEVAL DISPLAY FUNCTION ----------
def safe_get_result_data(result_data, key, default=None):
    """Safely get value from result_data whether it's a dict or has nested structure"""
    if isinstance(result_data, dict):
        return result_data.get(key, default)
    elif hasattr(result_data, 'get'):
        return result_data.get(key, default)
    else:
        return default

def display_retrieval_results(results):
    """Fixed function to display retrieval results without AttributeError"""
    if 'error' in results:
        st.error(f"Retrieval error: {results['error']}")
        return
    
    if 'results' not in results or not results['results']:
        st.info("No campaign results found for the query")
        return
    
    # Show retrieval method info
    retrieval_method = safe_get_result_data(results, 'retrieval_method', 'standard')
    if retrieval_method == 'company_keyword_matching':
        exact_matches = safe_get_result_data(results, 'exact_matches', 0)
        partial_matches = safe_get_result_data(results, 'partial_matches', 0)
        total_matches = safe_get_result_data(results, 'total_matches', 0)
        
        st.success(f"🎯 **SMART Company Retrieval Results**")
        st.info(f"**Exact Matches:** {exact_matches} | **Partial Matches:** {partial_matches} | **Total:** {total_matches}")
        st.markdown(f'<div class="smart-badge">🚀 USING SMART TWO-STAGE RETRIEVAL</div>', unsafe_allow_html=True)
    else:
        st.success(f"✅ Found {len(results['results'])} campaign matches for: \"{safe_get_result_data(results, 'query', 'Unknown')}\"")
        st.info(f"**Search Field:** {safe_get_result_data(results, 'search_field', 'all')} | **Complete Records:** {safe_get_result_data(results, 'include_complete_records', True)}")
    
    # Display results with contact records - FIXED VERSION
    for i, result_data in enumerate(results['results']):
        # SAFELY get values using the helper function
        match_type = safe_get_result_data(result_data, 'match_type')
        company_matched = safe_get_result_data(result_data, 'company_matched')
        similarity = safe_get_result_data(result_data, 'similarity', 0.0)
        
        # Check if we have complete records or just chunk content
        complete_records = safe_get_result_data(result_data, 'complete_record')
        
        if complete_records:
            # Display with complete contact records
            display_contact_record(
                complete_records, 
                i+1, 
                similarity,
                safe_get_result_data(results, 'query', ''),  # Pass query for contextual column selection
                match_type,
                company_matched
            )
        else:
            # Fallback to regular chunk display
            content = safe_get_result_data(result_data, 'content', 'No content available')
            with st.expander(f"📄 Rank #{i+1} (Similarity: {similarity:.3f})", expanded=False):
                st.text_area(
                    "Chunk Content",
                    value=content,
                    height=300,
                    key=f"chunk_content_{i}",
                    disabled=True,
                    label_visibility="collapsed"
                )

# ---------- Streamlit App ----------
st.set_page_config(page_title="Media Campaign Processor", layout="wide", page_icon="🎯")

# Enhanced header
st.markdown("""
<div style="background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%); padding: 30px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(242, 111, 33, 0.3); border: 1px solid rgba(242, 111, 33, 0.2);">
    <h1 style="color: white; text-align: center; margin: 0; font-size: 2.8em; font-weight: 700; letter-spacing: 1px;">
        <span style="color: #f26f21;">🎯</span> Media Campaign <span style="color: #f26f21;">P</span>rocessor
    </h1>
    <p style="color: #e2e8f0; text-align: center; margin: 12px 0 0 0; font-size: 1.3em; font-weight: 400;">Advanced Campaign Data Processing + SMART Company Retrieval + AI-Powered Insights</p>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if "api_results" not in st.session_state:
    st.session_state.api_results = None
if "media_campaign_results" not in st.session_state:
    st.session_state.media_campaign_results = None
if "process_status" not in st.session_state:
    st.session_state.process_status = {
        "preprocessing": "pending",
        "chunking": "pending", 
        "embedding": "pending",
        "storage": "pending",
        "retrieval": "pending"
    }
if "process_timings" not in st.session_state:
    st.session_state.process_timings = {}
if "file_info" not in st.session_state:
    st.session_state.file_info = {}
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "use_openai" not in st.session_state:
    st.session_state.use_openai = False
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "openai_base_url" not in st.session_state:
    st.session_state.openai_base_url = ""
if "process_large_files" not in st.session_state:
    st.session_state.process_large_files = True
if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = None
if "use_turbo" not in st.session_state:
    st.session_state.use_turbo = True
if "batch_size" not in st.session_state:
    st.session_state.batch_size = 256
if "use_smart_retrieval" not in st.session_state:
    st.session_state.use_smart_retrieval = True

# Sidebar with process tracking and system info
with st.sidebar:
    st.markdown("""
    <div style="background: var(--primary-gradient); padding: 25px; border-radius: 15px; margin-bottom: 20px;">
        <h2 style="color: white; text-align: center; margin: 0; font-size: 1.5em;">⚡ Process Tracker</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # API connection test
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        st.success("✅ API Connected")
        
        capabilities = get_capabilities_api()
        if capabilities.get('media_campaign_support'):
            st.info("🎯 Media Campaign Mode Available")
        if capabilities.get('smart_company_retrieval'):
            st.success("🚀 SMART Company Retrieval Available")
            
    except:
        st.error("❌ API Not Connected")
    
    st.markdown("---")
    
    # Smart Retrieval Toggle
    with st.expander("🎯 Retrieval Settings", expanded=False):
        st.session_state.use_smart_retrieval = st.checkbox(
            "Enable SMART Company Retrieval", 
            value=st.session_state.use_smart_retrieval,
            help="Uses two-stage retrieval: exact company matching first, then semantic fallback"
        )
        
        if st.session_state.use_smart_retrieval:
            st.success("✅ SMART Retrieval: Company searches will prioritize exact matches")
            st.info("""
            **How it works:**
            - Stage 1: Exact/partial company name matching
            - Stage 2: Semantic similarity fallback
            - Guarantees company-specific results first
            """)
        else:
            st.warning("⚠️ Standard semantic search only")
    
    # OpenAI Configuration
    with st.expander("🤖 OpenAI Configuration", expanded=False):
        st.session_state.use_openai = st.checkbox("Use OpenAI API", value=st.session_state.use_openai)
        
        if st.session_state.use_openai:
            st.session_state.openai_api_key = st.text_input("OpenAI API Key", 
                                                          value=st.session_state.openai_api_key,
                                                          type="password",
                                                          help="Your OpenAI API key")
            st.session_state.openai_base_url = st.text_input("OpenAI Base URL (optional)", 
                                                           value=st.session_state.openai_base_url,
                                                           placeholder="https://api.openai.com/v1",
                                                           help="Custom OpenAI-compatible API endpoint")
            
            if st.session_state.openai_api_key:
                st.success("✅ OpenAI API Configured")
            else:
                st.warning("⚠️ Please enter OpenAI API Key")
    
    # Large File Configuration
    with st.expander("💾 Large File Settings", expanded=False):
        st.session_state.process_large_files = st.checkbox(
            "Enable Large File Processing", 
            value=st.session_state.process_large_files,
            help="Process files larger than 100MB in batches to avoid memory issues"
        )
        
        if st.session_state.process_large_files:
            st.info("""**Large File Features:**
            - Direct disk streaming (no memory overload)
            - Batch processing for memory efficiency
            - Automatic chunking for files >100MB
            - Progress tracking for large datasets
            - Support for 3GB+ files
            """)
    
    # Process steps display
    st.markdown("### ⚙️ Processing Steps")
    
    steps = [
        ("preprocessing", "🧹 Preprocessing"),
        ("chunking", "📦 Chunking"), 
        ("embedding", "🤖 Embedding"),
        ("storage", "💾 Vector DB"),
        ("retrieval", "🔍 Retrieval")
    ]
    
    for step_key, step_name in steps:
        status = st.session_state.process_status.get(step_key, "pending")
        timing = st.session_state.process_timings.get(step_key, "")
        
        if status == "completed":
            icon = "✅"
            color = "completed"
            timing_display = f"({timing})" if timing else ""
        elif status == "running":
            icon = "🟠"
            color = "running"
            timing_display = ""
        else:
            icon = "⚪"
            color = "pending"
            timing_display = ""
        
        st.markdown(f"""
        <div class="process-step {color}">
            {icon} <strong>{step_name}</strong> {timing_display}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Information
    st.markdown("### 💻 System Information")
    try:
        system_info = get_system_info_api()
        st.write(f"**Memory Usage:** {system_info.get('memory_usage', 'N/A')}")
        st.write(f"**Available Memory:** {system_info.get('available_memory', 'N/A')}")
        st.write(f"**Total Memory:** {system_info.get('total_memory', 'N/A')}")
        st.write(f"**Batch Size:** {system_info.get('embedding_batch_size', 'N/A')}")
        if system_info.get('large_file_support'):
            st.write(f"**Max File Size:** {system_info.get('max_recommended_file_size', 'N/A')}")
    except:
        st.write("**Memory Usage:** N/A")
        st.write("**Available Memory:** N/A")
        st.write("**Total Memory:** N/A")
    
    # File Information
    st.markdown("### 📁 File Information")
    if st.session_state.file_info:
        file_info = st.session_state.file_info
        st.write(f"**File Name:** {file_info.get('name', 'N/A')}")
        st.write(f"**File Size:** {file_info.get('size', 'N/A')}")
        st.write(f"**Upload Time:** {file_info.get('upload_time', 'N/A')}")
        if file_info.get('large_file_processed'):
            st.success("✅ Large File Optimized")
        if file_info.get('turbo_mode'):
            st.success("⚡ Turbo Mode Enabled")
        if file_info.get('data_type') == 'media_campaign':
            st.success("🎯 Media Campaign Data")
    else:
        try:
            file_info = get_file_info_api()
            if file_info and 'filename' in file_info:
                st.write(f"**File Name:** {file_info.get('filename', 'N/A')}")
                st.write(f"**File Size:** {file_info.get('file_size', 0) / 1024:.2f} KB")
                st.write(f"**Upload Time:** {file_info.get('upload_time', 'N/A')}")
                st.write(f"**File Location:** Backend storage")
        except:
            st.write("**File Info:** Not available")
    
    st.markdown("---")
    
    if st.session_state.api_results:
        st.markdown("### 📊 Last Results")
        result = st.session_state.api_results
        st.write(f"**Mode:** {result.get('mode', 'N/A')}")
        if 'summary' in result:
            st.write(f"**Chunks:** {result['summary'].get('chunks', 'N/A')}")
            st.write(f"**Storage:** {result['summary'].get('stored', 'N/A')}")
            st.write(f"**Model:** {result['summary'].get('embedding_model', 'N/A')}")
            if result['summary'].get('turbo_mode'):
                st.success("⚡ Turbo Mode Used")
            if result['summary'].get('retrieval_ready'):
                st.success("🔍 Retrieval Ready")
            if result['summary'].get('large_file_processed'):
                st.success("🚀 Large File Optimized")
            if result['summary'].get('media_campaign_processed'):
                st.success("🎯 Media Campaign Optimized")
    
    if st.button("🔄 Reset Session", use_container_width=True):
        if st.session_state.get('temp_file_path') and os.path.exists(st.session_state.temp_file_path):
            os.unlink(st.session_state.temp_file_path)
        
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main Media Campaign Interface
st.markdown("## 🎯 Media Campaign Data Processor")

# File upload section
st.markdown("### 📁 Upload Campaign Data Source")

input_source = st.radio("Select Input Source:", ["📁 Upload CSV File", "🗄️ Database Import"], key="media_campaign_input_source")

if input_source == "📁 Upload CSV File":
    uploaded_file = st.file_uploader("Choose a CSV file with campaign data", type=["csv"], key="media_campaign_file_upload")
    
    if uploaded_file is not None:
        with st.spinner("🔄 Streaming file to disk..."):
            temp_path, file_info = handle_file_upload(uploaded_file)
            st.session_state.temp_file_path = temp_path
            st.session_state.file_info = file_info
        
        file_size_str = file_info["size"]
        file_size_bytes = os.path.getsize(temp_path)
        
        if is_large_file(file_size_bytes):
            st.markdown(f"""
            <div class="large-file-warning">
                <strong>🚀 Large File Detected: {file_size_str}</strong><br>
                Large file processing is {'ENABLED' if st.session_state.process_large_files else 'DISABLED'}<br>
                <em>File streamed to disk - no memory overload</em>
            </div>
            """, unsafe_allow_html=True)
        
        st.success(f"✅ **{uploaded_file.name}** loaded! ({file_size_str})")
        
        # Show data preview
        try:
            df = pd.read_csv(temp_path, nrows=10)
            st.session_state.preview_df = df
            st.session_state.current_df = df
            
            st.markdown("#### 👁️ Data Preview")
            st.dataframe(df.head(5), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading preview: {str(e)}")
    
    use_db_config = None
    
else:  # Database Import
    st.markdown("#### 🗄️ Database Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        db_type = st.selectbox("Database Type", ["mysql", "postgresql"], key="media_campaign_db_type")
        host = st.text_input("Host", "localhost", key="media_campaign_host")
        port = st.number_input("Port", 1, 65535, 3306 if db_type == "mysql" else 5432, key="media_campaign_port")
    
    with col2:
        username = st.text_input("Username", key="media_campaign_username")
        password = st.text_input("Password", type="password", key="media_campaign_password")
        database = st.text_input("Database", key="media_campaign_database")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔌 Test Connection", key="media_campaign_test_conn", use_container_width=True):
            with st.spinner("Testing connection..."):
                res = db_test_connection_api({
                    "db_type": db_type,
                    "host": host,
                    "port": str(port),
                    "username": username,
                    "password": password,
                    "database": database,
                })
                if res.get("status") == "success":
                    st.success("✅ Connection successful!")
                else:
                    st.error(f"❌ Connection failed: {res.get('message', 'Unknown error')}")
    
    with col2:
        if st.button("📋 List Tables", key="media_campaign_list_tables", use_container_width=True):
            with st.spinner("Fetching tables..."):
                res = db_list_tables_api({
                    "db_type": db_type,
                    "host": host,
                    "port": str(port),
                    "username": username,
                    "password": password,
                    "database": database,
                })
                tables = res.get("tables", [])
                st.session_state["media_campaign_db_tables"] = tables
                if tables:
                    st.success(f"✅ Found {len(tables)} tables in database '{database}'")
                    st.info(f"📋 Tables: {', '.join(tables)}")
                else:
                    st.warning("⚠️ No tables found in this database")
    
    tables = st.session_state.get("media_campaign_db_tables", [])
    if tables:
        table_name = st.selectbox("Select Table", tables, key="media_campaign_table_select")
        use_db_config = {
            "use_db": True,
            "db_type": db_type,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "database": database,
            "table_name": table_name
        }
        
        st.info(f"🔧 **Selected Configuration:** Database: `{database}`, Table: `{table_name}`")
    else:
        use_db_config = None
        st.info("👆 Test connection and list tables first")

# Media Campaign Configuration
st.markdown("### ⚙️ Campaign Processing Configuration")

col1, col2 = st.columns(2)

with col1:
    # Enhanced chunking methods
    chunk_method = st.selectbox("Chunking Method", 
                              ["record_based", "company_based", "source_based", "semantic_clustering", "document_based"],
                              help="How to group campaign data for processing")
    
    if chunk_method == "record_based":
        chunk_size = st.slider("Records per Chunk", 1, 20, 5,
                             help="Number of complete contact records per chunk")
    elif chunk_method == "company_based":
        st.info("Groups contacts by company name")
        chunk_size = st.slider("Max records per company", 5, 50, 10,
                             help="Maximum contacts per company group")
    elif chunk_method == "source_based":
        st.info("Groups contacts by lead source")
        chunk_size = st.slider("Max records per source", 5, 30, 8,
                             help="Maximum contacts per source group")
    elif chunk_method == "semantic_clustering":
        st.info("AI-powered grouping of similar contacts")
        chunk_size = st.slider("Number of clusters", 3, 20, 10,
                             help="Number of semantic clusters to create")
    else:  # document_based
        st.info("Groups contacts by document key")
        chunk_size = 1  # Not used for document-based
    
    model_choice = st.selectbox("Embedding Model", 
                              ["paraphrase-MiniLM-L6-v2", "all-MiniLM-L6-v2", "text-embedding-ada-002"],
                              help="AI model to generate embeddings")

with col2:
    storage_choice = st.selectbox("Vector Storage", 
                                ["faiss", "chroma"],
                                help="Vector database for storing embeddings")
    
    preserve_structure = st.checkbox("Preserve Record Structure", value=True,
                                   help="Keep complete contact records together")
    
    st.session_state.use_turbo = st.checkbox("Enable Turbo Mode", 
                                           value=st.session_state.use_turbo,
                                           help="Parallel processing for faster execution")
    
    st.session_state.batch_size = st.slider("Batch Size", 
                                          64, 512, st.session_state.batch_size, 32,
                                          help="Embedding batch size")

# Document key column selection
document_key_column = None
if chunk_method == "document_based":
    if st.session_state.get('temp_file_path') and os.path.exists(st.session_state.temp_file_path):
        try:
            df_columns = pd.read_csv(st.session_state.temp_file_path, nrows=0).columns.tolist()
            document_key_column = st.selectbox("Select Column for Document Grouping", 
                                             df_columns,
                                             help="Choose which column to use for grouping records into documents")
        except Exception as e:
            st.error(f"Error reading columns: {str(e)}")
    else:
        st.info("Upload a file to see available columns for document grouping")

# Display Media Campaign pipeline info
chunking_descriptions = {
    "record_based": f"Record-based chunking ({chunk_size} records per chunk)",
    "company_based": f"Company-based grouping (max {chunk_size} records per company)",
    "source_based": f"Source-based grouping (max {chunk_size} records per source)",
    "semantic_clustering": f"AI semantic clustering ({chunk_size} clusters)",
    "document_based": f"Document-based grouping using column: {document_key_column if document_key_column else 'auto-selected'}"
}

st.markdown(f"""
<div class="campaign-card">
    <div class="card-title">🎯 Media Campaign Pipeline</div>
    <div class="card-content">
        • <strong>Specialized Preprocessing:</strong><br>
        &nbsp;&nbsp;✓ Field-aware data cleaning<br>
        &nbsp;&nbsp;✓ Contact record preservation<br>
        &nbsp;&nbsp;✓ Campaign data validation<br>
        &nbsp;&nbsp;✓ Structured data handling<br>
        • {chunking_descriptions[chunk_method]}<br>
        • {model_choice} embedding model<br>
        • {storage_choice.upper()} vector storage<br>
        • {'✅ Record structure preserved' if preserve_structure else 'Record structure optimized'}<br>
        • {'⚡ Parallel processing' if st.session_state.use_turbo else 'Sequential processing'}<br>
        • Batch embedding with size {st.session_state.batch_size}<br>
        • 3GB+ file support with disk streaming<br>
        • 🎯 Campaign-specific retrieval<br>
        • {'🚀 SMART Company Retrieval' if st.session_state.use_smart_retrieval else 'Standard semantic search'}<br>
    </div>
</div>
""", unsafe_allow_html=True)

run_enabled = (
    (input_source == "📁 Upload CSV File" and st.session_state.get('temp_file_path') is not None) or
    (input_source == "🗄️ Database Import" and use_db_config is not None)
)

if st.button("🚀 Run Media Campaign Pipeline", type="primary", use_container_width=True, disabled=not run_enabled):
    with st.spinner("Running Media Campaign pipeline..."):
        try:
            st.session_state.process_status["preprocessing"] = "running"
            
            config = {
                "chunk_method": chunk_method,
                "chunk_size": chunk_size,
                "model_choice": model_choice,
                "storage_choice": storage_choice,
                "preserve_record_structure": preserve_structure
            }
            
            # Add document key column if document chunking is selected
            if chunk_method == "document_based" and document_key_column:
                config["document_key_column"] = document_key_column
            
            if input_source == "🗄️ Database Import" and use_db_config:
                st.info(f"🔄 Importing data from database: {use_db_config['database']}.{use_db_config['table_name']}")
            
            if input_source == "📁 Upload CSV File":
                result = call_media_campaign_api(
                    st.session_state.temp_file_path,
                    st.session_state.file_info["name"],
                    config,
                    use_db_config,
                    st.session_state.use_openai,
                    st.session_state.openai_api_key,
                    st.session_state.openai_base_url,
                    st.session_state.process_large_files,
                    st.session_state.use_turbo,
                    st.session_state.batch_size
                )
            else:
                result = call_media_campaign_api(
                    None, None, config, use_db_config,
                    st.session_state.use_openai,
                    st.session_state.openai_api_key,
                    st.session_state.openai_base_url,
                    st.session_state.process_large_files,
                    st.session_state.use_turbo,
                    st.session_state.batch_size
                )
            
            if 'error' in result:
                st.error(f"❌ Pipeline Error: {result['error']}")
            else:
                for step in ["preprocessing", "chunking", "embedding", "storage"]:
                    st.session_state.process_status[step] = "completed"
                    st.session_state.process_timings[step] = "Completed"
                
                st.session_state.api_results = result
                
                if 'summary' in result:
                    if result['summary'].get('large_file_processed'):
                        st.success("✅ Large file processed efficiently with disk streaming!")
                    elif result['summary'].get('turbo_mode'):
                        st.success("⚡ Turbo mode completed successfully!")
                    else:
                        st.success("✅ Media Campaign pipeline completed successfully!")
                    
                    if result['summary'].get('media_campaign_processed'):
                        st.success("🎯 Campaign data optimized for retrieval!")
                
                st.session_state.process_status["retrieval"] = "completed"
            
        except Exception as e:
            st.error(f"❌ API Error: {str(e)}")
        finally:
            if st.session_state.get('temp_file_path') and os.path.exists(st.session_state.temp_file_path):
                os.unlink(st.session_state.temp_file_path)
                st.session_state.temp_file_path = None

# Results and Retrieval Section
if st.session_state.api_results and 'error' not in st.session_state.api_results:
    st.markdown("---")
    st.markdown("## 📊 Processing Results")
    
    result = st.session_state.api_results
    
    if 'summary' in result:
        summary = result['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📈 Records Processed</h3>
                <h2>{summary.get('rows', 'N/A')}</h2>
                <p>Total campaign records</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📦 Chunks Created</h3>
                <h2>{summary.get('chunks', 'N/A')}</h2>
                <p>Embedding chunks</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🤖 Embedding Model</h3>
                <h2>{summary.get('embedding_model', 'N/A').split('/')[-1]}</h2>
                <p>Vector generation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            storage_type = summary.get('stored', 'N/A').upper()
            st.markdown(f"""
            <div class="metric-card">
                <h3>💾 Vector Storage</h3>
                <h2>{storage_type}</h2>
                <p>Similarity search</p>
            </div>
            """, unsafe_allow_html=True)
        
        if summary.get('turbo_mode'):
            st.success("⚡ **Turbo Mode:** Parallel processing enabled for faster execution")
        
        if summary.get('large_file_processed'):
            st.success("🚀 **Large File Processing:** File processed efficiently with disk streaming")
        
        if summary.get('retrieval_ready'):
            st.success("🔍 **Retrieval System:** Ready for similarity search")
            
        if summary.get('media_campaign_processed'):
            st.success("🎯 **Media Campaign:** Data optimized for contact retrieval")
            if st.session_state.use_smart_retrieval:
                st.success("🚀 **SMART Company Retrieval:** Enabled - Company searches will prioritize exact matches")
    
    # Export options
    st.markdown("#### 💾 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download Chunks (CSV)", use_container_width=True):
            try:
                chunks_content = download_chunks_csv()
                st.download_button(
                    label="⬇️ Save Chunks as CSV",
                    data=chunks_content,
                    file_name="campaign_chunks.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error downloading chunks: {str(e)}")
    
    with col2:
        if st.button("📝 Embeddings (JSON)", use_container_width=True):
            try:
                embeddings_json = download_embeddings_json()
                st.download_button(
                    label="⬇️ Save Embeddings as JSON",
                    data=embeddings_json,
                    file_name="campaign_embeddings.json",
                    mime="application/json",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error downloading embeddings: {str(e)}")
    
    with col3:
        if st.button("🧹 Preprocessed Data", use_container_width=True):
            try:
                preprocessed_content = download_preprocessed_data()
                if preprocessed_content:
                    st.download_button(
                        label="⬇️ Save Preprocessed Data",
                        data=preprocessed_content,
                        file_name="preprocessed_campaign_data.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                else:
                    st.warning("No preprocessed data available")
            except Exception as e:
                st.error(f"Error downloading preprocessed data: {str(e)}")
    
    # Enhanced Retrieval Section with SMART COLUMN DISPLAY
    st.markdown("---")
    st.markdown("## 🎯 Campaign Data Retrieval")
    
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        query = st.text_area("Search Campaign Data:", 
                           placeholder="Search for companies, lead sources, status, or any campaign data...",
                           height=100)
    
    with col2:
        search_field = st.selectbox("Search Field:", 
                                  ["all", "company", "lead_source", "lead_status", "campaign_source"],
                                  help="Search in specific fields or all fields")
        
        k = st.number_input("Number of results:", 1, 50, 5)
    
    with col3:
        include_complete = st.checkbox("Complete Records", value=True,
                                     help="Return full contact records")
        
        st.markdown("")
        if st.button("🔍 Search Campaign", use_container_width=True):
            if query:
                with st.spinner("Searching campaign data..."):
                    try:
                        # Use SMART retrieval if enabled, otherwise use standard
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
                            
                    except Exception as e:
                        st.error(f"Retrieval error: {str(e)}")
            else:
                st.warning("Please enter a query first")
        
        if st.button("🧹 Clear Results", use_container_width=True):
            st.session_state.media_campaign_results = None
    
    # Display media campaign retrieval results - USING FIXED FUNCTION
    if st.session_state.media_campaign_results:
        results = st.session_state.media_campaign_results
        display_retrieval_results(results)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-muted); padding: 20px;">
    <p>🚀 <strong>Media Campaign Processor</strong> - Advanced Campaign Data Processing + SMART Company Retrieval + AI-Powered Insights</p>
    <p>Built with FastAPI, Streamlit, and advanced NLP technologies</p>
</div>
""", unsafe_allow_html=True)