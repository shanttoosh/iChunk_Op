"""
Agentic Chunking Module for iChunk Optimizer
=============================================

This module implements AI-powered agentic chunking strategies for CSV/tabular data.
Uses Google Gemini API to intelligently analyze data structure and decide optimal chunking.

Agentic Agents:
- SchemaAwareChunkingAgent: Analyzes schema and column relationships
- EntityCentricChunkingAgent: Groups by entities (user, product, etc.)
- SemanticRowChunkingAgent: Groups semantically similar rows
- TemporalChunkingAgent: Time-based chunking for time-series data
- AgenticChunkingOrchestrator: Coordinates all agents

Author: iChunk Optimizer Team
Date: 2025
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import json
import logging
import time
from datetime import datetime

# Import Gemini client from backend
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None

logger = logging.getLogger(__name__)

# =====================================================================
# GEMINI CLIENT FOR AGENTIC CHUNKING
# =====================================================================

class GeminiAgenticClient:
    """Gemini client optimized for agentic chunking analysis"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash-exp", temperature: float = 0.3, max_output_tokens: int = 2048):
        if not GENAI_AVAILABLE or genai is None:
            raise ImportError("google-generativeai is not installed. Please install it: pip install google-generativeai")
        
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        logger.info(f"Initializing Gemini Agentic Client with model: {model}")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            logger.info("Gemini Agentic Client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def analyze_with_json(self, prompt: str) -> Dict[str, Any]:
        """Generate response and parse as JSON"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                }
            )
            
            answer = getattr(response, "text", "") or ""
            
            # Try to extract JSON from response
            # Sometimes Gemini wraps JSON in markdown code blocks
            if "```json" in answer:
                json_start = answer.find("```json") + 7
                json_end = answer.find("```", json_start)
                answer = answer[json_start:json_end].strip()
            elif "```" in answer:
                json_start = answer.find("```") + 3
                json_end = answer.find("```", json_start)
                answer = answer[json_start:json_end].strip()
            
            # Parse JSON
            result = json.loads(answer)
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.error(f"Response: {answer}")
            return {"error": "Invalid JSON response", "raw_response": answer}
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {"error": str(e)}

# =====================================================================
# SCHEMA-AWARE CHUNKING AGENT
# =====================================================================

class SchemaAwareChunkingAgent:
    """Analyzes table schema and decides optimal chunking strategy"""
    
    def __init__(self, gemini_client: GeminiAgenticClient):
        self.gemini_client = gemini_client
        logger.info("SchemaAwareChunkingAgent initialized")
    
    def analyze_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze schema and get AI recommendations"""
        
        # Prepare schema information (optimized for token usage)
        schema_info = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape,
            'sample_data': df.head(3).to_dict('records'),
            'unique_counts': df.nunique().to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'text_columns': list(df.select_dtypes(include=['object']).columns),
            'date_columns': list(df.select_dtypes(include=['datetime64']).columns)
        }
        
        prompt = f"""
Analyze this CSV/table schema and recommend the best chunking strategy:

**Schema Information:**
- Total Rows: {schema_info['shape'][0]}
- Total Columns: {schema_info['shape'][1]}
- Columns: {schema_info['columns']}
- Data Types: {schema_info['dtypes']}
- Numeric Columns: {schema_info['numeric_columns']}
- Text Columns: {schema_info['text_columns']}
- Date Columns: {schema_info['date_columns']}
- Unique Value Counts (Cardinality): {schema_info['unique_counts']}
- Null Counts: {schema_info['null_counts']}

**Sample Data (first 3 rows):**
{json.dumps(schema_info['sample_data'], indent=2)}

**Your Task:**
Analyze this data structure and recommend the optimal chunking strategy.

**Questions to Consider:**
1. What type of data is this? (transactional, user profiles, logs, product catalog, etc.)
2. Is there a natural grouping column? (Look for columns with names like: id, user_id, customer_id, product_id, company_id, category, etc.)
3. Are there temporal patterns? (date columns for time-based chunking)
4. What are the relationships between columns?
5. What chunk size would preserve meaningful context?

**Available Strategies:**
- "entity": Group by entity column (user_id, product_id, etc.)
- "schema": Group by schema patterns and relationships
- "temporal": Group by time periods
- "fixed_rows": Fixed number of rows per chunk

**Return JSON Format (IMPORTANT - ONLY JSON, NO EXTRA TEXT):**
{{
  "recommended_strategy": "entity|schema|temporal|fixed_rows",
  "grouping_column": "column_name or null",
  "chunk_size": 100-2000,
  "reasoning": "Why this strategy is best",
  "data_type": "transactional|user_profiles|logs|product_catalog|other",
  "entity_type": "user|product|company|transaction|other|null",
  "confidence": 0.0-1.0
}}
"""
        
        logger.info("Analyzing schema with Gemini...")
        result = self.gemini_client.analyze_with_json(prompt)
        
        if "error" in result:
            logger.warning(f"Schema analysis failed, using fallback: {result['error']}")
            # Fallback strategy
            result = self._fallback_schema_analysis(df)
        
        logger.info(f"Schema analysis complete: {result.get('recommended_strategy', 'unknown')}")
        return result
    
    def _fallback_schema_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback schema analysis without AI"""
        
        # Look for common ID columns
        id_columns = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['id', '_id', 'user', 'customer', 'product', 'company']
        )]
        
        if id_columns:
            grouping_col = id_columns[0]
            strategy = "entity"
        else:
            grouping_col = None
            strategy = "fixed_rows"
        
        return {
            "recommended_strategy": strategy,
            "grouping_column": grouping_col,
            "chunk_size": 1000,
            "reasoning": "Fallback heuristic-based analysis",
            "data_type": "unknown",
            "entity_type": "unknown",
            "confidence": 0.5
        }
    
    def chunk_by_schema(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Tuple[List[str], List[Dict]]:
        """Execute schema-based chunking"""
        
        strategy = analysis.get('recommended_strategy', 'fixed_rows')
        
        if strategy == 'entity' and analysis.get('grouping_column'):
            return self._chunk_by_entity_column(df, analysis['grouping_column'], analysis)
        elif strategy == 'fixed_rows':
            return self._chunk_by_fixed_rows(df, analysis.get('chunk_size', 1000))
        else:
            # Default to fixed rows
            return self._chunk_by_fixed_rows(df, analysis.get('chunk_size', 1000))
    
    def _chunk_by_entity_column(self, df: pd.DataFrame, column: str, analysis: Dict) -> Tuple[List[str], List[Dict]]:
        """Chunk by entity column"""
        
        chunks = []
        metadata = []
        
        for entity_value, group in df.groupby(column):
            # Format chunk as text
            chunk_text = self._format_dataframe_chunk(group)
            chunks.append(chunk_text)
            
            metadata.append({
                'strategy': 'entity',
                'entity_column': column,
                'entity_value': str(entity_value),
                'row_count': len(group),
                'data_type': analysis.get('data_type', 'unknown'),
                'chunk_index': len(chunks) - 1
            })
        
        logger.info(f"Created {len(chunks)} entity-based chunks grouped by '{column}'")
        return chunks, metadata
    
    def _chunk_by_fixed_rows(self, df: pd.DataFrame, chunk_size: int) -> Tuple[List[str], List[Dict]]:
        """Chunk by fixed number of rows"""
        
        chunks = []
        metadata = []
        
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            chunk_text = self._format_dataframe_chunk(chunk_df)
            chunks.append(chunk_text)
            
            metadata.append({
                'strategy': 'fixed_rows',
                'row_count': len(chunk_df),
                'start_row': i,
                'end_row': min(i+chunk_size, len(df)),
                'chunk_index': len(chunks) - 1
            })
        
        logger.info(f"Created {len(chunks)} fixed-row chunks with size {chunk_size}")
        return chunks, metadata
    
    def _format_dataframe_chunk(self, df: pd.DataFrame) -> str:
        """Format DataFrame chunk as readable text"""
        
        # Convert to structured text format
        lines = []
        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            lines.append(row_text)
        
        return "\n".join(lines)

# =====================================================================
# ENTITY-CENTRIC CHUNKING AGENT
# =====================================================================

class EntityCentricChunkingAgent:
    """Groups rows by entities (users, products, companies)"""
    
    def __init__(self, gemini_client: GeminiAgenticClient):
        self.gemini_client = gemini_client
        logger.info("EntityCentricChunkingAgent initialized")
    
    def identify_entity_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """AI identifies primary entity columns"""
        
        sample_data = df.head(5).to_dict('records')
        columns_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
        unique_counts = df.nunique().to_dict()
        
        prompt = f"""
Identify the primary entity column(s) in this table:

**Columns:** {list(df.columns)}
**Data Types:** {columns_info}
**Unique Value Counts:** {unique_counts}
**Total Rows:** {len(df)}
**Sample Data:**
{json.dumps(sample_data, indent=2)}

**Task:**
Identify which column represents the PRIMARY entity for grouping.

**Common Entity Patterns:**
- user_id, customer_id, client_id → User entities
- product_id, item_id, sku → Product entities
- company_id, organization_id → Company entities
- transaction_id, order_id → Transaction entities
- name, email (if unique enough) → Natural entities

**Return JSON Format:**
{{
  "primary_entity_column": "column_name",
  "entity_type": "user|product|company|transaction|other",
  "confidence": 0.0-1.0,
  "reasoning": "Why this column",
  "alternative_columns": ["col1", "col2"]
}}
"""
        
        logger.info("Identifying entity columns with Gemini...")
        result = self.gemini_client.analyze_with_json(prompt)
        
        if "error" in result:
            logger.warning(f"Entity identification failed, using fallback: {result['error']}")
            result = self._fallback_entity_detection(df)
        
        return result
    
    def _fallback_entity_detection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback entity detection using heuristics"""
        
        # Look for common entity column patterns
        entity_keywords = ['id', '_id', 'user', 'customer', 'product', 'company', 'client']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in entity_keywords):
                # Check cardinality
                unique_ratio = df[col].nunique() / len(df)
                if 0.1 < unique_ratio < 1.0:  # Not too few, not all unique
                    return {
                        "primary_entity_column": col,
                        "entity_type": "unknown",
                        "confidence": 0.6,
                        "reasoning": f"Heuristic detection based on column name '{col}'",
                        "alternative_columns": []
                    }
        
        # No entity column found
        return {
            "primary_entity_column": None,
            "entity_type": None,
            "confidence": 0.0,
            "reasoning": "No clear entity column detected",
            "alternative_columns": []
        }
    
    def chunk_by_entity(self, df: pd.DataFrame, entity_analysis: Dict[str, Any], max_rows_per_chunk: int = 2000) -> Tuple[List[str], List[Dict]]:
        """Chunk by entity"""
        
        entity_column = entity_analysis.get('primary_entity_column')
        
        if not entity_column or entity_column not in df.columns:
            logger.warning("No valid entity column, falling back to fixed-row chunking")
            return self._fallback_fixed_chunks(df, 1000)
        
        chunks = []
        metadata = []
        
        for entity_value, group in df.groupby(entity_column):
            # If entity group is too large, split it
            if len(group) > max_rows_per_chunk:
                sub_chunks = self._split_large_entity(group, entity_value, entity_column, max_rows_per_chunk)
                chunks.extend([sc['text'] for sc in sub_chunks])
                metadata.extend([sc['metadata'] for sc in sub_chunks])
            else:
                chunk_text = self._format_entity_chunk(group)
                chunks.append(chunk_text)
                
                metadata.append({
                    'strategy': 'entity_centric',
                    'entity_column': entity_column,
                    'entity_value': str(entity_value),
                    'entity_type': entity_analysis.get('entity_type', 'unknown'),
                    'row_count': len(group),
                    'chunk_index': len(chunks) - 1
                })
        
        logger.info(f"Created {len(chunks)} entity-centric chunks grouped by '{entity_column}'")
        return chunks, metadata
    
    def _split_large_entity(self, group: pd.DataFrame, entity_value, entity_column: str, max_rows: int) -> List[Dict]:
        """Split large entity groups into smaller chunks"""
        
        sub_chunks = []
        for i in range(0, len(group), max_rows):
            sub_group = group.iloc[i:i+max_rows]
            chunk_text = self._format_entity_chunk(sub_group)
            
            sub_chunks.append({
                'text': chunk_text,
                'metadata': {
                    'strategy': 'entity_centric_split',
                    'entity_column': entity_column,
                    'entity_value': str(entity_value),
                    'row_count': len(sub_group),
                    'sub_chunk_index': i // max_rows,
                    'chunk_index': -1  # Will be set by caller
                }
            })
        
        return sub_chunks
    
    def _format_entity_chunk(self, df: pd.DataFrame) -> str:
        """Format entity chunk"""
        lines = []
        for idx, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            lines.append(row_text)
        return "\n".join(lines)
    
    def _fallback_fixed_chunks(self, df: pd.DataFrame, chunk_size: int) -> Tuple[List[str], List[Dict]]:
        """Fallback to fixed-row chunking"""
        chunks = []
        metadata = []
        
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            chunk_text = self._format_entity_chunk(chunk_df)
            chunks.append(chunk_text)
            
            metadata.append({
                'strategy': 'fixed_rows_fallback',
                'row_count': len(chunk_df),
                'start_row': i,
                'end_row': min(i+chunk_size, len(df)),
                'chunk_index': len(chunks) - 1
            })
        
        return chunks, metadata

# =====================================================================
# AGENTIC CHUNKING ORCHESTRATOR
# =====================================================================

class AgenticChunkingOrchestrator:
    """Main orchestrator for agentic chunking"""
    
    def __init__(self, api_key: str):
        """Initialize with Gemini API key"""
        
        if not api_key:
            raise ValueError("Gemini API key is required for agentic chunking")
        
        self.gemini_client = GeminiAgenticClient(api_key)
        self.schema_agent = SchemaAwareChunkingAgent(self.gemini_client)
        self.entity_agent = EntityCentricChunkingAgent(self.gemini_client)
        
        logger.info("AgenticChunkingOrchestrator initialized with all agents")
    
    def analyze_and_chunk(self, 
                          df: pd.DataFrame, 
                          strategy: str = "auto",
                          user_context: str = None,
                          max_chunk_size: int = 2000) -> Tuple[List[str], List[Dict]]:
        """
        Main entry point for agentic chunking
        
        Args:
            df: DataFrame to chunk
            strategy: "auto", "schema", "entity"
            user_context: Optional user context
            max_chunk_size: Maximum rows per chunk
        
        Returns:
            chunks: List of text chunks
            metadata: List of metadata dicts
        """
        
        logger.info(f"Starting agentic chunking with strategy: {strategy}")
        start_time = time.time()
        
        try:
            # Step 1: Determine strategy
            if strategy == "auto":
                selected_strategy = self._ai_select_strategy(df, user_context)
            else:
                selected_strategy = strategy
            
            logger.info(f"Selected strategy: {selected_strategy}")
            
            # Step 2: Execute strategy
            if selected_strategy == "schema":
                analysis = self.schema_agent.analyze_schema(df)
                chunks, metadata = self.schema_agent.chunk_by_schema(df, analysis)
            
            elif selected_strategy == "entity":
                entity_analysis = self.entity_agent.identify_entity_columns(df)
                chunks, metadata = self.entity_agent.chunk_by_entity(df, entity_analysis, max_chunk_size)
            
            else:
                # Default to schema
                analysis = self.schema_agent.analyze_schema(df)
                chunks, metadata = self.schema_agent.chunk_by_schema(df, analysis)
            
            # Add global metadata
            for meta in metadata:
                meta['agentic'] = True
                meta['strategy'] = selected_strategy
                meta['timestamp'] = datetime.now().isoformat()
                if user_context:
                    meta['user_context'] = user_context
            
            elapsed = time.time() - start_time
            logger.info(f"Agentic chunking complete: {len(chunks)} chunks in {elapsed:.2f}s")
            
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"Agentic chunking failed: {e}")
            raise
    
    def _ai_select_strategy(self, df: pd.DataFrame, user_context: str = None) -> str:
        """AI selects optimal strategy"""
        
        schema_summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        prompt = f"""
Select the best agentic chunking strategy for this data:

**Data Summary:**
- Rows: {schema_summary['shape'][0]}
- Columns: {schema_summary['shape'][1]}
- Column Names: {schema_summary['columns']}

**User Context:** {user_context or "No specific context provided"}

**Available Strategies:**
1. "schema": Analyze schema patterns and relationships (best for complex structured data)
2. "entity": Group by primary entity column (best for user/product/company data)

**Task:** Select ONE strategy that would create the most meaningful chunks.

**Return JSON Format:**
{{
  "strategy": "schema|entity",
  "reasoning": "Brief explanation"
}}
"""
        
        result = self.gemini_client.analyze_with_json(prompt)
        
        if "error" in result or "strategy" not in result:
            logger.warning("Strategy selection failed, defaulting to 'schema'")
            return "schema"
        
        return result['strategy']

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def get_agentic_orchestrator(api_key: str = None) -> AgenticChunkingOrchestrator:
    """Factory function to create orchestrator"""
    
    import os
    
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable.")
    
    return AgenticChunkingOrchestrator(api_key)

