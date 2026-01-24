"""
LangGraph Pipeline: Data Discovery Agent Workflow

Pipeline Order:
1. schema_extract_agent -> Extract schema metadata from database
2. data_profile_agent -> Profile tables with statistical fingerprints (includes samples)
3. entity_discovery_agent -> Discover business entities from schema
4. lookup_discovery_agent -> Discover lookup/dimension tables
5. candidate_score_agent -> Score column candidates using name similarity
6. semantic_mapping_agent -> Map target columns to source columns
7. lookup_mapping_agent -> Map lookup values between source and target
"""

from typing import Dict, List, Any, Tuple
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import csv
from datetime import datetime
import pandas as pd

from schema_extract_agent import extract_schema_metadata
from data_profile_agent import profile_database
from entity_discovery_agent import entity_discovery_agent
from lookup_discovery_agent import lookup_discovery_agent
from name_similarity import hybrid_name_similarity
from candidate_score_agent import candidate_score_agent
from sematic_mapping_agent import semantic_mapping_agent
from lookup_mapping_agent import lookup_mapping_agent

# LangChain imports for LLM-based filtering
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
import os


# ==========================================
# Utility Functions
# ==========================================

def load_business_glossary(glossary_path: str = "business_glossary.json") -> Dict[str, Any]:
    """Load business glossary from JSON file"""
    try:
        if os.path.exists(glossary_path):
            with open(glossary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        print(f"  ⚠ Business glossary not found at {glossary_path}")
        return {"entities": {}, "columns": {}}
    except Exception as e:
        print(f"  ⚠ Failed to load business glossary: {e}")
        return {"entities": {}, "columns": {}}

def extract_samples_from_profile(profile: Dict[str, Dict]) -> Dict[str, Dict[str, List[Any]]]:
    """Extract samples from profile data structure"""
    samples_map = {}
    for table, table_profile in profile.items():
        samples_map[table] = {}
        for col, col_profile in table_profile.items():
            samples_map[table][col] = col_profile.get("samples", [])
    return samples_map


# ==========================================
# State Management
# ==========================================

class PipelineState(TypedDict):
    """Pipeline state definition"""
    db_url: str
    target_db_url: str
    business_glossary: Dict[str, Any]
    source_schema: Dict[str, Any]
    target_schema: Dict[str, Any]
    source_profile: Dict[str, Any]
    target_profile: Dict[str, Any]
    source_entities: Dict[str, List[str]]
    target_entities: Dict[str, List[str]]
    source_lookups: Dict[str, Any]
    target_lookups: Dict[str, Any]
    lookup_mappings: Dict[str, List[Dict[str, Any]]]
    column_candidates: Dict[str, List[Dict[str, Any]]]
    candidate_scores: Dict[str, List[Dict[str, float]]]
    mappings: Dict[str, Dict[str, Any]]
    errors: List[str]


# ==========================================
# Pipeline Nodes
# ==========================================

def node_extract_schema(state: PipelineState) -> PipelineState:
    """Extract schema from both source and target databases"""
    print("\n[1/5] EXTRACTING SCHEMA METADATA...")
    
    try:
        source_schema = extract_schema_metadata(state["db_url"])
        target_schema = extract_schema_metadata(state["target_db_url"])
        
        state["source_schema"] = source_schema
        state["target_schema"] = target_schema
        
        print(f"[OK] Source schema extracted: {len(source_schema)} tables")
        print(f"[OK] Target schema extracted: {len(target_schema)} tables")
        
    except Exception as e:
        error_msg = f"Schema extraction failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        state["errors"].append(error_msg)
    
    return state


def node_profile_data(state: PipelineState) -> PipelineState:
    """Profile source and target databases"""
    print("\n[2/5] PROFILING DATA...")
    
    try:
        print("  Profiling source database...")
        source_profile = profile_database(state["db_url"])
        state["source_profile"] = source_profile
        print(f"  [OK] Source database profiled")
        
        print("  Profiling target database...")
        target_profile = profile_database(state["target_db_url"])
        state["target_profile"] = target_profile
        print(f"  [OK] Target database profiled")
        
    except Exception as e:
        error_msg = f"Data profiling failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        state["errors"].append(error_msg)
    
    return state


def node_discover_entities(state: PipelineState) -> PipelineState:
    """Discover business entities from schema"""
    print("\n[3/5] DISCOVERING BUSINESS ENTITIES...")
    
    try:
        business_glossary = state.get("business_glossary", {})
        
        print("  Discovering source entities...")
        source_entities = entity_discovery_agent(state["source_schema"], business_glossary)
        state["source_entities"] = source_entities
        print(f"  [OK] Source entities: {list(source_entities.keys()) if source_entities else 'None'}")
        
        print("  Discovering target entities...")
        target_entities = entity_discovery_agent(state["target_schema"], business_glossary)
        state["target_entities"] = target_entities
        print(f"  [OK] Target entities: {list(target_entities.keys()) if target_entities else 'None'}")
        
    except Exception as e:
        error_msg = f"Entity discovery failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        state["errors"].append(error_msg)
    
    return state


def node_discover_lookups(state: PipelineState) -> PipelineState:
    """Discover lookup/dimension tables"""
    print("\n[4/7] DISCOVERING LOOKUP TABLES...")
    
    try:
        print("  Discovering source lookup tables...")
        print("    Evaluating source tables:")
        source_lookups = lookup_discovery_agent(state["db_url"], state["source_schema"])
        state["source_lookups"] = source_lookups
        print(f"  [OK] Found {len(source_lookups)} source lookup tables")
        if source_lookups:
            for table in source_lookups.keys():
                print(f"       ✓ {table}")
        else:
            print("       (No lookup tables found)")
        
        print("  Discovering target lookup tables...")
        print("    Evaluating target tables:")
        target_lookups = lookup_discovery_agent(state["target_db_url"], state["target_schema"])
        state["target_lookups"] = target_lookups
        print(f"  [OK] Found {len(target_lookups)} target lookup tables")
        if target_lookups:
            for table in target_lookups.keys():
                print(f"       ✓ {table}")
        else:
            print("       (No lookup tables found)")
        
    except Exception as e:
        error_msg = f"Lookup discovery failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        state["errors"].append(error_msg)
    
    return state


def _score_single_target_column(target_key: Tuple[str, str], state: PipelineState, col_info: Tuple[int, int]) -> Tuple[str, List[Dict]]:
    """Score a single target column against all source columns using multi-factor scoring (runs in thread)"""
    target_table, target_col = target_key
    col_idx, total_cols = col_info
    
    scores = []
    source_count = len(state['source_schema'])
    
    print(f"    [Col {col_idx}/{total_cols}] Scoring '{target_table}.{target_col}' against {source_count} source tables...")
    
    # Get target column profile
    target_profile = state["target_profile"].get(target_table, {}).get(target_col, {})
    
    # STEP 1: Use LLM to select top 10 relevant source columns
    print(f"      [LLM FILTER] Selecting top 10 relevant columns for '{target_col}'...")
    filtered_columns = _llm_filter_relevant_columns(target_table, target_col, state)
    print(f"      [LLM FILTER] Selected {len(filtered_columns)} relevant columns")
    
    # STEP 2: Score only the filtered columns
    for source_table_idx, (source_table, source_cols) in enumerate(state["source_schema"].items(), 1):
        source_col_names = [c["column"] for c in source_cols["columns"]]
        
        for source_col in source_col_names:
            # Only score if LLM selected this column
            col_key = f"{source_table}.{source_col}"
            if col_key not in filtered_columns:
                continue
            
            # Get source column profile
            source_profile = state["source_profile"].get(source_table, {}).get(source_col, {})
            
            # Calculate name similarity with business glossary
            business_glossary = state.get("business_glossary", {})
            sim_result = hybrid_name_similarity(source_col, target_col, business_glossary)
            name_sim = sim_result["score"]
            
            # Calculate multi-factor candidate score
            candidate_score = candidate_score_agent(
                src_profile=source_profile,
                tgt_profile=target_profile,
                name_similarity=name_sim
            )
            
            # Get samples from profile
            samples = state["source_profile"].get(source_table, {}).get(source_col, {}).get("samples", [])
            
            scores.append({
                "source_table": source_table,
                "source_column": source_col,
                "score": candidate_score["final_score"],
                "method": sim_result["method"],
                "decision": candidate_score["decision"],
                "breakdown": candidate_score["breakdown"],
                "samples": samples[:5]
            })
        
        print(f"      [Source {source_table_idx}/{source_count}] {source_table}: scored filtered columns")
    
    # Sort by score (descending)
    scores.sort(key=lambda x: x["score"], reverse=True)
    
    key = f"{target_table}.{target_col}"
    if scores:
        print(f"      -> Top match: {scores[0]['source_table']}.{scores[0]['source_column']} (score: {scores[0]['score']:.4f}, decision: {scores[0]['decision']})")
    else:
        print(f"      -> No matches found")
    
    return key, scores[:5]


def _llm_filter_relevant_columns(target_table: str, target_col: str, state: PipelineState, top_k: int = 10) -> List[str]:
    """
    Use LLM to select top K most relevant source columns for a target column.
    Returns list of column keys like 'table.column'
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Build source column list
        source_columns = []
        for source_table, source_cols in state["source_schema"].items():
            for col in source_cols["columns"]:
                col_name = col["column"]
                col_type = col.get("type", "unknown")
                source_columns.append(f"{source_table}.{col_name} ({col_type})")
        
        # Create LLM prompt
        prompt = ChatPromptTemplate.from_template("""
You are a data mapping expert. Given a target column, select the TOP {top_k} most relevant source columns.

TARGET COLUMN:
  Table: {target_table}
  Column: {target_col}

AVAILABLE SOURCE COLUMNS:
{source_columns}

Return a JSON array with exactly the column identifiers (e.g., "table.column") of the top {top_k} most relevant columns.
Consider semantic similarity, data types, and naming conventions.

Return ONLY valid JSON like this:
{{"selected_columns": ["table1.column1", "table2.column2", ...]}}
""")
        
        # Format source columns as string
        source_cols_str = "\n".join([f"  - {col}" for col in source_columns])
        
        # Call LLM
        chain = prompt | llm
        response = chain.invoke({
            "target_table": target_table,
            "target_col": target_col,
            "source_columns": source_cols_str,
            "top_k": top_k
        })
        
        # Parse response
        response_text = response.content.strip()
        
        # Extract JSON from response (handle markdown code fences)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        selected = result.get("selected_columns", [])
        
        return selected[:top_k]  # Ensure we get at most top_k
        
    except Exception as e:
        print(f"      [LLM FILTER ERROR] {str(e)} - falling back to all columns")
        # Fallback: return all columns if LLM filtering fails
        all_columns = []
        for source_table, source_cols in state["source_schema"].items():
            for col in source_cols["columns"]:
                all_columns.append(f"{source_table}.{col['column']}")
        return all_columns


def node_score_candidates(state: PipelineState) -> PipelineState:
    """Score column candidates using name similarity (parallel processing)"""
    print("\n[5/7] SCORING COLUMN CANDIDATES (PARALLEL)...")
    
    try:
        candidate_scores = {}
        
        # Collect all target columns to process
        all_target_columns = []
        total_tables = len(state["target_schema"])
        total_target_cols = sum(len(t["columns"]) for t in state["target_schema"].values())
        col_counter = 0
        
        for table_idx, (target_table, target_cols) in enumerate(state["target_schema"].items(), 1):
            target_col_names = [c["column"] for c in target_cols["columns"]]
            print(f"  Processing target table [{table_idx}/{total_tables}]: {target_table} ({len(target_col_names)} columns)")
            
            for target_col in target_col_names:
                col_counter += 1
                all_target_columns.append(((target_table, target_col), (col_counter, total_target_cols)))
        
        print(f"  Starting parallel scoring of {total_target_cols} columns with ThreadPoolExecutor...")
        
        # Use ThreadPoolExecutor to score columns in parallel
        max_workers = 4  # Adjust based on your system - more threads for I/O bound tasks
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scoring tasks
            futures = {
                executor.submit(_score_single_target_column, target_key, state, col_info): target_key 
                for target_key, col_info in all_target_columns
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                try:
                    key, scores = future.result()
                    candidate_scores[key] = scores
                    completed += 1
                    print(f"  [Progress] {completed}/{total_target_cols} columns scored")
                except Exception as e:
                    target_key = futures[future]
                    print(f"  [ERROR] Failed to score {target_key}: {str(e)}")
        
        state["candidate_scores"] = candidate_scores
        print(f"\n[OK] Scored {len(candidate_scores)} target columns (parallel)")
        
    except Exception as e:
        error_msg = f"Candidate scoring failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        state["errors"].append(error_msg)
    
    return state


def node_semantic_mapping(state: PipelineState) -> PipelineState:
    """Map target columns to source columns"""
    print("\n[6/7] PERFORMING SEMANTIC MAPPING...")
    
    try:
        mappings = {}
        
        # For each target column with candidates
        for target_col_key, candidates in state["candidate_scores"].items():
            if not candidates:
                continue
            
            # Prepare candidate info for semantic mapping agent (from scored candidates)
            candidate_info = [
                {
                    "source": f"{c['source_table']}.{c['source_column']}",
                    "samples": c.get("samples", [])
                }
                for c in candidates[:3]  # Top 3 candidates
            ]
            
            # Run semantic mapping agent with business glossary
            business_glossary = state.get("business_glossary", {})
            mapping = semantic_mapping_agent(
                target_column=target_col_key,
                candidate_sources=[c["source"] for c in candidate_info],
                sample_values=[c["samples"] for c in candidate_info],
                business_glossary=business_glossary
            )
            
            mappings[target_col_key] = mapping
        
        state["mappings"] = mappings
        print(f"[OK] Created mappings for {len(mappings)} columns")
        
    except Exception as e:
        error_msg = f"Semantic mapping failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        state["errors"].append(error_msg)
    
    return state


def node_map_lookups(state: PipelineState) -> PipelineState:
    """Map lookup values between source and target systems"""
    print("\n[7/7] MAPPING LOOKUP VALUES...")
    
    try:
        lookup_mappings = {}
        
        # For each source lookup table, find corresponding target lookup and map values
        for source_lookup_name, source_lookup_info in state["source_lookups"].items():
            print(f"  Processing lookup table: {source_lookup_name}")
            
            # Try to find matching target lookup table
            target_lookup_name = None
            target_lookup_info = None
            
            # Simple heuristic: look for similar table names
            for target_name, target_info in state["target_lookups"].items():
                if source_lookup_name.lower() in target_name.lower() or target_name.lower() in source_lookup_name.lower():
                    target_lookup_name = target_name
                    target_lookup_info = target_info
                    break
            
            if not target_lookup_name:
                print(f"    [SKIP] No matching target lookup table found")
                continue
            
            print(f"    Found target match: {target_lookup_name}")
            
            # Fetch lookup data from databases
            try:
                from sqlalchemy import create_engine, text
                
                source_engine = create_engine(state["db_url"])
                target_engine = create_engine(state["target_db_url"])
                
                # Get data from lookup tables
                source_lookup_df = pd.read_sql(f"SELECT * FROM {source_lookup_name}", source_engine)
                target_lookup_df = pd.read_sql(f"SELECT * FROM {target_lookup_name}", target_engine)
                
                # Map lookup values
                mapping_result = lookup_mapping_agent(source_lookup_df, target_lookup_df)
                
                lookup_mappings[source_lookup_name] = {
                    "target_table": target_lookup_name,
                    "mappings": mapping_result.get("mappings", [])
                }
                
                print(f"    [OK] Mapped {len(mapping_result.get('mappings', []))} lookup values")
                
            except Exception as e:
                print(f"    [ERROR] Failed to map lookup values: {str(e)}")
                state["errors"].append(f"Lookup mapping failed for {source_lookup_name}: {str(e)}")
        
        state["lookup_mappings"] = lookup_mappings
        print(f"[OK] Completed lookup mapping for {len(lookup_mappings)} lookup tables")
        
    except Exception as e:
        error_msg = f"Lookup mapping failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        state["errors"].append(error_msg)
    
    return state


# ==========================================
# Export Results
# ==========================================

def export_mappings_to_csv(result: Dict[str, Any], filename: str = None) -> str:
    """Export mapping results to CSV with multi-factor scores"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mapping_results_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'target_table', 'target_column', 'target_full_name',
            'source_table', 'source_column', 'source_full_name',
            'final_score', 'similarity_method', 'decision',
            'name_similarity', 'regex_score', 'entropy_score', 'cardinality_score', 'null_ratio_score',
            'mapping_type', 'mapping_confidence',
            'transformation_sql', 'rank'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Iterate through all target columns
        for target_col_key, mapping in result['mappings'].items():
            target_table, target_column = target_col_key.split('.', 1)
            
            # Get candidate scores for this target column
            candidates = result['candidate_scores'].get(target_col_key, [])
            
            # Get the mapped source columns
            mapped_sources = mapping.get('source_columns', [])
            
            # Write rows for top candidates with their scores
            for rank, candidate in enumerate(candidates[:5], 1):  # Top 5 candidates
                source_table = candidate['source_table']
                source_column = candidate['source_column']
                source_full = f"{source_table}.{source_column}"
                
                # Check if this source is in the final mapping
                is_mapped = source_full in mapped_sources
                
                # Get breakdown scores
                breakdown = candidate.get('breakdown', {})
                
                row = {
                    'target_table': target_table,
                    'target_column': target_column,
                    'target_full_name': target_col_key,
                    'source_table': source_table,
                    'source_column': source_column,
                    'source_full_name': source_full,
                    'final_score': f"{candidate['score']:.4f}",
                    'similarity_method': candidate['method'],
                    'decision': candidate.get('decision', 'N/A'),
                    'name_similarity': f"{breakdown.get('name_similarity', 0):.4f}",
                    'regex_score': f"{breakdown.get('regex_score', 0):.4f}",
                    'entropy_score': f"{breakdown.get('entropy_score', 0):.4f}",
                    'cardinality_score': f"{breakdown.get('cardinality_score', 0):.4f}",
                    'null_ratio_score': f"{breakdown.get('null_ratio_score', 0):.4f}",
                    'mapping_type': mapping.get('mapping_type', '') if is_mapped else '',
                    'mapping_confidence': mapping.get('confidence', '') if is_mapped else '',
                    'transformation_sql': mapping.get('transformation_sql', '') if is_mapped else '',
                    'rank': rank
                }
                writer.writerow(row)
    
    print(f"\n[OK] Mappings exported to: {filename}")
    return filename


def export_rank1_mappings_to_csv(result: Dict[str, Any], filename: str = None) -> str:
    """Export only rank 1 (top candidate) mapping results to CSV"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mapping_results_rank1_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'target_table', 'target_column', 'target_full_name',
            'source_table', 'source_column', 'source_full_name',
            'final_score', 'similarity_method', 'decision',
            'name_similarity', 'regex_score', 'entropy_score', 'cardinality_score', 'null_ratio_score',
            'mapping_type', 'mapping_confidence',
            'transformation_sql'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Iterate through all target columns
        for target_col_key, mapping in result['mappings'].items():
            target_table, target_column = target_col_key.split('.', 1)
            
            # Get candidate scores for this target column
            candidates = result['candidate_scores'].get(target_col_key, [])
            
            # Get the mapped source columns
            mapped_sources = mapping.get('source_columns', [])
            
            # Write only the rank 1 (first/top) candidate
            if candidates:
                rank = 1
                candidate = candidates[0]  # Top candidate only
                source_table = candidate['source_table']
                source_column = candidate['source_column']
                source_full = f"{source_table}.{source_column}"
                
                # Check if this source is in the final mapping
                is_mapped = source_full in mapped_sources
                
                # Get breakdown scores
                breakdown = candidate.get('breakdown', {})
                
                row = {
                    'target_table': target_table,
                    'target_column': target_column,
                    'target_full_name': target_col_key,
                    'source_table': source_table,
                    'source_column': source_column,
                    'source_full_name': source_full,
                    'final_score': f"{candidate['score']:.4f}",
                    'similarity_method': candidate['method'],
                    'decision': candidate.get('decision', 'N/A'),
                    'name_similarity': f"{breakdown.get('name_similarity', 0):.4f}",
                    'regex_score': f"{breakdown.get('regex_score', 0):.4f}",
                    'entropy_score': f"{breakdown.get('entropy_score', 0):.4f}",
                    'cardinality_score': f"{breakdown.get('cardinality_score', 0):.4f}",
                    'null_ratio_score': f"{breakdown.get('null_ratio_score', 0):.4f}",
                    'mapping_type': mapping.get('mapping_type', '') if is_mapped else '',
                    'mapping_confidence': mapping.get('confidence', '') if is_mapped else '',
                    'transformation_sql': mapping.get('transformation_sql', '') if is_mapped else ''
                }
                writer.writerow(row)
    
    print(f"\n[OK] Rank 1 mappings exported to: {filename}")
    return filename


# ==========================================
# Build Graph
# ==========================================

def build_pipeline() -> StateGraph:
    """Build the LangGraph pipeline"""
    
    workflow = StateGraph(PipelineState)
    
    # Add nodes
    workflow.add_node("extract_schema", node_extract_schema)
    workflow.add_node("profile_data", node_profile_data)
    workflow.add_node("discover_entities", node_discover_entities)
    workflow.add_node("discover_lookups", node_discover_lookups)
    workflow.add_node("score_candidates", node_score_candidates)
    workflow.add_node("semantic_mapping", node_semantic_mapping)
    workflow.add_node("map_lookups", node_map_lookups)
    
    # Define edges (pipeline order)
    workflow.add_edge(START, "extract_schema")
    workflow.add_edge("extract_schema", "profile_data")
    workflow.add_edge("profile_data", "discover_entities")
    workflow.add_edge("discover_entities", "discover_lookups")
    workflow.add_edge("discover_lookups", "score_candidates")
    workflow.add_edge("score_candidates", "semantic_mapping")
    workflow.add_edge("semantic_mapping", "map_lookups")
    workflow.add_edge("map_lookups", END)
    
    return workflow.compile()


# ==========================================
# Execute Pipeline
# ==========================================

def run_pipeline(source_db: str, target_db: str, glossary_path: str = "business_glossary.json") -> Dict[str, Any]:
    """Execute the data discovery pipeline
    
    Args:
        source_db: Source database connection URL
        target_db: Target database connection URL
        glossary_path: Path to business glossary JSON file (default: business_glossary.json)
    """
    
    print("=" * 70)
    print("DATA DISCOVERY AGENT PIPELINE")
    print("=" * 70)
    
    # Load business glossary
    print("\n[LOADING BUSINESS GLOSSARY]")
    business_glossary = load_business_glossary(glossary_path)
    print(f"  [OK] Loaded {len(business_glossary.get('entities', {}))} entity definitions")
    print(f"  [OK] Loaded {len(business_glossary.get('columns', {}))} column definitions")
    
    pipeline = build_pipeline()
    
    initial_state: PipelineState = {
        "db_url": source_db,
        "target_db_url": target_db,
        "business_glossary": business_glossary,
        "source_schema": {},
        "target_schema": {},
        "source_profile": {},
        "target_profile": {},
        "source_entities": {},
        "target_entities": {},
        "source_lookups": {},
        "target_lookups": {},
        "lookup_mappings": {},
        "column_candidates": {},
        "candidate_scores": {},
        "mappings": {},
        "errors": []
    }
    
    result = pipeline.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 70)
    print("PIPELINE RESULTS")
    print("=" * 70)
    
    if result["errors"]:
        print("\n[WARNINGS]")
        for error in result["errors"]:
            print(f"  - {error}")
    
    print("\n[SOURCE ENTITIES]")
    for entity, tables in result["source_entities"].items():
        print(f"  {entity}: {tables}")
    
    print("\n[TARGET ENTITIES]")
    for entity, tables in result["target_entities"].items():
        print(f"  {entity}: {tables}")
    
    print("\n[SOURCE LOOKUP TABLES]")
    for table, info in result["source_lookups"].items():
        print(f"  {table}: {info.get('columns', [])[:3]}...")
    
    print("\n[TARGET LOOKUP TABLES]")
    for table, info in result["target_lookups"].items():
        print(f"  {table}: {info.get('columns', [])[:3]}...")
    
    print("\n[LOOKUP MAPPINGS]")
    for source_lookup, mapping_info in result["lookup_mappings"].items():
        print(f"  {source_lookup} -> {mapping_info.get('target_table', 'N/A')}")
        print(f"    Mapped {len(mapping_info.get('mappings', []))} values")
    
    print("\n[COLUMN MAPPINGS (Sample)]")
    for target_col, mapping in list(result["mappings"].items())[:5]:
        print(f"  {target_col}:")
        print(f"    Type: {mapping.get('mapping_type')}")
        print(f"    Confidence: {mapping.get('confidence')}")
        print(f"    Sources: {mapping.get('source_columns')}")
    
    return result


# ==========================================
# Example Usage
# ==========================================

if __name__ == "__main__":
    # Configure database URLs
    SOURCE_DB = "postgresql://postgres:postgres@localhost:5432/front2_db"
    TARGET_DB = "postgresql://postgres:postgres@localhost:5432/var2_db"
    
    # Run pipeline
    result = run_pipeline(SOURCE_DB, TARGET_DB)
    
    # Access results
    print("\n\n[SUMMARY]")
    print(f"  - source_schema: {len(result['source_schema'])} tables")
    print(f"  - target_schema: {len(result['target_schema'])} tables")
    print(f"  - source_lookups: {len(result['source_lookups'])} lookup tables")
    print(f"  - target_lookups: {len(result['target_lookups'])} lookup tables")
    print(f"  - lookup_mappings: {len(result['lookup_mappings'])} lookup mappings")
    print(f"  - column_mappings: {len(result['mappings'])} columns mapped")
    print(f"  - errors: {len(result['errors'])} errors")
    
    # Export to CSV
    csv_file = export_mappings_to_csv(result)
    print(f"\n[DONE] Results saved to {csv_file}")
    
    # Export rank 1 mappings to separate CSV
    csv_file_rank1 = export_rank1_mappings_to_csv(result)
    print(f"[DONE] Rank 1 results saved to {csv_file_rank1}")
