import networkx as nx
from typing import Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from schema_extract_agent import extract_schema_metadata

# -----------------------------
# LLM Setup
# -----------------------------

llm = ChatOpenAI(model="gpt-4o", temperature=0)

entity_prompt = ChatPromptTemplate.from_template(
    """
You are a senior telecom data architect.

Given the following database tables and columns:

{tables}

{business_glossary_context}

Identify the business entity these tables represent.
Return only the entity name.

Examples:
Customer, Account, Subscription, Address, Identity, KYC, Payment, Billing.

Return a single entity label.
"""
)

entity_chain = entity_prompt | llm | StrOutputParser()

# -----------------------------
# Relationship Graph Builder
# -----------------------------

def build_table_graph(schema: dict) -> nx.Graph:
    """
    Build table relationship graph using FK constraints.
    """
    G = nx.Graph()

    for table in schema.keys():
        G.add_node(table)

    for table, meta in schema.items():
        for fk in meta["foreign_keys"]:
            src = table
            tgt = fk["ref_table"]
            G.add_edge(src, tgt)

    return G

# -----------------------------
# Graph Clustering → Entity Groups
# -----------------------------

def discover_table_clusters(schema: dict) -> Dict[str, List[str]]:
    """
    Discover entity clusters using FK graph connectivity.
    """
    G = build_table_graph(schema)
    clusters = list(nx.connected_components(G))

    return {
        f"cluster_{i+1}": list(cluster)
        for i, cluster in enumerate(clusters)
    }

# -----------------------------
# Semantic Entity Labeling Agent
# -----------------------------

def label_entity(cluster_tables: List[str], schema: dict, business_glossary: dict = None) -> str:
    """
    Use LLM to label entity meaning.
    """
    table_info = []

    for t in cluster_tables:
        try:
            if t not in schema:
                print(f"  ⚠ Table '{t}' not found in schema")
                continue
            
            cols = [c["column"] for c in schema[t]["columns"]]
            table_info.append(f"{t}: {', '.join(cols[:10])}")
        except KeyError as e:
            print(f"  ⚠ Error processing table '{t}': {e}")
            continue

    if not table_info:
        return "Unknown Entity"
    
    # Build business glossary context
    glossary_context = ""
    if business_glossary and "entities" in business_glossary:
        glossary_context = "\n\nBusiness Glossary - Entity Definitions:\n"
        for entity_name, entity_def in business_glossary["entities"].items():
            # Handle both string and dict formats
            if isinstance(entity_def, dict):
                description = entity_def.get('description', 'No description')
            else:
                description = str(entity_def)
            glossary_context += f"- {entity_name}: {description}\n"
    
    return entity_chain.invoke({
        "tables": "\n".join(table_info),
        "business_glossary_context": glossary_context
    }).strip()

# -----------------------------
# Full Entity Discovery Pipeline
# -----------------------------

def entity_discovery_agent(schema: dict, business_glossary: dict = None) -> dict:
    """
    Discovers business entities in a relational schema.
    
    Args:
        schema: Database schema dictionary
        business_glossary: Optional business glossary with entity definitions
            Expected format: {"entities": {"EntityName": {"description": "..."}}}
    """
    try:
        clusters = discover_table_clusters(schema)
        
        if not clusters:
            print("  ⚠ No clusters discovered from schema")
            return {}

        entities = {}

        for cluster_id, tables in clusters.items():
            try:
                entity_name = label_entity(tables, schema, business_glossary)
                if entity_name and entity_name != "Unknown Entity":
                    entities[entity_name] = {"tables": tables}
            except Exception as e:
                print(f"  ⚠ Error labeling cluster {cluster_id}: {e}")
                continue
        
        print(f"Discovered {len(entities)} business entities.")
        return entities
    
    except Exception as e:
        print(f"  ✗ Entity discovery failed: {e}")
        return {}

# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":
    
    # Database URLs
    SOURCE_DB = "postgresql://postgres:postgres@localhost:5432/front_db"
    
    # Extract schema from real database
    print("Extracting schema from database...")
    schema = extract_schema_metadata(SOURCE_DB)
    
    if not schema:
        print("No schema extracted. Using sample schema for testing...")
        schema = {
            "client": {
                "columns": [
                    {"column": "client_id", "type": "VARCHAR", "nullable": False, "default": None},
                    {"column": "first_name", "type": "VARCHAR", "nullable": False, "default": None}
                ],
                "primary_key": ["client_id"],
                "foreign_keys": [],
                "unique_constraints": []
            },
            "client_identity": {
                "columns": [
                    {"column": "document_number", "type": "VARCHAR", "nullable": False, "default": None},
                    {"column": "client_id", "type": "VARCHAR", "nullable": False, "default": None}
                ],
                "primary_key": ["document_number"],
                "foreign_keys": [{"column": ["client_id"], "ref_table": "client", "ref_column": ["client_id"]}],
                "unique_constraints": []
            },
            "client_address": {
                "columns": [
                    {"column": "city", "type": "VARCHAR", "nullable": False, "default": None},
                    {"column": "client_id", "type": "VARCHAR", "nullable": False, "default": None}
                ],
                "primary_key": ["city"],
                "foreign_keys": [{"column": ["client_id"], "ref_table": "client", "ref_column": ["client_id"]}],
                "unique_constraints": []
            },
            "service_subscription": {
                "columns": [
                    {"column": "msisdn", "type": "VARCHAR", "nullable": False, "default": None}
                ],
                "primary_key": ["msisdn"],
                "foreign_keys": [],
                "unique_constraints": []
            },
            "billing_account": {
                "columns": [
                    {"column": "billing_account_id", "type": "VARCHAR", "nullable": False, "default": None}
                ],
                "primary_key": ["billing_account_id"],
                "foreign_keys": [],
                "unique_constraints": []
            }
        }
    
    print(f"Schema contains {len(schema)} tables: {list(schema.keys())}\n")
    
    # Discover entities
    entities = entity_discovery_agent(schema)

    print("\n--- DISCOVERED BUSINESS ENTITIES ---\n")
    for k, v in entities.items():
        print(k, "→", v)
