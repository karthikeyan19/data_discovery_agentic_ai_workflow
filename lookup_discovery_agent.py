from sqlalchemy import create_engine, inspect, text
import pandas as pd

LOOKUP_KEYWORDS = [
    "lookup", "ref", "dim", "type", "code", "enum", "status", "class", "master"
]


def is_lookup_table_name(table_name: str) -> bool:
    return any(k in table_name.lower() for k in LOOKUP_KEYWORDS)


def is_low_cardinality(engine, table: str, threshold: int = 200) -> bool:
    try:
        df = pd.read_sql(text(f"SELECT COUNT(*) AS cnt FROM {table}"), engine)
        return int(df.iloc[0]["cnt"]) <= threshold
    except:
        return False


def is_lookup_by_columns(columns: list) -> bool:
    col_names = [c["column"].lower() for c in columns]

    has_code = any("code" in c or c.endswith("_cd") for c in col_names)
    has_desc = any("desc" in c or "name" in c or "label" in c for c in col_names)

    return has_code and has_desc


def lookup_discovery_agent(db_url: str, schema: dict) -> dict:
    """
    Discovers lookup / dimension tables.
    """

    engine = create_engine(db_url)
    inspector = inspect(engine)

    lookups = {}

    for table, meta in schema.items():

        score = 0
        scoring_details = []

        # 1. Name heuristic
        if is_lookup_table_name(table):
            score += 2
            scoring_details.append(f"name_match(+2)")

        # 2. Column pattern (code + description)
        if is_lookup_by_columns(meta["columns"]):
            score += 2
            scoring_details.append(f"column_pattern(+2)")

        # 3. Cardinality check (low cardinality = few distinct values)
        if is_low_cardinality(engine, table):
            score += 2
            scoring_details.append(f"low_cardinality(+2)")

        # 4. PK simple (single column primary key)
        if len(meta["primary_key"]) == 1:
            score += 1
            scoring_details.append(f"simple_pk(+1)")

        print(f"    {table}: score={score} [{', '.join(scoring_details) if scoring_details else 'no matches'}]")

        # Lowered threshold from 4 to 3 for better discovery
        if score >= 3:
            lookups[table] = {
                "columns": [c["column"] for c in meta["columns"]],
                "primary_key": meta["primary_key"],
                "score": score,
                "scoring": scoring_details
            }

    return lookups


# ==========================================
# Test Main Method
# ==========================================

if __name__ == "__main__":
    from schema_extract_agent import extract_schema_metadata
    
    # Test database URLs
    SOURCE_DB = "postgresql://postgres:postgres@localhost:5432/front_db"
    TARGET_DB = "postgresql://postgres:postgres@localhost:5432/var_db"
    
    print("=" * 70)
    print("LOOKUP TABLE DISCOVERY AGENT - TEST")
    print("=" * 70)
    
    # Test on source database
    print("\n[1] Testing on SOURCE database...")
    try:
        source_schema = extract_schema_metadata(SOURCE_DB)
        source_lookups = lookup_discovery_agent(SOURCE_DB, source_schema)
        
        print(f"[OK] Found {len(source_lookups)} lookup tables in source database")
        for table, info in source_lookups.items():
            print(f"  - {table}")
            print(f"    Columns: {info['columns'][:3]}{'...' if len(info['columns']) > 3 else ''}")
            print(f"    Primary Key: {info['primary_key']}")
    except Exception as e:
        print(f"[ERROR] Source database test failed: {str(e)}")
    
    # Test on target database
    print("\n[2] Testing on TARGET database...")
    try:
        target_schema = extract_schema_metadata(TARGET_DB)
        target_lookups = lookup_discovery_agent(TARGET_DB, target_schema)
        
        print(f"[OK] Found {len(target_lookups)} lookup tables in target database")
        for table, info in target_lookups.items():
            print(f"  - {table}")
            print(f"    Columns: {info['columns'][:3]}{'...' if len(info['columns']) > 3 else ''}")
            print(f"    Primary Key: {info['primary_key']}")
    except Exception as e:
        print(f"[ERROR] Target database test failed: {str(e)}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
