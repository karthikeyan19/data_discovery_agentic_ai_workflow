from sqlalchemy import create_engine, inspect


def extract_schema_metadata(db_url: str) -> dict:
    """
    Extract full relational schema metadata from a database.
    Returns tables, columns, data types, PK, FK, constraints.
    """
    try:
        engine = create_engine(db_url)
        inspector = inspect(engine)
        
        schema = {}
        
        for table in inspector.get_table_names():
            columns = []
            pk = inspector.get_pk_constraint(table)
            fks = inspector.get_foreign_keys(table)
            uniques = inspector.get_unique_constraints(table)
            
            # Extract columns for this table
            for col in inspector.get_columns(table):
                columns.append({
                    "column": col['name'],
                    "type": str(col['type']),
                    "nullable": col['nullable'],
                    "default": col.get('default')
                })
            
            # Build schema for this table
            schema[table] = {
                "columns": columns,
                "primary_key": pk.get('constrained_columns', []),
                "foreign_keys": [
                    {
                        "column": fk['constrained_columns'],
                        "ref_table": fk['referred_table'],
                        "ref_column": fk['referred_columns']
                    } for fk in fks
                ],
                "unique_constraints": [uc['column_names'] for uc in uniques]
            }
            
            print(f"Extracted schema for table: {table}")
        
        return schema
    
    except Exception as e:
        print(f"Error extracting schema: {e}")
        return {}

# Example usage
if __name__ == "__main__":
    SOURCE_DB = "postgresql://postgres:postgres@localhost:5432/front_db"
    TARGET_DB = "postgresql://postgres:postgres@localhost:5432/var_db"


    source_schema = extract_schema_metadata(SOURCE_DB)
    target_schema = extract_schema_metadata(TARGET_DB)


    print("SOURCE SCHEMA:", source_schema)
    print("TARGET SCHEMA:", target_schema)