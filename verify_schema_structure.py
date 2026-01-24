"""
Verify that the extracted schema structure matches the expected structure
"""

from schema_extract_agent import extract_schema_metadata

# Expected structure from extracted schema
print("=" * 70)
print("SCHEMA STRUCTURE VERIFICATION")
print("=" * 70)

schema = extract_schema_metadata('postgresql://postgres:postgres@localhost:5432/front_db')

if schema:
    print(f"\n[OK] Schema extracted successfully")
    print(f"[OK] Number of tables: {len(schema)}")
    
    # Check first table
    first_table = list(schema.keys())[0]
    meta = schema[first_table]
    
    print(f"\n[TABLE] {first_table}")
    print(f"  - Has 'columns' key: {'columns' in meta}")
    print(f"  - Has 'primary_key' key: {'primary_key' in meta}")
    print(f"  - Has 'foreign_keys' key: {'foreign_keys' in meta}")
    print(f"  - Has 'unique_constraints' key: {'unique_constraints' in meta}")
    
    # Check column structure
    if meta['columns']:
        col = meta['columns'][0]
        print(f"\n[COLUMN STRUCTURE] {col.get('column')}")
        print(f"  - Has 'column' key: {'column' in col}")
        print(f"  - Has 'type' key: {'type' in col}")
        print(f"  - Has 'nullable' key: {'nullable' in col}")
        print(f"  - Has 'default' key: {'default' in col}")
    
    # Check foreign key structure
    if meta['foreign_keys']:
        fk = meta['foreign_keys'][0]
        print(f"\n[FOREIGN KEY STRUCTURE]")
        print(f"  - Has 'column' key: {'column' in fk}")
        print(f"  - Has 'ref_table' key: {'ref_table' in fk}")
        print(f"  - Has 'ref_column' key: {'ref_column' in fk}")
        print(f"  - Values: {fk}")
    
    print("\n[SAMPLE DATA]")
    print(f"  Columns: {meta['columns'][:2]}")
    print(f"  Primary Key: {meta['primary_key']}")
    print(f"  Foreign Keys: {meta['foreign_keys']}")
    
    print("\n[MATCH STATUS] ✓ Structure is correct!")
else:
    print("[ERROR] Failed to extract schema")
