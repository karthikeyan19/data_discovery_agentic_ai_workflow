import pandas as pd
import re
import math
from collections import Counter
from sqlalchemy import create_engine, inspect, text

# -----------------------------
# Utility Functions
# -----------------------------

def infer_regex(samples: list) -> str:
    """
    Infer dominant regex pattern from column samples.
    """
    if not samples:
        return None

    patterns = []

    for s in samples:
        if s is None:
            continue

        s = str(s)

        if re.fullmatch(r"\d+", s):
            patterns.append(r"^\d+$")
        elif re.fullmatch(r"[A-Za-z ]+", s):
            patterns.append(r"^[A-Za-z ]+$")
        elif re.fullmatch(r"[A-Za-z0-9_-]+", s):
            patterns.append(r"^[A-Za-z0-9_-]+$")
        elif re.fullmatch(r"\d{3}-\d{2}-\d{4}", s):
            patterns.append(r"^\d{3}-\d{2}-\d{4}$")  # US SSN
        elif re.fullmatch(r"\d{10}", s):
            patterns.append(r"^\d{10}$")            # US phone
        elif "@" in s:
            patterns.append(r"EMAIL")
        else:
            patterns.append("MIXED")

    return Counter(patterns).most_common(1)[0][0]


def shannon_entropy(values: list) -> float:
    """
    Compute Shannon entropy to measure randomness / uniqueness.
    """
    if not values:
        return 0.0

    counts = Counter(values)
    total = len(values)

    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return round(entropy, 4)


# -----------------------------
# Column Profiling Agent
# -----------------------------

def profile_table(engine, table: str, sample_limit: int = 1000) -> dict:
    """
    Profiles a table and computes statistical fingerprint per column.
    """
    profile = {}

    df = pd.read_sql(
        text(f"SELECT * FROM {table} LIMIT {sample_limit}"),
        engine
    )

    total_rows = len(df)

    for col in df.columns:
        non_null_values = df[col].dropna().astype(str).tolist()

        null_ratio = round((total_rows - len(non_null_values)) / max(total_rows, 1), 4)
        unique_ratio = round(len(set(non_null_values)) / max(len(non_null_values), 1), 4)
        cardinality = len(set(non_null_values))
        entropy = shannon_entropy(non_null_values)
        regex = infer_regex(non_null_values[:100])

        profile[col] = {
            "null_ratio": null_ratio,
            "unique_ratio": unique_ratio,
            "cardinality": cardinality,
            "entropy": entropy,
            "regex": regex,
            "samples": non_null_values[:10]
        }

    return profile


# -----------------------------
# Full DB Profiling Agent
# -----------------------------

def profile_database(db_url: str) -> dict:
    """
    Profiles entire database.
    """
    engine = create_engine(db_url)
    inspector = inspect(engine)

    db_profile = {}

    for table in inspector.get_table_names():
        db_profile[table] = profile_table(engine, table)
    print(f"Completed profiling database: {db_profile}")
    return db_profile


# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":
    SOURCE_DB = "postgresql://postgres:postgres@localhost:5432/front_db"

    profile = profile_database(SOURCE_DB)

    import json
    print(json.dumps(profile, indent=2))
