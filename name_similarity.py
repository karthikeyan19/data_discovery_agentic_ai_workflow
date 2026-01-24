from rapidfuzz import fuzz
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# -----------------------------
# LLM + Embeddings Setup
# -----------------------------

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# -----------------------------
# Utility Functions
# -----------------------------

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# -----------------------------
# Fuzzy Similarity
# -----------------------------

def fuzzy_similarity(a: str, b: str) -> float:
    return fuzz.token_sort_ratio(a, b) / 100.0

# -----------------------------
# Embedding Similarity
# -----------------------------

_embedding_cache = {}

def embedding_similarity(a: str, b: str) -> float:
    key = tuple(sorted([a, b]))
    if key in _embedding_cache:
        return _embedding_cache[key]

    va = embeddings.embed_query(a.replace("_", " "))
    vb = embeddings.embed_query(b.replace("_", " "))

    score = cosine_similarity(va, vb)
    _embedding_cache[key] = score

    return score

# -----------------------------
# LLM Similarity (Fallback)
# -----------------------------

name_similarity_prompt = ChatPromptTemplate.from_template(
    """
You are a data migration expert.

Compare the semantic meaning of two database column names.

Column A: {a}
Column B: {b}

{business_glossary_context}

Return ONLY a float number between 0 and 1.
No explanation.
"""
)

name_similarity_chain = name_similarity_prompt | llm | StrOutputParser()

def llm_similarity(a: str, b: str, business_glossary: dict = None) -> float:
    try:
        # Build business glossary context
        glossary_context = ""
        if business_glossary and "columns" in business_glossary:
            glossary_context = "\nBusiness Glossary - Column Definitions:\n"
            for col_name, col_def in business_glossary["columns"].items():
                # Handle both string and dict formats
                if isinstance(col_def, dict):
                    description = col_def.get('description', 'No description')
                    business_term = col_def.get('business_term', '')
                else:
                    description = str(col_def)
                    business_term = ''
                
                glossary_context += f"- {col_name}: {description}"
                if business_term:
                    glossary_context += f" (Business Term: {business_term})"
                glossary_context += "\n"
        
        response = name_similarity_chain.invoke({
            "a": a.replace("_", " "),
            "b": b.replace("_", " "),
            "business_glossary_context": glossary_context
        }).strip()
        
        # Parse the response - handle potential non-numeric responses
        score = float(response)
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        return round(score, 4)
    except ValueError as e:
        print(f"Warning: LLM returned non-numeric response for '{a}' vs '{b}': {response}")
        return 0.5  # Return neutral score instead of 0
    except Exception as e:
        print(f"Warning: LLM similarity failed for '{a}' vs '{b}': {str(e)}")
        return 0.5  # Return neutral score instead of 0

# -----------------------------
# Hybrid Similarity Engine
# -----------------------------

def hybrid_name_similarity(a: str, b: str, business_glossary: dict = None) -> dict:
    """
    Returns final similarity score using hybrid approach.
    
    Args:
        a: First column name
        b: Second column name
        business_glossary: Optional business glossary with column definitions
            Expected format: {"columns": {"column_name": {"description": "...", "business_term": "..."}}}
    """
    # Step 1: Fuzzy
    fz = fuzzy_similarity(a, b)

    if fz >= 0.85:
        return {
            "score": round(fz, 4),
            "method": "fuzzy"
        }

    # Step 2: Embedding
    emb = embedding_similarity(a, b)

    if emb >= 0.85:
        return {
            "score": round(emb, 4),
            "method": "embedding"
        }

    # Step 3: LLM
    llm_score = llm_similarity(a, b, business_glossary)

    return {
        "score": llm_score,
        "method": "llm"
    }

# -----------------------------
# Example Tests
# -----------------------------

if __name__ == "__main__":
    pairs = [
        ("dob", "date_of_birth"),
        ("msisdn", "mobile_number"),
        ("cust_id", "customer_identifier"),
        ("acct_no", "billing_account_id"),
        ("fname", "first_name"),
        ("gender", "email"),
        ("verification_status", "kyc_status"),
        ("imsi", "sim_id")
    ]

    for a, b in pairs:
        res = hybrid_name_similarity(a, b)
        print(f"{a:25} <-> {b:30} → {res}")
