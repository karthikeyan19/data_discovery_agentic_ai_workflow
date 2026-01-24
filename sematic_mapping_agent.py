import json
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

semantic_mapping_prompt = ChatPromptTemplate.from_template("""
You are a senior enterprise data migration architect.

Your task:
Given a target database column and a list of candidate source columns with sample values,
determine the best mapping and any required transformation.

Target Column:
{target_column}

Candidate Source Columns:
{candidates}

Sample Values:
{samples}

{business_glossary_context}

Rules:
- If 1 source column is sufficient, return direct mapping.
- If multiple columns are needed, infer transformation (concat, decode, cast, case).
- If transformation is required, generate SQL-style expression.
- Use business glossary definitions to guide mapping decisions.

Return strictly in JSON format:

{{
  "mapping_type": "direct | derived | conditional | lookup",
  "source_columns": ["table.column", ...],
  "confidence": 0.xx,
  "transformation_sql": "SQL_EXPRESSION"
}}
""")

output_parser = JsonOutputParser()
semantic_mapping_chain = semantic_mapping_prompt | llm | output_parser

def semantic_mapping_agent(
    target_column: str,
    candidate_sources: list,
    sample_values: list,
    business_glossary: dict = None
) -> dict:
    """
    Run semantic mapping inference.
    
    Args:
        target_column: Target column name
        candidate_sources: List of candidate source columns
        sample_values: Sample values for each candidate
        business_glossary: Optional business glossary with column definitions
            Expected format: {"columns": {"column_name": {"description": "...", "business_term": "..."}}}
    """
    # Build business glossary context
    glossary_context = ""
    if business_glossary and "columns" in business_glossary:
        glossary_context = "\n\nBusiness Glossary - Column Definitions:\n"
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

    prompt = semantic_mapping_prompt.format(
        target_column=target_column,
        candidates=candidate_sources,
        samples=sample_values,
        business_glossary_context=glossary_context
    )

    response = llm.invoke(prompt).content.strip()
    
    # Remove markdown code fences if present
    if response.startswith("```json"):
        response = response[7:]  # Remove ```json
    if response.startswith("```"):
        response = response[3:]  # Remove ```
    if response.endswith("```"):
        response = response[:-3]  # Remove trailing ```
    response = response.strip()
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {
            "mapping_type": "unknown",
            "source_columns": [],
            "confidence": 0.0,
            "transformation_sql": None,
            "raw_response": response
        }
# -----------------------------
# Sample Runs
# -----------------------------

if __name__ == "__main__":

    # -------- Example 1: Derived mapping (concat) --------

    target_column = "party.full_name"

    candidate_sources = [
        "client.first_name",
        "client.last_name"
    ]

    sample_values = [
        ["John", "Emily"],
        ["Miller", "Johnson"]
    ]

    result = semantic_mapping_agent(
        target_column,
        candidate_sources,
        sample_values
    )

    print("\n--- Example 1: FULL_NAME Mapping ---\n")
    print(json.dumps(result, indent=2))

    # -------- Example 2: Direct mapping --------

    target_column = "party.email"

    candidate_sources = [
        "client.email_address",
        "client.secondary_phone"
    ]

    sample_values = [
        ["john.miller@gmail.com", "emily.johnson@yahoo.com"],
        ["4155559988"]
    ]

    result = semantic_mapping_agent(
        target_column,
        candidate_sources,
        sample_values
    )

    print("\n--- Example 2: EMAIL Mapping ---\n")
    print(json.dumps(result, indent=2))

    # -------- Example 3: Conditional mapping --------

    target_column = "kyc_verification.kyc_status"

    candidate_sources = [
        "client_identity.verification_status"
    ]

    sample_values = [
        ["VERIFIED", "PENDING", "REJECTED"]
    ]

    result = semantic_mapping_agent(
        target_column,
        candidate_sources,
        sample_values
    )

    print("\n--- Example 3: KYC STATUS Mapping ---\n")
    print(json.dumps(result, indent=2))
