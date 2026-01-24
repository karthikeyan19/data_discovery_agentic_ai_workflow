from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import json

# ==========================================
# LLM Configuration
# ==========================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ==========================================
# Output Schema
# ==========================================

class LookupMapping(BaseModel):
    """Model for individual lookup value mapping"""
    source_code: str = Field(description="Source system code/value")
    target_code: str = Field(description="Target system code/value")
    confidence: float = Field(description="Confidence score between 0 and 1")


class LookupMappingResult(BaseModel):
    """Model for the complete mapping result"""
    mappings: List[LookupMapping] = Field(description="List of mapped lookup values")


# ==========================================
# Prompts
# ==========================================

lookup_mapping_prompt = ChatPromptTemplate.from_template("""
You are a senior telecom data migration architect.

Your task:
Map lookup values between source and target systems based on semantic meaning.

SOURCE LOOKUP TABLE:
{source_lookup}

TARGET LOOKUP TABLE:
{target_lookup}

Instructions:
1. Analyze both lookup tables
2. Find semantic matches between source and target codes
3. Assign confidence scores (0.0-1.0) based on match quality
4. Return only valid JSON

Return JSON with exactly this structure:
{{
  "mappings": [
    {{"source_code": "...", "target_code": "...", "confidence": 0.95}},
    {{"source_code": "...", "target_code": "...", "confidence": 0.85}}
  ]
}}
""")


# ==========================================
# Lookup Mapping Agent
# ==========================================

def lookup_mapping_agent(
    source_lookup_df,
    target_lookup_df
) -> Dict[str, Any]:
    """
    Maps lookup values between source and target systems using LLM.
    
    Args:
        source_lookup_df: DataFrame with source lookup table data
        target_lookup_df: DataFrame with target lookup table data
    
    Returns:
        Dictionary with mapped lookup values and confidence scores
    """
    
    # Convert DataFrames to readable format
    source_lookup = source_lookup_df.to_dict(orient="records")
    target_lookup = target_lookup_df.to_dict(orient="records")
    
    # Create parser for JSON output
    parser = JsonOutputParser(pydantic_object=LookupMappingResult)
    
    # Build LCEL chain: prompt | llm | parser
    chain = lookup_mapping_prompt | llm | parser
    
    try:
        # Invoke the chain
        result = chain.invoke({
            "source_lookup": json.dumps(source_lookup, indent=2),
            "target_lookup": json.dumps(target_lookup, indent=2)
        })
        
        return result
        
    except Exception as e:
        # Handle parsing errors
        print(f"[ERROR] Mapping failed: {str(e)}")
        return {
            "mappings": [],
            "error": str(e)
        }


# ==========================================
# Test Main Method
# ==========================================

if __name__ == "__main__":
    import pandas as pd
    
    print("=" * 70)
    print("LOOKUP MAPPING AGENT - TEST")
    print("=" * 70)
    
    # Sample source lookup table
    source_lookup_df = pd.DataFrame([
        {"code": "ACT", "description": "Active"},
        {"code": "INA", "description": "Inactive"},
        {"code": "SUS", "description": "Suspended"},
        {"code": "TRM", "description": "Terminated"}
    ])
    
    # Sample target lookup table
    target_lookup_df = pd.DataFrame([
        {"code": "A", "description": "Active Status"},
        {"code": "I", "description": "Inactive Status"},
        {"code": "S", "description": "On Hold"},
        {"code": "D", "description": "Deactivated"}
    ])
    
    print("\n[1] SOURCE LOOKUP TABLE:")
    print(source_lookup_df.to_string())
    
    print("\n[2] TARGET LOOKUP TABLE:")
    print(target_lookup_df.to_string())
    
    print("\n[3] Running lookup mapping agent...")
    try:
        result = lookup_mapping_agent(source_lookup_df, target_lookup_df)
        
        print("\n[OK] Mapping completed!")
        print("\n[MAPPINGS]:")
        if "mappings" in result:
            for mapping in result["mappings"]:
                print(f"  {mapping['source_code']} -> {mapping['target_code']} (confidence: {mapping['confidence']:.2f})")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
