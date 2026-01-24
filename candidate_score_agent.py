"""
Candidate Score Agent
----------------------

Computes a final confidence score for a source → target column mapping
by combining semantic, statistical, and structural signals.

Used in:
  Schema Matching
  Column Discovery
  M&A Data Migration
  ETL Automation

Author: You 😄
"""

from typing import Dict

# -----------------------------
# Utility Similarity Functions
# -----------------------------

def safe_similarity(a: float, b: float) -> float:
    """
    Computes numeric similarity safely.
    Returns value between 0 and 1.
    """
    if max(a, b) == 0:
        return 1.0
    return max(0.0, 1 - abs(a - b) / max(a, b))


def regex_similarity(src_regex: str, tgt_regex: str) -> float:
    """
    Computes regex compatibility score.
    """
    if not src_regex or not tgt_regex:
        return 0.5

    if src_regex == tgt_regex:
        return 1.0

    if src_regex in tgt_regex or tgt_regex in src_regex:
        return 0.8

    # Domain heuristics
    if src_regex.startswith("^\\d") and tgt_regex.startswith("^\\d"):
        return 0.6

    return 0.0


# -----------------------------
# Candidate Score Agent
# -----------------------------

def candidate_score_agent(
    src_profile: Dict,
    tgt_profile: Dict,
    name_similarity: float,
    weights: Dict[str, float] = None
) -> Dict:
    """
    Compute final confidence score for a column mapping.

    Parameters:
        src_profile (dict): source column profiling stats
        tgt_profile (dict): target column profiling stats
        name_similarity (float): semantic similarity score (0–1)
        weights (dict): optional override weights

    Returns:
        dict with:
          - final_score
          - breakdown
          - decision
    """

    # Default enterprise-grade weights
    default_weights = {
        "name": 0.35,
        "regex": 0.25,
        "entropy": 0.15,
        "cardinality": 0.15,
        "null_ratio": 0.10
    }

    if weights:
        default_weights.update(weights)

    # Compute partial scores
    regex_score = regex_similarity(
        src_profile.get("regex"),
        tgt_profile.get("regex")
    )

    entropy_score = safe_similarity(
        src_profile.get("entropy", 0),
        tgt_profile.get("entropy", 0)
    )

    cardinality_score = safe_similarity(
        src_profile.get("cardinality", 0),
        tgt_profile.get("cardinality", 0)
    )

    null_score = 1 - abs(
        src_profile.get("null_ratio", 0) - tgt_profile.get("null_ratio", 0)
    )

    # Weighted final score
    final_score = (
        default_weights["name"] * name_similarity +
        default_weights["regex"] * regex_score +
        default_weights["entropy"] * entropy_score +
        default_weights["cardinality"] * cardinality_score +
        default_weights["null_ratio"] * null_score
    )

    final_score = round(final_score, 4)

    decision = decide_mapping(final_score)
    result = {
        "final_score": final_score,
        "decision": decision,
        "breakdown": {
            "name_similarity": round(name_similarity, 4),
            "regex_score": round(regex_score, 4),
            "entropy_score": round(entropy_score, 4),
            "cardinality_score": round(cardinality_score, 4),
            "null_ratio_score": round(null_score, 4)
        }
    }
    print(f"Candidate Score Result: {result}")
    return result


# -----------------------------
# Decision Agent
# -----------------------------

def decide_mapping(score: float) -> str:
    """
    Determines action based on confidence score.
    """
    if score >= 0.90:
        return "AUTO_ACCEPT"
    elif score >= 0.75:
        return "HUMAN_REVIEW"
    else:
        return "REJECT"


# -----------------------------
# Batch Scoring Helper
# -----------------------------

def batch_candidate_scoring(
    candidate_matches: Dict,
    source_profiles: Dict,
    target_profiles: Dict,
    similarity_func
) -> Dict:
    """
    Scores all candidate mappings.

    candidate_matches:
        {
            "party.full_name": [
                {"source": "client.first_name"},
                {"source": "client.last_name"}
            ]
        }
    """

    results = {}

    for tgt, candidates in candidate_matches.items():
        tgt_table, tgt_col = tgt.split(".")
        tgt_profile = target_profiles[tgt_table][tgt_col]

        scored = []

        for cand in candidates:
            src_table, src_col = cand["source"].split(".")
            src_profile = source_profiles[src_table][src_col]

            name_sim = similarity_func(src_col, tgt_col)

            score = candidate_score_agent(
                src_profile,
                tgt_profile,
                name_sim
            )

            scored.append({
                "source": cand["source"],
                **score
            })

        results[tgt] = sorted(
            scored, key=lambda x: x["final_score"], reverse=True
        )

    return results


# -----------------------------
# Example Run
# -----------------------------

if __name__ == "__main__":

    src_profile = {
        "regex": "^\\d{10}$",
        "entropy": 9.8,
        "cardinality": 1000,
        "null_ratio": 0.01
    }

    tgt_profile = {
        "regex": "^\\d{10}$",
        "entropy": 9.9,
        "cardinality": 995,
        "null_ratio": 0.02
    }

    name_similarity = 0.96

    score = candidate_score_agent(
        src_profile,
        tgt_profile,
        name_similarity
    )

    print("\n--- CANDIDATE SCORE ---\n")
    print(score)
