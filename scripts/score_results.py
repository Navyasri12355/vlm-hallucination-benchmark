"""
score_results.py
----------------
Scores raw VLM responses against ground truth labels.
Produces per-record scores and aggregated accuracy tables.

Usage:
    python scripts/score_results.py \
        --responses  results/llava_raw_responses.json \
        --scores     results/llava_scores.json \
        --summary    results/llava_summary.json

Output files:
    llava_scores.json   — one record per benchmark item with:
                            normalised_response, correct (bool), score_method
    llava_summary.json  — accuracy broken down by:
                            subcategory, difficulty, question_type, category

Scoring methods by subcategory:
    H1a/b/c, H2a, H3a/b, H5a/c  → binary yes/no match
    H5b                           → A/B choice match
    H4 free_form                  → exact integer match
    H4 binary_correct/wrong       → binary yes/no match
    H5d, H7c                      → keyword denial match
    H2b                           → skipped (ambiguous ground truth)
"""

import json
import re
import argparse
import logging
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Words that indicate a "no" response
NO_KEYWORDS  = {"no", "not", "none", "never", "cannot", "doesn't", "don't",
                "isn't", "aren't", "absent", "missing", "without", "false"}

# Words that indicate a "yes" response
YES_KEYWORDS = {"yes", "correct", "true", "indeed", "certainly", "absolutely",
                "definitely", "visible", "present", "there is", "there are"}

# Subcategories with ambiguous/unanswerable ground truth — skip scoring
SKIP_SUBCATS = {"H2b"}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Response normalisation
# ---------------------------------------------------------------------------

def normalise_yes_no(response: str) -> str | None:
    """
    Extract a yes/no signal from a free-text response.
    Returns 'yes', 'no', or None if ambiguous.

    Strategy:
      1. Check the first word — models often lead with Yes/No
      2. Scan for strong keyword signals
      3. Return None if conflicting or absent
    """
    if not response:
        return None

    text = response.lower().strip()

    # Check first word first — most reliable signal
    first_word = re.split(r"[\s,\.!]", text)[0]
    if first_word in ("yes", "yeah", "yep", "correct", "true"):
        return "yes"
    if first_word in ("no", "nope", "not", "false"):
        return "no"

    # Scan for keyword signals
    has_yes = any(kw in text for kw in YES_KEYWORDS)
    has_no  = any(kw in text for kw in NO_KEYWORDS)

    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"

    return None   # ambiguous


def normalise_choice(response: str) -> str | None:
    """
    Extract an A/B choice from a response.
    Returns 'A', 'B', or None.
    """
    if not response:
        return None
    text = response.strip()

    # Look for explicit (A) or (B) or standalone A/B
    match = re.search(r"\b([AB])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for option labels spelled out
    text_lower = text.lower()
    if text_lower.startswith("option a") or text_lower.startswith("a)"):
        return "A"
    if text_lower.startswith("option b") or text_lower.startswith("b)"):
        return "B"

    return None


def normalise_integer(response: str) -> str | None:
    """
    Extract the first integer from a response.
    Returns the integer as a string, or None.
    """
    if not response:
        return None
    # Written-out numbers for small counts
    word_to_int = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10,
    }
    text = response.lower().strip()
    for word, val in word_to_int.items():
        if re.search(rf"\b{word}\b", text):
            return str(val)
    # Digit extraction
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return match.group(1)
    return None


def normalise_denial(response: str, absent_object: str) -> str | None:
    """
    Score H5d / H7c open-ended refusal questions.
    Correct response should deny the existence of absent_object.

    Returns 'denial' if the response contains a denial signal, else 'no_denial'.
    """
    if not response:
        return "no_denial"

    text = response.lower().strip()
    obj  = absent_object.lower().strip()

    # Check for explicit denial keywords
    has_denial = any(kw in text for kw in NO_KEYWORDS)

    # Bonus: check if the object name appears alongside a denial
    obj_mentioned  = obj in text
    denial_near_obj = obj_mentioned and has_denial

    if denial_near_obj or (has_denial and not _is_affirmative(text)):
        return "denial"
    return "no_denial"


def _is_affirmative(text: str) -> bool:
    """Return True if the text reads as primarily affirmative."""
    affirmative_starts = ("yes", "sure", "the ", "it is", "there is",
                          "there are", "i can see", "i see")
    return any(text.startswith(s) for s in affirmative_starts)


# ---------------------------------------------------------------------------
# Per-record scoring
# ---------------------------------------------------------------------------

def score_record(record: dict) -> dict:
    """
    Score a single response record.
    Returns the record augmented with:
        normalised_response  — the extracted answer token
        correct              — True/False/None (None = unscored)
        score_method         — which scoring method was applied
    """
    subcat   = record.get("subcategory", "")
    gt       = record.get("ground_truth", "").strip().lower()
    response = record.get("raw_response") or ""
    meta     = record.get("metadata", {})
    q_type   = meta.get("question_type", "")   # used for H4

    # Skip subcategories with ambiguous ground truth
    if subcat in SKIP_SUBCATS:
        return {**record,
                "normalised_response": None,
                "correct": None,
                "score_method": "skipped_ambiguous"}

    # Skip records that errored during inference
    if record.get("error") is not None:
        return {**record,
                "normalised_response": None,
                "correct": None,
                "score_method": "skipped_inference_error"}

    # ── H5b — A/B choice ──────────────────────────────────────────────────
    if subcat == "H5b":
        norm = normalise_choice(response)
        correct = (norm == gt.upper()) if norm is not None else None
        return {**record,
                "normalised_response": norm,
                "correct": correct,
                "score_method": "choice_ab"}

    # ── H4 free-form counting ─────────────────────────────────────────────
    if subcat == "H4" and q_type == "free_form":
        norm    = normalise_integer(response)
        correct = (norm == gt) if norm is not None else None
        return {**record,
                "normalised_response": norm,
                "correct": correct,
                "score_method": "integer_match"}

    # ── H5d / H7c — open-ended refusal ───────────────────────────────────
    if subcat in ("H5d", "H7c"):
        absent_obj = meta.get("absent_object") or \
                     meta.get("false_presupposition_object") or ""
        norm    = normalise_denial(response, absent_obj)
        correct = (norm == "denial")
        return {**record,
                "normalised_response": norm,
                "correct": correct,
                "score_method": "keyword_denial"}

    # ── Default: binary yes/no ────────────────────────────────────────────
    norm    = normalise_yes_no(response)
    correct = (norm == gt) if norm is not None else None
    return {**record,
            "normalised_response": norm,
            "correct": correct,
            "score_method": "yes_no"}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_accuracy(records: list) -> dict:
    """
    Compute accuracy for a list of scored records.
    Only counts records where correct is not None.
    Returns {"accuracy": float, "correct": int, "total": int, "unscored": int}
    """
    scored   = [r for r in records if r["correct"] is not None]
    correct  = sum(1 for r in scored if r["correct"])
    unscored = len(records) - len(scored)
    total    = len(scored)
    accuracy = correct / total if total > 0 else None
    return {
        "accuracy":  round(accuracy, 4) if accuracy is not None else None,
        "correct":   correct,
        "total":     total,
        "unscored":  unscored,
    }


def build_summary(scored_records: list) -> dict:
    """
    Build accuracy breakdown by subcategory, category, difficulty,
    and question_type.
    """
    by_subcat    = defaultdict(list)
    by_category  = defaultdict(list)
    by_difficulty = defaultdict(list)
    by_qtype     = defaultdict(list)

    for r in scored_records:
        subcat = r.get("subcategory", "unknown")
        cat    = r.get("category",    "unknown")
        diff   = r.get("difficulty",  "unknown")
        qtype  = r.get("metadata", {}).get("question_type", "default")

        by_subcat[subcat].append(r)
        by_category[cat].append(r)
        by_difficulty[diff].append(r)
        by_qtype[qtype].append(r)

    summary = {
        "overall": compute_accuracy(scored_records),
        "by_subcategory": {
            k: compute_accuracy(v)
            for k, v in sorted(by_subcat.items())
        },
        "by_category": {
            k: compute_accuracy(v)
            for k, v in sorted(by_category.items())
        },
        "by_difficulty": {
            k: compute_accuracy(v)
            for k, v in sorted(by_difficulty.items())
        },
        "by_question_type": {
            k: compute_accuracy(v)
            for k, v in sorted(by_qtype.items())
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    log.info(f"Loading responses: {args.responses}")
    records = load_json(args.responses)
    log.info(f"  {len(records)} records loaded.")

    # Score each record
    scored = [score_record(r) for r in records]

    # Summary statistics
    summary = build_summary(scored)

    # Print to console
    log.info("\n── Scoring Summary ─────────────────────────────────")
    overall = summary["overall"]
    log.info(f"  Overall accuracy: {overall['accuracy']:.1%}  "
             f"({overall['correct']}/{overall['total']} scored, "
             f"{overall['unscored']} unscored)")

    log.info("\n  By subcategory:")
    for subcat, acc in summary["by_subcategory"].items():
        if acc["total"] == 0:
            continue
        log.info(f"    {subcat:6s}  {acc['accuracy']:.1%}  "
                 f"({acc['correct']}/{acc['total']})")

    log.info("\n  By difficulty:")
    for diff, acc in summary["by_difficulty"].items():
        if acc["total"] == 0:
            continue
        log.info(f"    {diff:8s}  {acc['accuracy']:.1%}  "
                 f"({acc['correct']}/{acc['total']})")

    # Save outputs
    save_json(scored,   args.scores)
    save_json(summary,  args.summary)
    log.info(f"\nSaved scores  → {args.scores}")
    log.info(f"Saved summary → {args.summary}")
    log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score LLaVA responses against benchmark ground truth"
    )
    parser.add_argument(
        "--responses",
        default="results/llava_raw_responses.json",
        help="Raw model responses from run_llava_inference.py",
    )
    parser.add_argument(
        "--scores",
        default="results/llava_scores.json",
        help="Output: per-record scores",
    )
    parser.add_argument(
        "--summary",
        default="results/llava_summary.json",
        help="Output: aggregated accuracy summary",
    )
    args = parser.parse_args()
    main(args)