"""
generate_dataset.py
-------------------
Generates the VLM hallucination benchmark dataset from MS-COCO val2017
and Visual Genome annotations.

Produces: data/processed/benchmark_v1.json

Usage:
    python scripts/generate_dataset.py \
        --coco_ann  annotations/instances_val2017.json \
        --vg_attrs  visual_genome/attributes.json \
        --vg_rels   visual_genome/relationships.json \
        --vg_imgs   image_data.json \
        --findings  results/exploration_findings.json \
        --output    data/processed/benchmark_v1.json \
        --seed      42

Categories generated:
    H1a  Object Existence — random sampling (absent/present pairs)
    H1b  Object Existence — popular-category absent
    H1c  Object Existence — adversarial co-occurrence absent
    H2a  Object Attribute — color
    H2b  Object Attribute — shape/size  (flagged for manual review)
    H3a  Relational — spatial (asymmetric predicates only; inverted = "no")
    H3b  Relational — action/interaction
    H4   Counting / Numerical
    H5a  Existence Negation — negative pronoun
    H5b  Existence Negation — implicit negation
    H5c  Existence Negation — contrastive pair
    H5d  Existence Negation — unsolvable probe
    H7c  Bias & Interference — text-to-image interference

Note on taxonomy alignment:
    H1b/H1c here follow POPE-style sampling nomenclature (popular/adversarial
    absent), which differs from taxonomy.md definitions (H1b = existent object
    manipulation, H1c = knowledge manipulation). These IDs will be reconciled
    in a future taxonomy update.

Note on --target_scenarios:
    --target_scenarios controls the number of source scenarios sampled per
    sub-category. Some generators emit multiple questions per scenario
    (H1a: 2, H3a: up to 2, H4: 3), so the final question count per
    sub-category will be a multiple of target_scenarios for those categories.
"""

import json
import random
import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_PER_SUBCAT = 400

# Person is too dominant as adversarial pair — skip as adversarial target
ADVERSARIAL_SKIP = {"person"}

# Color options for H2a distractor generation
COLORS = [
    "red", "blue", "green", "yellow", "orange",
    "purple", "black", "white", "brown", "pink", "grey"
]

# Size comparison words for H2b
SIZE_WORDS = ["larger", "smaller", "bigger", "taller", "shorter", "wider"]

# Asymmetric spatial predicates only — safe to invert subject/object → "no"
ASYMMETRIC_SPATIAL_PREDICATES = {
    "to the left of", "to the right of",
    "above", "below",
    "on top of", "under",
    "in front of", "behind",
    "inside", "outside",
    "on",
}

# All spatial predicates (positive-only H3a questions)
ALL_SPATIAL_PREDICATES = ASYMMETRIC_SPATIAL_PREDICATES | {
    "next to", "near", "beside", "adjacent to", "surrounding", "overlapping",
    "across from",
}

# Expanded action predicates for H3b
ACTION_PREDICATES = {
    "holding", "wearing", "carrying", "eating", "drinking",
    "riding", "driving", "pulling", "pushing", "throwing",
    "catching", "kicking", "hitting", "cutting",
    "sitting on", "standing on", "lying on", "kneeling on",
    "standing in", "sitting in", "leaning on", "leaning against",
    "resting on", "perched on",
    "looking at", "watching", "reading", "using",
    "playing with", "walking on", "running on", "jumping over",
    "flying over", "hanging from", "attached to", "connected to",
    "covered in", "mounted on", "growing on", "painted on",
    "reflected in", "parked on", "parked in", "tied to",
}

# Text-to-image interference templates for H7c.
# {ART} = indefinite article, {FALSE_OBJ} = object name.
# Denial is computed at generation time via no_existence_denial() so it
# correctly handles plural categories — not stored in this list.
INTERFERENCE_TEMPLATES = [
    "Since {ART} {FALSE_OBJ} is in this image, what color is it?",
    "Given that there is {ART} {FALSE_OBJ} here, where is it located?",
    "The {FALSE_OBJ} in this photo appears to be what color?",
    "Can you describe the {FALSE_OBJ} visible in this scene?",
    "What is the {FALSE_OBJ} doing in this image?",
]

# COCO-specific plural/display overrides: canonical name → (plural_form, is_plural)
# is_plural=True means the category is grammatically plural (uses "are/Are/there are")
COCO_DISPLAY = {
    "skis":      ("skis",           True),
    "scissors":  ("scissors",       True),
    "pants":     ("pairs of pants", True),
    "knife":     ("knives",         False),
    "person":    ("people",         True),
    "sheep":     ("sheep",          False),   # sheep is singular or plural
    "fish":      ("fish",           False),
    "aircraft":  ("aircraft",       False),
}

# Minimum character length for VG annotation tokens to be accepted
MIN_TOKEN_LEN   = 3
# Minimum ratio of alphabetic characters
MIN_ALPHA_RATIO = 0.8
# Minimum length specifically for needs-review object strings
MIN_REVIEW_TOKEN_LEN = 4


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------

def indefinite_article(word: str) -> str:
    """Return 'an' before vowel sounds, 'a' otherwise."""
    return "an" if word and word[0].lower() in "aeiou" else "a"


def is_plural_token(name: str) -> bool:
    """
    Heuristic plurality check for arbitrary object strings including VG tokens.
    Returns True if the name is likely grammatically plural.

    Strategy:
      1. COCO_DISPLAY overrides take precedence (authoritative).
      2. Known-singular exceptions that end in 's' are explicitly excluded.
      3. Multi-word tokens: check the last word only.
      4. Ends-in-s/es/ies heuristic covers common VG nouns like
         'rocks', 'trees', 'glasses', 'branches', 'bushes'.
    """
    if name in COCO_DISPLAY:
        return COCO_DISPLAY[name][1]

    # Known singular words ending in 's' that would otherwise be misclassified
    SINGULAR_EXCEPTIONS = {
        "grass", "glass", "class", "mass", "pass", "moss", "loss",
        "bus", "gas", "plus", "minus", "virus", "status", "campus",
        "bonus", "focus", "process", "dress", "stress", "press",
        "address", "mattress", "success", "access", "compass",
        "canvas", "atlas", "series", "species", "news",
    }
    last_word = name.split()[-1] if " " in name else name
    if last_word in SINGULAR_EXCEPTIONS:
        return False
    # Double-s ending (e.g. 'grass', 'moss') → singular
    if last_word.endswith("ss"):
        return False
    # Standard English plural suffixes
    if last_word.endswith("ies") or last_word.endswith("es") or last_word.endswith("s"):
        return True
    return False


def display_plural(name: str) -> str:
    """Return the display plural form for a category name."""
    if name in COCO_DISPLAY:
        return COCO_DISPLAY[name][0]
    if name.endswith("y") and len(name) > 1 and name[-2] not in "aeiou":
        return name[:-1] + "ies"
    if name.endswith("s") or name.endswith("x") or name.endswith("z"):
        return name + "es"
    return name + "s"


def existence_question(obj: str, present: bool) -> tuple[str, str]:
    """
    Return (question, ground_truth) with correct number agreement.

    Examples:
        "Is there a dog in this image?"   → "yes"/"no"
        "Are there any skis in this image?" → "yes"/"no"
    """
    gt = "yes" if present else "no"
    if is_plural_token(obj):
        plural = display_plural(obj) if obj not in COCO_DISPLAY else COCO_DISPLAY[obj][0]
        verb   = "Are"
        q = f"{verb} there any {plural} in this image?"
    else:
        art = indefinite_article(obj)
        q   = f"Is there {art} {obj} in this image?"
    return q, gt


def no_existence_denial(obj: str) -> str:
    """Return a grammatically correct denial string for a given object."""
    if is_plural_token(obj):
        plural = display_plural(obj) if obj not in COCO_DISPLAY else COCO_DISPLAY[obj][0]
        return f"there are no {plural} in this image"
    return f"there is no {obj} in this image"


def be_verb(obj: str, capitalize: bool = True) -> str:
    """Return 'Is'/'Are' (or 'is'/'are') based on plurality of obj."""
    v = "Are" if is_plural_token(obj) else "Is"
    return v if capitalize else v.lower()


def neg_pronoun_question(obj: str, template_idx: int) -> str:
    """
    Return a negative-pronoun question with correct number agreement.
    Uses template_idx to select from singular or plural template variants.
    """
    if is_plural_token(obj):
        plural = display_plural(obj) if obj not in COCO_DISPLAY else COCO_DISPLAY[obj][0]
        templates = [
            f"Are there no {plural} in this image?",
            f"There are no {plural} in this image, correct?",
            f"Does this image contain no {plural}?",
            f"Is it true that no {plural} appear in this image?",
        ]
    else:
        templates = [
            f"Is there no {obj} in this image?",
            f"There is no {obj} in this image, correct?",
            f"Does this image contain no {obj}?",
            f"Is it true that no {obj} appears in this image?",
        ]
    return templates[template_idx % len(templates)]


# ---------------------------------------------------------------------------
# Sanitization helpers
# ---------------------------------------------------------------------------

def sanitize_vg_token(token: str) -> str | None:
    """
    Return the cleaned token if it passes quality checks, else None.
    Filters out:
      - Very short strings (< MIN_TOKEN_LEN chars)
      - Non-alphabetic junk (alpha ratio < MIN_ALPHA_RATIO)
      - Single-word tokens shorter than 4 chars that are likely truncations
        (e.g. 'gree', 'towl') — multi-word tokens like 'bus stop' are exempt
    """
    token = token.strip().lower()
    if len(token) < MIN_TOKEN_LEN:
        return None
    alpha_chars = sum(1 for c in token if c.isalpha())
    if alpha_chars / max(len(token), 1) < MIN_ALPHA_RATIO:
        return None
    # Single-word short tokens are likely VG truncation artifacts
    if " " not in token and len(token) < 4:
        return None
    return token


# Strings that appear in VG annotations as annotation markers, quantifiers,
# or sentence fragments rather than object names — reject from needs-review set
REVIEW_ARTIFACT_STOPLIST = {
    # Annotation markers
    "removed", "deleted", "unknown", "none", "null", "na", "n/a",
    "unlabeled", "other", "misc", "item", "object", "thing", "stuff",
    # Quantifiers / vague descriptors that are not object names
    "several", "many", "some", "few", "large", "small", "big", "tall",
    "various", "multiple", "different", "certain", "another",
    # Sentence fragments / bad annotations
    "bad sentence", "no label", "not sure", "hard to tell",
    "someon", "someon", "someon e",
}


def sanitize_review_token(token: str) -> str | None:
    """
    Stricter sanitization for needs-review H2b object strings.
    Rejects:
      - Tokens shorter than MIN_REVIEW_TOKEN_LEN characters
      - Tokens containing non-alpha/space characters
      - Known annotation artifacts, quantifiers, and sentence fragments
        from REVIEW_ARTIFACT_STOPLIST
    """
    token = token.strip().lower()
    if len(token) < MIN_REVIEW_TOKEN_LEN:
        return None
    if not all(c.isalpha() or c == " " for c in token):
        return None
    if token in REVIEW_ARTIFACT_STOPLIST:
        return None
    # Reject if any word in a multi-word token is a known artifact
    if any(word in REVIEW_ARTIFACT_STOPLIST for word in token.split()):
        return None
    return token


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def save_json(data: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved {len(data)} records → {path}")


def make_record(
    image_id:     int,
    category:     str,
    subcategory:  str,
    question:     str,
    ground_truth: str,
    difficulty:   str,
    metadata:     dict = None,
) -> dict:
    return {
        "image_id":     image_id,
        "category":     category,
        "subcategory":  subcategory,
        "question":     question,
        "ground_truth": ground_truth,
        "difficulty":   difficulty,
        "metadata":     metadata or {},
    }


def difficulty_from_count(count: int, low: int = 100, high: int = 500) -> str:
    if count >= high:
        return "easy"
    elif count >= low:
        return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# COCO annotation cache — built once in main(), passed to all generators
# ---------------------------------------------------------------------------

def build_annotation_cache(coco: COCO) -> tuple:
    """
    Returns:
        img_to_cats        dict[image_id → set[category_id]]
        img_to_cat_counts  dict[image_id → dict[category_id → int]]

    Iterates all COCO annotations exactly once.
    """
    img_to_cats       = defaultdict(set)
    img_to_cat_counts = defaultdict(lambda: defaultdict(int))
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_to_cats[ann["image_id"]].add(ann["category_id"])
        img_to_cat_counts[ann["image_id"]][ann["category_id"]] += 1
    return img_to_cats, img_to_cat_counts


# ---------------------------------------------------------------------------
# Co-occurrence matrix
# ---------------------------------------------------------------------------

def build_cooccurrence(coco: COCO, img_to_cats: dict) -> tuple:
    log.info("Building co-occurrence matrix...")
    cats      = coco.loadCats(coco.getCatIds())
    cat_ids   = [c["id"] for c in cats]
    cat_names = {c["id"]: c["name"] for c in cats}
    id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    n         = len(cat_ids)

    cooccurrence = np.zeros((n, n), dtype=np.int32)
    for present in img_to_cats.values():
        pl = list(present)
        for i in pl:
            for j in pl:
                if i != j:
                    cooccurrence[id_to_idx[i]][id_to_idx[j]] += 1

    log.info("Co-occurrence matrix built.")
    return cooccurrence, cat_ids, id_to_idx, cat_names


def get_adversarial_absent(
    present_ids:  set,
    cooccurrence: np.ndarray,
    cat_ids:      list,
    id_to_idx:    dict,
    cat_names:    dict,
    skip_names:   set = ADVERSARIAL_SKIP,
    topk:         int = 1,
) -> list:
    scores = np.zeros(len(cat_ids), dtype=np.int32)
    for cid in present_ids:
        scores += cooccurrence[id_to_idx[cid]]
    for cid in present_ids:
        scores[id_to_idx[cid]] = 0
    for cid, name in cat_names.items():
        if name in skip_names:
            scores[id_to_idx[cid]] = 0
    top_indices = np.argsort(scores)[::-1][:topk]
    return [cat_names[cat_ids[i]] for i in top_indices if scores[i] > 0]


# ---------------------------------------------------------------------------
# H1 — Object Existence
# ---------------------------------------------------------------------------

def generate_h1(
    coco:            COCO,
    img_to_cats:     dict,
    cooccurrence:    np.ndarray,
    cat_ids:         list,
    id_to_idx:       dict,
    cat_names:       dict,
    instance_counts: dict,
    target:          int = TARGET_PER_SUBCAT,
    rng:             random.Random = None,
) -> list:
    rng = rng or random.Random(42)
    records = []

    # Sorted for deterministic sampling regardless of Python hash randomization
    all_cat_names = sorted(cat_names.values())
    popular_cats  = sorted(instance_counts, key=instance_counts.get, reverse=True)[:20]

    images = sorted(img_to_cats.keys())
    rng.shuffle(images)
    h1a = h1b = h1c = 0

    for img_id in images:
        present_ids   = img_to_cats[img_id]
        present_names = sorted(cat_names[c] for c in present_ids)
        present_set   = set(present_names)
        absent_names  = sorted(set(all_cat_names) - present_set)
        if not absent_names:
            continue

        # H1a — random absent + present pair
        if h1a < target:
            absent  = rng.choice(absent_names)
            present = rng.choice(present_names)
            q_abs, gt_abs = existence_question(absent,  present=False)
            q_pre, gt_pre = existence_question(present, present=True)
            records.append(make_record(
                img_id, "H1", "H1a", q_abs, gt_abs, "easy",
                {"absent_object": absent, "sampling": "random"},
            ))
            records.append(make_record(
                img_id, "H1", "H1a", q_pre, gt_pre, "easy",
                {"present_object": present, "sampling": "random"},
            ))
            h1a += 1

        # H1b — popular absent
        if h1b < target:
            popular_absent = [c for c in popular_cats if c not in present_set]
            if popular_absent:
                absent = rng.choice(popular_absent[:5])
                q, gt  = existence_question(absent, present=False)
                records.append(make_record(
                    img_id, "H1", "H1b", q, gt, "medium",
                    {"absent_object": absent, "sampling": "popular"},
                ))
                h1b += 1

        # H1c — adversarial co-occurrence
        if h1c < target:
            adversarial = get_adversarial_absent(
                present_ids, cooccurrence, cat_ids, id_to_idx, cat_names
            )
            if adversarial:
                absent = adversarial[0]
                q, gt  = existence_question(absent, present=False)
                records.append(make_record(
                    img_id, "H1", "H1c", q, gt, "hard",
                    {
                        "absent_object":   absent,
                        "sampling":        "adversarial",
                        "present_objects": present_names,
                    },
                ))
                h1c += 1

        if h1a >= target and h1b >= target and h1c >= target:
            break

    log.info(f"H1: {h1a} random (H1a), {h1b} popular (H1b), {h1c} adversarial (H1c)")
    return records


# ---------------------------------------------------------------------------
# H2 — Object Attribute
# ---------------------------------------------------------------------------

def generate_h2(
    vg_attributes: list,
    vg_coco_map:   dict,
    coco_val_ids:  set,
    target:        int = TARGET_PER_SUBCAT,
    rng:           random.Random = None,
) -> list:
    rng = rng or random.Random(42)
    records = []
    h2a = h2b = 0

    for img_data in vg_attributes:
        vg_id   = img_data.get("image_id") or img_data.get("id")
        coco_id = vg_coco_map.get(vg_id)
        if coco_id not in coco_val_ids:
            continue

        objs = img_data.get("attributes", [])

        for obj in objs:
            obj_name = sanitize_vg_token(obj.get("names", [""])[0])
            if not obj_name:
                continue

            attrs = [a.lower().strip() for a in obj.get("attributes", [])]
            bv    = be_verb(obj_name)

            # H2a — color
            if h2a < target:
                true_colors = [a for a in attrs if a in COLORS]
                if true_colors:
                    true_color  = true_colors[0]
                    wrong_color = rng.choice([c for c in COLORS if c != true_color])
                    records.append(make_record(
                        coco_id, "H2", "H2a",
                        f"{bv} the {obj_name} {true_color}?",
                        "yes", "medium",
                        {"object": obj_name, "true_color": true_color},
                    ))
                    records.append(make_record(
                        coco_id, "H2", "H2a",
                        f"{bv} the {obj_name} {wrong_color}?",
                        "no", "medium",
                        {"object": obj_name, "true_color": true_color,
                         "wrong_color": wrong_color},
                    ))
                    h2a += 1

            # H2b — size (flagged for review)
            if h2b < target:
                size_attrs = [a for a in attrs if any(
                    s in a for s in ["large", "small", "big", "tall", "short", "wide"]
                )]
                if size_attrs and len(objs) > 1:
                    other_names = sorted(
                        norm for o in objs
                        for norm in [sanitize_vg_token(o.get("names", [""])[0])]
                        if norm and norm != obj_name
                    )
                    if other_names:
                        other_obj = rng.choice(other_names)
                        size_word = rng.choice(SIZE_WORDS)
                        records.append(make_record(
                            coco_id, "H2", "H2b",
                            f"{bv} the {obj_name} {size_word} than the {other_obj}?",
                            "ambiguous", "hard",
                            {
                                "object_a":           obj_name,
                                "object_b":           other_obj,
                                "size_word":          size_word,
                                "needs_verification": True,
                            },
                        ))
                        h2b += 1

            if h2a >= target and h2b >= target:
                break

        if h2a >= target and h2b >= target:
            break

    log.info(f"H2: {h2a} color (H2a), {h2b} size (H2b)")
    return records


# ---------------------------------------------------------------------------
# H3 — Relational
# ---------------------------------------------------------------------------

def generate_h3(
    vg_relationships: list,
    vg_coco_map:      dict,
    coco_val_ids:     set,
    target:           int = TARGET_PER_SUBCAT,
    rng:              random.Random = None,
) -> list:
    rng = rng or random.Random(42)
    records = []
    h3a = h3b = 0

    for img_data in vg_relationships:
        vg_id   = img_data.get("image_id") or img_data.get("id")
        coco_id = vg_coco_map.get(vg_id)
        if coco_id not in coco_val_ids:
            continue

        for rel in img_data.get("relationships", []):
            subj = sanitize_vg_token(rel.get("subject", {}).get("names", [""])[0])
            obj  = sanitize_vg_token(rel.get("object",  {}).get("names", [""])[0])
            pred = rel.get("predicate", "").lower().strip()
            if not subj or not obj or not pred:
                continue

            # H3a — spatial
            if h3a < target and pred in ALL_SPATIAL_PREDICATES:
                records.append(make_record(
                    coco_id, "H3", "H3a",
                    f"Is the {subj} {pred} the {obj}?",
                    "yes", "medium",
                    {"subject": subj, "predicate": pred, "object": obj},
                ))
                if pred in ASYMMETRIC_SPATIAL_PREDICATES:
                    records.append(make_record(
                        coco_id, "H3", "H3a",
                        f"Is the {obj} {pred} the {subj}?",
                        "no", "hard",
                        {"subject": obj, "predicate": pred, "object": subj,
                         "inverted": True},
                    ))
                h3a += 1

            # H3b — action/interaction
            if h3b < target and pred in ACTION_PREDICATES:
                records.append(make_record(
                    coco_id, "H3", "H3b",
                    f"Is the {subj} {pred} the {obj}?",
                    "yes", "medium",
                    {"subject": subj, "predicate": pred, "object": obj},
                ))
                h3b += 1

            if h3a >= target and h3b >= target:
                break

        if h3a >= target and h3b >= target:
            break

    log.info(f"H3: {h3a} spatial (H3a), {h3b} action (H3b)")
    return records


# ---------------------------------------------------------------------------
# H4 — Counting
# ---------------------------------------------------------------------------

def generate_h4(
    coco:              COCO,
    img_to_cat_counts: dict,
    instance_counts:   dict,
    target:            int = TARGET_PER_SUBCAT,
    rng:               random.Random = None,
) -> list:
    rng = rng or random.Random(42)
    records = []

    cat_names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    images    = sorted(img_to_cat_counts.keys())
    rng.shuffle(images)
    count = 0

    for img_id in images:
        if count >= target:
            break
        valid = {cid: n for cid, n in img_to_cat_counts[img_id].items()
                 if 2 <= n <= 8}
        if not valid:
            continue

        cid, true_count = rng.choice(sorted(valid.items()))
        obj_name    = cat_names[cid]
        obj_plural  = display_plural(obj_name)
        wrong_count = max(1, true_count + rng.choice([-1, 1, 2]))
        diff        = difficulty_from_count(instance_counts.get(obj_name, 0))

        records.append(make_record(
            img_id, "H4", "H4",
            f"How many {obj_plural} are in this image?",
            str(true_count), diff,
            {"object": obj_name, "true_count": true_count,
             "question_type": "free_form"},
        ))
        records.append(make_record(
            img_id, "H4", "H4",
            f"Are there exactly {wrong_count} {obj_plural} in this image?",
            "no", "hard",
            {"object": obj_name, "true_count": true_count,
             "wrong_count": wrong_count, "question_type": "binary_wrong"},
        ))
        records.append(make_record(
            img_id, "H4", "H4",
            f"Are there exactly {true_count} {obj_plural} in this image?",
            "yes", "medium",
            {"object": obj_name, "true_count": true_count,
             "question_type": "binary_correct"},
        ))
        count += 1

    log.info(f"H4: {count} counting scenarios → "
             f"{len([r for r in records if r['category'] == 'H4'])} questions")
    return records


# ---------------------------------------------------------------------------
# H5 — Existence Negation (H5a, H5b, H5c, H5d)
# ---------------------------------------------------------------------------

def generate_h5(
    coco:        COCO,
    img_to_cats: dict,
    cat_names:   dict,
    target:      int = TARGET_PER_SUBCAT,
    rng:         random.Random = None,
) -> list:
    rng = rng or random.Random(42)
    records = []

    all_cat_names = sorted(cat_names.values())
    images        = sorted(img_to_cats.keys())
    rng.shuffle(images)

    # Translate image_id → set of category names (used for fast lookup in H5c)
    img_to_cat_names = {
        img_id: set(cat_names[c] for c in cids)
        for img_id, cids in img_to_cats.items()
    }

    h5a = h5b = h5c = h5d = 0
    template_counter = 0   # cycles through neg-pronoun templates deterministically

    for img_id in images:
        present_names = sorted(img_to_cat_names[img_id])
        present_set   = set(present_names)
        absent_names  = sorted(set(all_cat_names) - present_set)
        if not absent_names:
            continue

        # H5a — negative pronoun framing (number-agreement-aware)
        if h5a < target:
            absent   = rng.choice(absent_names)
            question = neg_pronoun_question(absent, template_counter)
            template_counter += 1
            records.append(make_record(
                img_id, "H5", "H5a",
                question,
                "yes",   # correct: yes, there IS no X
                "hard",
                {"absent_object": absent},
            ))
            h5a += 1

        # H5b — implicit negation: which of A/B is NOT present
        if h5b < target and absent_names and present_names:
            absent  = rng.choice(absent_names)
            present = rng.choice(present_names)
            opts    = [absent, present]
            rng.shuffle(opts)
            opt_a, opt_b = opts[0], opts[1]
            correct = "A" if opt_a == absent else "B"
            records.append(make_record(
                img_id, "H5", "H5b",
                f"Which of these is NOT in the image: (A) {opt_a} or (B) {opt_b}?",
                correct, "medium",
                {"option_a": opt_a, "option_b": opt_b, "absent_object": absent},
            ))
            h5b += 1

        # H5c — contrastive pair
        # Reference image (img_id) HAS shared_obj.
        # Target image (img_b)  does NOT have shared_obj.
        # Image ID kept in metadata only — not in question text.
        if h5c < target and present_names:
            shared_obj = rng.choice(present_names)
            absent_candidates = sorted(
                i for i in images
                if i != img_id
                and shared_obj not in img_to_cat_names.get(i, set())
            )
            if absent_candidates:
                img_b = rng.choice(absent_candidates[:50])
                # Rephrase as contrastive: "...like the reference image?"
                if is_plural_token(shared_obj):
                    plural = display_plural(shared_obj) \
                        if shared_obj not in COCO_DISPLAY \
                        else COCO_DISPLAY[shared_obj][0]
                    question = (f"Are there any {plural} in this image, "
                                f"like the reference image?")
                else:
                    art      = indefinite_article(shared_obj)
                    question = (f"Does this image contain {art} {shared_obj}, "
                                f"like the reference image?")
                records.append(make_record(
                    img_b, "H5", "H5c",
                    question, "no", "hard",
                    {
                        "reference_image_id": img_id,
                        "target_object":      shared_obj,
                        "reference_has_obj":  True,
                        "target_has_obj":     False,
                    },
                ))
                h5c += 1

        # H5d — unsolvable probe (presupposition denial)
        if h5d < target and absent_names:
            absent = rng.choice(absent_names)
            denial = no_existence_denial(absent)
            if is_plural_token(absent):
                plural = display_plural(absent) \
                    if absent not in COCO_DISPLAY else COCO_DISPLAY[absent][0]
                probe_templates = [
                    f"What color are the {plural} in this image?",
                    f"Where are the {plural} located in this image?",
                    f"What are the {plural} doing in this image?",
                    f"How many {plural} are there in this image?",
                    f"Describe the {plural} in this image.",
                ]
            else:
                obj_plural = display_plural(absent)
                probe_templates = [
                    f"What color is the {absent} in this image?",
                    f"Where is the {absent} located in this image?",
                    f"What is the {absent} doing in this image?",
                    f"How many {obj_plural} are there in this image?",
                    f"Describe the {absent} in this image.",
                ]
            records.append(make_record(
                img_id, "H5", "H5d",
                rng.choice(probe_templates),
                denial, "hard",
                {"absent_object": absent, "expected_refusal": True},
            ))
            h5d += 1

        if h5a >= target and h5b >= target and h5c >= target and h5d >= target:
            break

    log.info(f"H5: {h5a} neg-pronoun (H5a), {h5b} implicit (H5b), "
             f"{h5c} contrastive (H5c), {h5d} unsolvable (H5d)")
    return records


# ---------------------------------------------------------------------------
# H7c — Text-to-Image Interference
# ---------------------------------------------------------------------------

def generate_h7c(
    coco:            COCO,
    img_to_cats:     dict,
    instance_counts: dict,
    target:          int = TARGET_PER_SUBCAT,
    rng:             random.Random = None,
) -> list:
    """
    Questions embed a false presupposition about an object NOT in the image.
    False objects are sampled from the top-20 most frequent absent COCO
    categories, making the interference semantically plausible.
    """
    rng = rng or random.Random(42)
    records = []

    cat_names     = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    all_cat_names = sorted(cat_names.values())
    popular_order = sorted(all_cat_names,
                           key=lambda n: instance_counts.get(n, 0), reverse=True)

    images = sorted(img_to_cats.keys())
    rng.shuffle(images)
    count = 0

    for img_id in images:
        if count >= target:
            break
        present_names = set(cat_names[c] for c in img_to_cats[img_id])
        absent_names  = [n for n in popular_order if n not in present_names]
        if not absent_names:
            continue

        false_obj = rng.choice(absent_names[:20])
        art       = indefinite_article(false_obj)
        denial    = no_existence_denial(false_obj)

        template = rng.choice(INTERFERENCE_TEMPLATES)
        question = template.format(ART=art, FALSE_OBJ=false_obj)

        records.append(make_record(
            img_id, "H7", "H7c",
            question, denial, "hard",
            {
                "false_presupposition_object": false_obj,
                "present_objects":             sorted(present_names),
                "interference_type":           "text_to_image",
                "expected_refusal":            True,
            },
        ))
        count += 1

    log.info(f"H7c: {count} text-to-image interference questions")
    return records


# ---------------------------------------------------------------------------
# Deduplication and quality filtering
# ---------------------------------------------------------------------------

def deduplicate(records: list) -> list:
    seen  = set()
    clean = []
    for r in records:
        key = (r["image_id"], r["question"])
        if key not in seen:
            seen.add(key)
            clean.append(r)
    removed = len(records) - len(clean)
    if removed:
        log.info(f"Deduplication removed {removed} duplicate records.")
    return clean


def filter_ambiguous(records: list) -> tuple:
    clean  = [r for r in records if not r["metadata"].get("needs_verification")]
    review = [r for r in records if r["metadata"].get("needs_verification")]
    if review:
        log.info(f"{len(review)} records flagged for manual verification (H2b).")
    return clean, review


def sanitize_needs_review(records: list) -> list:
    """
    Remove H2b records where either object string fails the stricter
    review-token quality check (too short, non-alpha, VG artifacts).
    """
    clean = []
    for r in records:
        obj_a = r["metadata"].get("object_a", "")
        obj_b = r["metadata"].get("object_b", "")
        if sanitize_review_token(obj_a) and sanitize_review_token(obj_b):
            clean.append(r)
        else:
            log.debug(f"Filtered needs_review artifact: '{obj_a}' / '{obj_b}'")
    removed = len(records) - len(clean)
    if removed:
        log.info(f"Sanitized {removed} artifact records from needs_review set.")
    return clean


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    rng = random.Random(args.seed)

    # Load COCO
    log.info("Loading COCO annotations...")
    coco = COCO(args.coco_ann)

    cat_names       = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    coco_val_ids    = set(coco.imgs.keys())
    instance_counts = {
        cat["name"]: len(coco.getAnnIds(catIds=cat["id"]))
        for cat in coco.loadCats(coco.getCatIds())
    }

    # Load findings
    findings = load_json(args.findings)
    log.info(f"Loaded findings: {findings['total_coco_val_images']} COCO images, "
             f"{findings['vg_coco_overlap_count']} VG/COCO overlap")

    # Build annotation cache once — shared across all generators
    log.info("Building annotation cache...")
    img_to_cats, img_to_cat_counts = build_annotation_cache(coco)

    # Build co-occurrence matrix
    cooccurrence, cat_ids, id_to_idx, _ = build_cooccurrence(coco, img_to_cats)

    # Load Visual Genome
    log.info("Loading Visual Genome data...")
    vg_images = load_json(args.vg_imgs)
    vg_coco_map = {
        img["image_id"]: img["coco_id"]
        for img in vg_images
        if img.get("coco_id") is not None
    }
    vg_attributes    = load_json(args.vg_attrs)
    vg_relationships = load_json(args.vg_rels)

    # Generate all categories
    all_records = []

    log.info("--- H1: Object Existence ---")
    all_records += generate_h1(
        coco, img_to_cats, cooccurrence, cat_ids, id_to_idx, cat_names,
        instance_counts, target=args.target_scenarios, rng=rng,
    )

    log.info("--- H2: Object Attribute ---")
    all_records += generate_h2(
        vg_attributes, vg_coco_map, coco_val_ids,
        target=args.target_scenarios, rng=rng,
    )

    log.info("--- H3: Relational ---")
    all_records += generate_h3(
        vg_relationships, vg_coco_map, coco_val_ids,
        target=args.target_scenarios, rng=rng,
    )

    log.info("--- H4: Counting ---")
    all_records += generate_h4(
        coco, img_to_cat_counts, instance_counts,
        target=args.target_scenarios, rng=rng,
    )

    log.info("--- H5: Existence Negation ---")
    all_records += generate_h5(
        coco, img_to_cats, cat_names,
        target=args.target_scenarios, rng=rng,
    )

    log.info("--- H7c: Text-to-Image Interference ---")
    all_records += generate_h7c(
        coco, img_to_cats, instance_counts,
        target=args.target_scenarios, rng=rng,
    )

    # Deduplication + quality filter
    all_records         = deduplicate(all_records)
    clean, needs_review = filter_ambiguous(all_records)
    needs_review        = sanitize_needs_review(needs_review)

    # Statistics
    log.info("\n── Dataset Statistics ──────────────────────────────")
    subcat_counts = defaultdict(int)
    for r in clean:
        subcat_counts[r["subcategory"]] += 1
    for subcat, n in sorted(subcat_counts.items()):
        log.info(f"  {subcat:6s}  {n:5d} questions")
    log.info(f"  {'TOTAL':6s}  {len(clean):5d} questions")
    log.info(f"  {'REVIEW':6s}  {len(needs_review):5d} flagged for manual verification")

    # Save
    save_json(clean,        args.output)
    save_json(needs_review, args.output.replace(".json", "_needs_review.json"))
    log.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate VLM hallucination benchmark dataset"
    )
    parser.add_argument("--coco_ann",  default="annotations/instances_val2017.json")
    parser.add_argument("--vg_attrs",  default="visual_genome/attributes.json")
    parser.add_argument("--vg_rels",   default="visual_genome/relationships.json")
    parser.add_argument("--vg_imgs",   default="image_data.json")
    parser.add_argument("--findings",  default="results/exploration_findings.json")
    parser.add_argument("--output",    default="data/processed/benchmark_v1.json")
    parser.add_argument(
        "--target_scenarios", type=int, default=TARGET_PER_SUBCAT,
        help=(
            "Number of source scenarios to sample per sub-category. "
            "Some generators emit multiple questions per scenario "
            "(H1a: 2, H3a: up to 2, H4: 3), so final question counts "
            "will exceed this value for those sub-categories."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
