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
    H1a  Object Existence — random absent
    H1b  Object Existence — popular absent
    H1c  Object Existence — adversarial / co-occurrence absent
    H2a  Object Attribute — color
    H2b  Object Attribute — shape/size
    H3a  Relational — spatial
    H3b  Relational — action/interaction
    H4   Counting
    H5a  Existence Negation — negative pronoun
    H5b  Existence Negation — implicit negation
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

# Minimum images an H2/H3 category needs to be included
MIN_VG_INSTANCES = 50

# Target questions per sub-category (will be capped at available instances)
TARGET_PER_SUBCAT = 400

# Person is too dominant as adversarial pair — always skip as adversarial target
ADVERSARIAL_SKIP = {"person"}

# Spatial relation templates
SPATIAL_TEMPLATES = [
    ("Is the {A} to the left of the {B}?",   "left_of",    "right_of"),
    ("Is the {A} to the right of the {B}?",  "right_of",   "left_of"),
    ("Is the {A} above the {B}?",             "above",      "below"),
    ("Is the {A} below the {B}?",             "below",      "above"),
    ("Is the {A} in front of the {B}?",       "in front of","behind"),
    ("Is the {A} behind the {B}?",            "behind",     "in front of"),
]

# Color options for H2a distractor generation
COLORS = [
    "red", "blue", "green", "yellow", "orange",
    "purple", "black", "white", "brown", "pink", "grey"
]

# Size comparison words for H2b
SIZE_WORDS = ["larger", "smaller", "bigger", "taller", "shorter", "wider"]

# Negative pronoun templates for H5a
NEGATIVE_PRONOUN_TEMPLATES = [
    "Is there no {OBJ} in this image?",
    "There is no {OBJ} in this image, correct?",
    "Does this image contain no {OBJ}?",
    "None of the objects visible are {OBJ}s, right?",
    "Is it true that no {OBJ} appears in this image?",
]

# Implicit negation templates for H5b
IMPLICIT_NEGATION_TEMPLATES = [
    "What object is missing from this otherwise complete {SCENE} scene?",
    "Which of these is NOT in the image: {OBJ_A} or {OBJ_B}?",
    "Is anything from a typical {SCENE} setting absent in this image?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict | list:
    with open(path) as f:
        return json.load(f)


def save_json(data: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved {len(data)} records → {path}")


def make_record(
    image_id: int,
    category: str,
    subcategory: str,
    question: str,
    ground_truth: str,
    difficulty: str,
    metadata: dict = None,
) -> dict:
    return {
        "image_id":    image_id,
        "category":    category,
        "subcategory": subcategory,
        "question":    question,
        "ground_truth": ground_truth,
        "difficulty":  difficulty,
        "metadata":    metadata or {},
    }


def difficulty_from_count(count: int, low=100, high=500) -> str:
    """Assign difficulty based on instance frequency."""
    if count >= high:
        return "easy"
    elif count >= low:
        return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Step 1 — Build co-occurrence matrix from COCO
# ---------------------------------------------------------------------------

def build_cooccurrence(coco: COCO) -> tuple[np.ndarray, list, dict, dict]:
    """
    Returns:
        cooccurrence  — NxN numpy array of co-occurrence counts
        cat_ids       — list of category IDs (index → id)
        id_to_idx     — dict: category id → matrix index
        cat_names     — dict: category id → name
    """
    log.info("Building co-occurrence matrix...")

    cats     = coco.loadCats(coco.getCatIds())
    cat_ids  = [c["id"] for c in cats]
    cat_names = {c["id"]: c["name"] for c in cats}
    id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    n = len(cat_ids)

    img_to_cats = defaultdict(set)
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_to_cats[ann["image_id"]].add(ann["category_id"])

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
    present_ids: set,
    cooccurrence: np.ndarray,
    cat_ids: list,
    id_to_idx: dict,
    cat_names: dict,
    skip_names: set = ADVERSARIAL_SKIP,
    topk: int = 1,
) -> list[str]:
    """
    For the set of present category IDs, find the absent category that
    most strongly co-occurs — skipping names in `skip_names`.
    Returns up to `topk` category names.
    """
    scores = np.zeros(len(cat_ids), dtype=np.int32)
    for cid in present_ids:
        scores += cooccurrence[id_to_idx[cid]]

    # Zero out present objects and skipped names
    for cid in present_ids:
        scores[id_to_idx[cid]] = 0
    for cid, name in cat_names.items():
        if name in skip_names:
            scores[id_to_idx[cid]] = 0

    top_indices = np.argsort(scores)[::-1][:topk]
    return [cat_names[cat_ids[i]] for i in top_indices if scores[i] > 0]


# ---------------------------------------------------------------------------
# Step 2 — H1: Object Existence
# ---------------------------------------------------------------------------

def generate_h1(
    coco: COCO,
    cooccurrence: np.ndarray,
    cat_ids: list,
    id_to_idx: dict,
    cat_names: dict,
    instance_counts: dict,
    target: int = TARGET_PER_SUBCAT,
    rng: random.Random = None,
) -> list[dict]:
    """Generate H1a (random), H1b (popular), H1c (adversarial) questions."""

    rng = rng or random.Random(42)
    records = []

    # Group images by present categories
    img_to_cats = defaultdict(set)
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_to_cats[ann["image_id"]].add(ann["category_id"])

    all_cat_names = list(cat_names.values())
    popular_cats  = sorted(instance_counts, key=instance_counts.get, reverse=True)[:20]

    images = list(img_to_cats.keys())
    rng.shuffle(images)

    h1a_count = h1b_count = h1c_count = 0

    for img_id in images:
        present_ids   = img_to_cats[img_id]
        present_names = {cat_names[c] for c in present_ids}
        absent_names  = set(all_cat_names) - present_names

        if not absent_names:
            continue

        # H1a — random absent
        if h1a_count < target:
            absent = rng.choice(list(absent_names))
            records.append(make_record(
                image_id    = img_id,
                category    = "H1",
                subcategory = "H1a",
                question    = f"Is there a {absent} in this image?",
                ground_truth= "no",
                difficulty  = "easy",
                metadata    = {"absent_object": absent, "sampling": "random"},
            ))
            # Also generate a positive (present object) question for balance
            present_obj = rng.choice(list(present_names))
            records.append(make_record(
                image_id    = img_id,
                category    = "H1",
                subcategory = "H1a",
                question    = f"Is there a {present_obj} in this image?",
                ground_truth= "yes",
                difficulty  = "easy",
                metadata    = {"present_object": present_obj, "sampling": "random"},
            ))
            h1a_count += 1

        # H1b — popular absent
        if h1b_count < target:
            popular_absent = [c for c in popular_cats if c not in present_names]
            if popular_absent:
                absent = rng.choice(popular_absent[:5])
                records.append(make_record(
                    image_id    = img_id,
                    category    = "H1",
                    subcategory = "H1b",
                    question    = f"Is there a {absent} in this image?",
                    ground_truth= "no",
                    difficulty  = "medium",
                    metadata    = {"absent_object": absent, "sampling": "popular"},
                ))
                h1b_count += 1

        # H1c — adversarial co-occurrence
        if h1c_count < target:
            adversarial = get_adversarial_absent(
                present_ids, cooccurrence, cat_ids, id_to_idx, cat_names
            )
            if adversarial:
                absent = adversarial[0]
                records.append(make_record(
                    image_id    = img_id,
                    category    = "H1",
                    subcategory = "H1c",
                    question    = f"Is there a {absent} in this image?",
                    ground_truth= "no",
                    difficulty  = "hard",
                    metadata    = {
                        "absent_object":   absent,
                        "sampling":        "adversarial",
                        "present_objects": list(present_names),
                    },
                ))
                h1c_count += 1

        if h1a_count >= target and h1b_count >= target and h1c_count >= target:
            break

    log.info(f"H1: {h1a_count} random, {h1b_count} popular, {h1c_count} adversarial")
    return records


# ---------------------------------------------------------------------------
# Step 3 — H2: Object Attribute (Color + Size) from Visual Genome
# ---------------------------------------------------------------------------

def generate_h2(
    vg_attributes: list,
    vg_coco_map: dict,
    coco_val_ids: set,
    target: int = TARGET_PER_SUBCAT,
    rng: random.Random = None,
) -> list[dict]:
    """
    Generate H2a (color) and H2b (size) questions from VG attribute annotations.
    Restricted to images in the VG/COCO overlap set.
    """
    rng = rng or random.Random(42)
    records = []
    h2a_count = h2b_count = 0

    for img_data in vg_attributes:
        vg_id  = img_data.get("image_id") or img_data.get("id")
        coco_id = vg_coco_map.get(vg_id)
        if coco_id not in coco_val_ids:
            continue

        for obj in img_data.get("attributes", []):
            obj_name = obj.get("names", [""])[0].lower().strip()
            if not obj_name:
                continue

            attrs = [a.lower().strip() for a in obj.get("attributes", [])]

            # H2a — Color
            if h2a_count < target:
                true_colors = [a for a in attrs if a in COLORS]
                if true_colors:
                    true_color   = true_colors[0]
                    wrong_colors = [c for c in COLORS if c != true_color]
                    wrong_color  = rng.choice(wrong_colors)

                    # Positive question
                    records.append(make_record(
                        image_id    = coco_id,
                        category    = "H2",
                        subcategory = "H2a",
                        question    = f"Is the {obj_name} {true_color}?",
                        ground_truth= "yes",
                        difficulty  = "medium",
                        metadata    = {
                            "object": obj_name,
                            "true_color": true_color,
                        },
                    ))
                    # Negative question (wrong color)
                    records.append(make_record(
                        image_id    = coco_id,
                        category    = "H2",
                        subcategory = "H2a",
                        question    = f"Is the {obj_name} {wrong_color}?",
                        ground_truth= "no",
                        difficulty  = "medium",
                        metadata    = {
                            "object":      obj_name,
                            "true_color":  true_color,
                            "wrong_color": wrong_color,
                        },
                    ))
                    h2a_count += 1

            # H2b — Size comparison
            if h2b_count < target:
                size_attrs = [a for a in attrs if any(
                    s in a for s in ["large", "small", "big", "tall", "short", "wide"]
                )]
                if size_attrs:
                    size_word = rng.choice(SIZE_WORDS)
                    other_obj = rng.choice(
                        [o.get("names", ["object"])[0]
                         for o in img_data.get("attributes", [])
                         if o.get("names", [""])[0] != obj_name]
                    ) if len(img_data.get("attributes", [])) > 1 else "table"

                    records.append(make_record(
                        image_id    = coco_id,
                        category    = "H2",
                        subcategory = "H2b",
                        question    = f"Is the {obj_name} {size_word} than the {other_obj}?",
                        ground_truth= "ambiguous",  # will be manually verified
                        difficulty  = "hard",
                        metadata    = {
                            "object_a":  obj_name,
                            "object_b":  other_obj,
                            "size_word": size_word,
                            "needs_verification": True,
                        },
                    ))
                    h2b_count += 1

            if h2a_count >= target and h2b_count >= target:
                break

        if h2a_count >= target and h2b_count >= target:
            break

    log.info(f"H2: {h2a_count} color (H2a), {h2b_count} size (H2b)")
    return records


# ---------------------------------------------------------------------------
# Step 4 — H3: Relational from Visual Genome
# ---------------------------------------------------------------------------

def generate_h3(
    vg_relationships: list,
    vg_coco_map: dict,
    coco_val_ids: set,
    target: int = TARGET_PER_SUBCAT,
    rng: random.Random = None,
) -> list[dict]:
    """
    Generate H3a (spatial) and H3b (action) questions from VG relationships.
    Restricted to VG/COCO overlap images.
    """
    rng   = rng or random.Random(42)
    records = []
    h3a_count = h3b_count = 0

    spatial_predicates = {
        "to the left of", "to the right of", "above", "below",
        "on top of", "under", "in front of", "behind", "next to",
        "inside", "outside", "on"
    }
    action_predicates = {
        "holding", "wearing", "riding", "eating", "carrying",
        "sitting on", "standing on", "looking at", "playing with",
        "attached to", "hanging from", "walking on"
    }

    for img_data in vg_relationships:
        vg_id   = img_data.get("image_id") or img_data.get("id")
        coco_id = vg_coco_map.get(vg_id)
        if coco_id not in coco_val_ids:
            continue

        for rel in img_data.get("relationships", []):
            subj = rel.get("subject", {}).get("names", [""])[0].lower().strip()
            obj  = rel.get("object",  {}).get("names", [""])[0].lower().strip()
            pred = rel.get("predicate", "").lower().strip()

            if not subj or not obj or not pred:
                continue

            # H3a — spatial
            if h3a_count < target and pred in spatial_predicates:

                # Positive: true relationship
                records.append(make_record(
                    image_id    = coco_id,
                    category    = "H3",
                    subcategory = "H3a",
                    question    = f"Is the {subj} {pred} the {obj}?",
                    ground_truth= "yes",
                    difficulty  = "medium",
                    metadata    = {
                        "subject":   subj,
                        "predicate": pred,
                        "object":    obj,
                    },
                ))

                # Negative: swap subject and object (inverted relationship)
                records.append(make_record(
                    image_id    = coco_id,
                    category    = "H3",
                    subcategory = "H3a",
                    question    = f"Is the {obj} {pred} the {subj}?",
                    ground_truth= "no",
                    difficulty  = "hard",
                    metadata    = {
                        "subject":    obj,
                        "predicate":  pred,
                        "object":     subj,
                        "inverted":   True,
                    },
                ))
                h3a_count += 1

            # H3b — action/interaction
            if h3b_count < target and pred in action_predicates:
                records.append(make_record(
                    image_id    = coco_id,
                    category    = "H3",
                    subcategory = "H3b",
                    question    = f"Is the {subj} {pred} the {obj}?",
                    ground_truth= "yes",
                    difficulty  = "medium",
                    metadata    = {
                        "subject":   subj,
                        "predicate": pred,
                        "object":    obj,
                    },
                ))
                h3b_count += 1

            if h3a_count >= target and h3b_count >= target:
                break

        if h3a_count >= target and h3b_count >= target:
            break

    log.info(f"H3: {h3a_count} spatial (H3a), {h3b_count} action (H3b)")
    return records


# ---------------------------------------------------------------------------
# Step 5 — H4: Counting
# ---------------------------------------------------------------------------

def generate_h4(
    coco: COCO,
    instance_counts: dict,
    target: int = TARGET_PER_SUBCAT,
    rng: random.Random = None,
) -> list[dict]:
    """
    Generate counting questions from COCO.
    Each image gets a question about one object category present.
    """
    rng = rng or random.Random(42)
    records = []

    img_to_cats_counts = defaultdict(lambda: defaultdict(int))
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_to_cats_counts[ann["image_id"]][ann["category_id"]] += 1

    cat_names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    images = list(img_to_cats_counts.keys())
    rng.shuffle(images)
    count = 0

    for img_id in images:
        if count >= target:
            break

        cat_counts = img_to_cats_counts[img_id]
        # Pick a category with count between 2–8 (more interesting than 1 or 20+)
        valid = {cid: n for cid, n in cat_counts.items() if 2 <= n <= 8}
        if not valid:
            continue

        cid, true_count = rng.choice(list(valid.items()))
        obj_name = cat_names[cid]

        # Off-by-one wrong answer
        wrong_count = true_count + rng.choice([-1, 1, 2])
        wrong_count = max(1, wrong_count)

        # Free-form question
        records.append(make_record(
            image_id    = img_id,
            category    = "H4",
            subcategory = "H4",
            question    = f"How many {obj_name}s are in this image?",
            ground_truth= str(true_count),
            difficulty  = difficulty_from_count(instance_counts.get(obj_name, 0)),
            metadata    = {
                "object":      obj_name,
                "true_count":  true_count,
                "question_type": "free_form",
            },
        ))

        # Binary yes/no with wrong count (off-by-one)
        records.append(make_record(
            image_id    = img_id,
            category    = "H4",
            subcategory = "H4",
            question    = f"Are there exactly {wrong_count} {obj_name}s in this image?",
            ground_truth= "no",
            difficulty  = "hard",
            metadata    = {
                "object":        obj_name,
                "true_count":    true_count,
                "wrong_count":   wrong_count,
                "question_type": "binary_wrong",
            },
        ))

        # Binary yes/no with correct count
        records.append(make_record(
            image_id    = img_id,
            category    = "H4",
            subcategory = "H4",
            question    = f"Are there exactly {true_count} {obj_name}s in this image?",
            ground_truth= "yes",
            difficulty  = "medium",
            metadata    = {
                "object":        obj_name,
                "true_count":    true_count,
                "question_type": "binary_correct",
            },
        ))

        count += 1

    log.info(f"H4: {count} counting scenarios → {len([r for r in records if r['category']=='H4'])} questions")
    return records


# ---------------------------------------------------------------------------
# Step 6 — H5: Existence Negation
# ---------------------------------------------------------------------------

def generate_h5(
    coco: COCO,
    cooccurrence: np.ndarray,
    cat_ids: list,
    id_to_idx: dict,
    cat_names: dict,
    supercategories: dict,
    target: int = TARGET_PER_SUBCAT,
    rng: random.Random = None,
) -> list[dict]:
    """
    Generate H5a (negative pronoun) and H5b (implicit negation) questions.
    """
    rng = rng or random.Random(42)
    records = []

    img_to_cats = defaultdict(set)
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_to_cats[ann["image_id"]].add(ann["category_id"])

    all_cat_names = list(cat_names.values())
    images = list(img_to_cats.keys())
    rng.shuffle(images)

    h5a_count = h5b_count = 0

    for img_id in images:
        present_ids   = img_to_cats[img_id]
        present_names = {cat_names[c] for c in present_ids}
        absent_names  = list(set(all_cat_names) - present_names)

        if not absent_names:
            continue

        # H5a — negative pronoun framing
        if h5a_count < target:
            absent = rng.choice(absent_names)
            template = rng.choice(NEGATIVE_PRONOUN_TEMPLATES)
            question = template.format(OBJ=absent)
            records.append(make_record(
                image_id    = img_id,
                category    = "H5",
                subcategory = "H5a",
                question    = question,
                ground_truth= "yes",  # correct answer: yes, there is no X
                difficulty  = "hard",
                metadata    = {
                    "absent_object": absent,
                    "template":      template,
                },
            ))
            h5a_count += 1

        # H5b — implicit negation (which of these two is NOT present)
        if h5b_count < target and len(absent_names) >= 1 and len(present_names) >= 1:
            absent  = rng.choice(absent_names)
            present = rng.choice(list(present_names))

            # Randomise order so absent isn't always option B
            opts = [absent, present]
            rng.shuffle(opts)
            opt_a, opt_b = opts[0], opts[1]
            correct = "A" if opt_a == absent else "B"

            records.append(make_record(
                image_id    = img_id,
                category    = "H5",
                subcategory = "H5b",
                question    = f"Which of these is NOT in the image: (A) {opt_a} or (B) {opt_b}?",
                ground_truth= correct,
                difficulty  = "medium",
                metadata    = {
                    "option_a":      opt_a,
                    "option_b":      opt_b,
                    "absent_object": absent,
                },
            ))
            h5b_count += 1

        if h5a_count >= target and h5b_count >= target:
            break

    log.info(f"H5: {h5a_count} negative pronoun (H5a), {h5b_count} implicit negation (H5b)")
    return records


# ---------------------------------------------------------------------------
# Step 7 — Deduplication and quality filtering
# ---------------------------------------------------------------------------

def deduplicate(records: list[dict]) -> list[dict]:
    """Remove duplicate (image_id, question) pairs."""
    seen = set()
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


def filter_ambiguous(records: list[dict]) -> tuple[list, list]:
    """
    Split records into clean (ready to use) and needs_review (ambiguous).
    H2b size questions are flagged needs_verification=True.
    """
    clean  = [r for r in records if not r["metadata"].get("needs_verification")]
    review = [r for r in records if r["metadata"].get("needs_verification")]
    if review:
        log.info(f"{len(review)} records flagged for manual verification (H2b size questions).")
    return clean, review


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    rng = random.Random(args.seed)

    # ── Load COCO ──────────────────────────────────────────────────────────
    log.info("Loading COCO annotations...")
    coco = COCO(args.coco_ann)

    cat_names      = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    supercategories = {c["name"]: c["supercategory"] for c in coco.loadCats(coco.getCatIds())}
    coco_val_ids   = set(coco.imgs.keys())

    # Instance counts per category name
    instance_counts = {}
    for cat in coco.loadCats(coco.getCatIds()):
        ann_ids = coco.getAnnIds(catIds=cat["id"])
        instance_counts[cat["name"]] = len(ann_ids)

    # ── Load exploration findings ──────────────────────────────────────────
    findings = load_json(args.findings)
    log.info(f"Loaded findings: {findings['total_coco_val_images']} COCO images, "
             f"{findings['vg_coco_overlap_count']} VG/COCO overlap")

    # ── Build co-occurrence matrix ─────────────────────────────────────────
    cooccurrence, cat_ids, id_to_idx, _ = build_cooccurrence(coco)

    # ── Load Visual Genome ─────────────────────────────────────────────────
    log.info("Loading Visual Genome data...")
    vg_images = load_json(args.vg_imgs)
    vg_coco_map = {
        img["image_id"]: img["coco_id"]
        for img in vg_images
        if img.get("coco_id") is not None
    }

    vg_attributes    = load_json(args.vg_attrs)
    vg_relationships = load_json(args.vg_rels)

    # ── Generate questions per category ───────────────────────────────────
    all_records = []

    log.info("--- H1: Object Existence ---")
    all_records += generate_h1(
        coco, cooccurrence, cat_ids, id_to_idx, cat_names,
        instance_counts, target=args.target, rng=rng
    )

    log.info("--- H2: Object Attribute ---")
    all_records += generate_h2(
        vg_attributes, vg_coco_map, coco_val_ids,
        target=args.target, rng=rng
    )

    log.info("--- H3: Relational ---")
    all_records += generate_h3(
        vg_relationships, vg_coco_map, coco_val_ids,
        target=args.target, rng=rng
    )

    log.info("--- H4: Counting ---")
    all_records += generate_h4(
        coco, instance_counts,
        target=args.target, rng=rng
    )

    log.info("--- H5: Existence Negation ---")
    all_records += generate_h5(
        coco, cooccurrence, cat_ids, id_to_idx, cat_names,
        supercategories, target=args.target, rng=rng
    )

    # ── Deduplication + quality filter ────────────────────────────────────
    all_records = deduplicate(all_records)
    clean, needs_review = filter_ambiguous(all_records)

    # ── Dataset statistics ────────────────────────────────────────────────
    log.info("\n── Dataset Statistics ──────────────────────────────")
    subcat_counts = defaultdict(int)
    for r in clean:
        subcat_counts[r["subcategory"]] += 1
    for subcat, n in sorted(subcat_counts.items()):
        log.info(f"  {subcat:6s}  {n:5d} questions")
    log.info(f"  {'TOTAL':6s}  {len(clean):5d} questions")
    log.info(f"  {'REVIEW':6s}  {len(needs_review):5d} flagged for manual verification")

    # ── Save outputs ───────────────────────────────────────────────────────
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
    parser.add_argument("--coco_ann", default="annotations/instances_val2017.json")
    parser.add_argument("--vg_attrs", default="visual_genome/attributes.json")
    parser.add_argument("--vg_rels",  default="visual_genome/relationships.json")
    parser.add_argument("--vg_imgs",  default="image_data.json")
    parser.add_argument("--findings", default="results/exploration_findings.json")
    parser.add_argument("--output",   default="data/processed/benchmark_v1.json")
    parser.add_argument("--target",   type=int, default=TARGET_PER_SUBCAT,
                        help="Target questions per sub-category")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    main(args)
