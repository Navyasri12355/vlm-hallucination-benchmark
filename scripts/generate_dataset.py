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
    H4   Counting / Numerical
    H5a  Existence Negation — negative pronoun
    H5b  Existence Negation — implicit negation
    H5c  Existence Negation — contrastive pair
    H5d  Existence Negation — unsolvable probe
    H7c  Bias & Interference — text-to-image interference
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

# Negative pronoun templates for H5a
NEGATIVE_PRONOUN_TEMPLATES = [
    "Is there no {OBJ} in this image?",
    "There is no {OBJ} in this image, correct?",
    "Does this image contain no {OBJ}?",
    "None of the objects visible are {OBJ}s, right?",
    "Is it true that no {OBJ} appears in this image?",
]

# Spatial predicates for H3a
SPATIAL_PREDICATES = {
    "to the left of", "to the right of", "above", "below",
    "on top of", "under", "in front of", "behind", "next to",
    "inside", "outside", "on", "beside", "near", "across from",
    "adjacent to", "surrounding", "overlapping"
}

# Expanded action predicates for H3b — based on actual VG predicate frequencies
ACTION_PREDICATES = {
    # Physical interaction
    "holding", "wearing", "carrying", "eating", "drinking",
    "riding", "driving", "pulling", "pushing", "throwing",
    "catching", "kicking", "hitting", "cutting",
    # Position/posture
    "sitting on", "standing on", "lying on", "kneeling on",
    "standing in", "sitting in", "leaning on", "leaning against",
    "resting on", "perched on",
    # Looking/attention
    "looking at", "watching", "reading", "using",
    # Other actions
    "playing with", "walking on", "running on", "jumping over",
    "flying over", "hanging from", "attached to", "connected to",
    "covered in", "mounted on", "growing on", "painted on",
    "reflected in", "parked on", "parked in", "tied to",
}

# Text-to-image interference templates for H7c
# The false presupposition is baked into the question
INTERFERENCE_TEMPLATES = [
    ("Since the {FALSE_OBJ} is in this image, what color is it?",
     "there is no {FALSE_OBJ} in this image"),
    ("Given that there is a {FALSE_OBJ} here, where is it located?",
     "there is no {FALSE_OBJ} in this image"),
    ("The {FALSE_OBJ} in this photo appears to be what color?",
     "there is no {FALSE_OBJ} in this image"),
    ("Can you describe the {FALSE_OBJ} visible in this scene?",
     "there is no {FALSE_OBJ} in this image"),
    ("What is the {FALSE_OBJ} doing in this image?",
     "there is no {FALSE_OBJ} in this image"),
]


# ---------------------------------------------------------------------------
# Helpers
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
    image_id: int,
    category: str,
    subcategory: str,
    question: str,
    ground_truth: str,
    difficulty: str,
    metadata: dict = None,
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


def difficulty_from_count(count: int, low=100, high=500) -> str:
    if count >= high:
        return "easy"
    elif count >= low:
        return "medium"
    return "hard"


def build_img_to_cats(coco: COCO) -> dict:
    img_to_cats = defaultdict(set)
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_to_cats[ann["image_id"]].add(ann["category_id"])
    return img_to_cats


# ---------------------------------------------------------------------------
# Co-occurrence matrix
# ---------------------------------------------------------------------------

def build_cooccurrence(coco: COCO):
    log.info("Building co-occurrence matrix...")
    cats      = coco.loadCats(coco.getCatIds())
    cat_ids   = [c["id"] for c in cats]
    cat_names = {c["id"]: c["name"] for c in cats}
    id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
    n = len(cat_ids)

    img_to_cats = build_img_to_cats(coco)
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

def generate_h1(coco, cooccurrence, cat_ids, id_to_idx, cat_names,
                instance_counts, target=TARGET_PER_SUBCAT, rng=None):
    rng = rng or random.Random(42)
    records = []

    img_to_cats = build_img_to_cats(coco)
    all_cat_names = list(cat_names.values())
    popular_cats  = sorted(instance_counts, key=instance_counts.get, reverse=True)[:20]

    images = list(img_to_cats.keys())
    rng.shuffle(images)
    h1a = h1b = h1c = 0

    for img_id in images:
        present_ids   = img_to_cats[img_id]
        present_names = {cat_names[c] for c in present_ids}
        absent_names  = list(set(all_cat_names) - present_names)
        if not absent_names:
            continue

        # H1a — random absent
        if h1a < target:
            absent = rng.choice(absent_names)
            records.append(make_record(
                img_id, "H1", "H1a",
                f"Is there a {absent} in this image?",
                "no", "easy",
                {"absent_object": absent, "sampling": "random"},
            ))
            present_obj = rng.choice(list(present_names))
            records.append(make_record(
                img_id, "H1", "H1a",
                f"Is there a {present_obj} in this image?",
                "yes", "easy",
                {"present_object": present_obj, "sampling": "random"},
            ))
            h1a += 1

        # H1b — popular absent
        if h1b < target:
            popular_absent = [c for c in popular_cats if c not in present_names]
            if popular_absent:
                absent = rng.choice(popular_absent[:5])
                records.append(make_record(
                    img_id, "H1", "H1b",
                    f"Is there a {absent} in this image?",
                    "no", "medium",
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
                records.append(make_record(
                    img_id, "H1", "H1c",
                    f"Is there a {absent} in this image?",
                    "no", "hard",
                    {
                        "absent_object":   absent,
                        "sampling":        "adversarial",
                        "present_objects": list(present_names),
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

def generate_h2(vg_attributes, vg_coco_map, coco_val_ids,
                target=TARGET_PER_SUBCAT, rng=None):
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
            obj_name = obj.get("names", [""])[0].lower().strip()
            if not obj_name:
                continue
            attrs = [a.lower().strip() for a in obj.get("attributes", [])]

            # H2a — color
            if h2a < target:
                true_colors = [a for a in attrs if a in COLORS]
                if true_colors:
                    true_color  = true_colors[0]
                    wrong_color = rng.choice([c for c in COLORS if c != true_color])
                    records.append(make_record(
                        coco_id, "H2", "H2a",
                        f"Is the {obj_name} {true_color}?",
                        "yes", "medium",
                        {"object": obj_name, "true_color": true_color},
                    ))
                    records.append(make_record(
                        coco_id, "H2", "H2a",
                        f"Is the {obj_name} {wrong_color}?",
                        "no", "medium",
                        {"object": obj_name, "true_color": true_color,
                         "wrong_color": wrong_color},
                    ))
                    h2a += 1

            # H2b — size
            if h2b < target:
                size_attrs = [a for a in attrs if any(
                    s in a for s in ["large", "small", "big", "tall", "short", "wide"]
                )]
                if size_attrs and len(objs) > 1:
                    other_names = [
                        o.get("names", ["object"])[0]
                        for o in objs
                        if o.get("names", [""])[0] != obj_name
                    ]
                    if other_names:
                        other_obj = rng.choice(other_names)
                        size_word = rng.choice(SIZE_WORDS)
                        records.append(make_record(
                            coco_id, "H2", "H2b",
                            f"Is the {obj_name} {size_word} than the {other_obj}?",
                            "ambiguous", "hard",
                            {"object_a": obj_name, "object_b": other_obj,
                             "size_word": size_word, "needs_verification": True},
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

def generate_h3(vg_relationships, vg_coco_map, coco_val_ids,
                target=TARGET_PER_SUBCAT, rng=None):
    rng = rng or random.Random(42)
    records = []
    h3a = h3b = 0

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
            if h3a < target and pred in SPATIAL_PREDICATES:
                records.append(make_record(
                    coco_id, "H3", "H3a",
                    f"Is the {subj} {pred} the {obj}?",
                    "yes", "medium",
                    {"subject": subj, "predicate": pred, "object": obj},
                ))
                # Inverted — swap subject and object
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

def generate_h4(coco, instance_counts, target=TARGET_PER_SUBCAT, rng=None):
    rng = rng or random.Random(42)
    records = []

    img_to_cat_counts = defaultdict(lambda: defaultdict(int))
    for ann in coco.loadAnns(coco.getAnnIds()):
        img_to_cat_counts[ann["image_id"]][ann["category_id"]] += 1

    cat_names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    images = list(img_to_cat_counts.keys())
    rng.shuffle(images)
    count = 0

    for img_id in images:
        if count >= target:
            break
        valid = {cid: n for cid, n in img_to_cat_counts[img_id].items()
                 if 2 <= n <= 8}
        if not valid:
            continue

        cid, true_count = rng.choice(list(valid.items()))
        obj_name    = cat_names[cid]
        wrong_count = max(1, true_count + rng.choice([-1, 1, 2]))

        # Free-form
        records.append(make_record(
            img_id, "H4", "H4",
            f"How many {obj_name}s are in this image?",
            str(true_count),
            difficulty_from_count(instance_counts.get(obj_name, 0)),
            {"object": obj_name, "true_count": true_count,
             "question_type": "free_form"},
        ))
        # Binary wrong
        records.append(make_record(
            img_id, "H4", "H4",
            f"Are there exactly {wrong_count} {obj_name}s in this image?",
            "no", "hard",
            {"object": obj_name, "true_count": true_count,
             "wrong_count": wrong_count, "question_type": "binary_wrong"},
        ))
        # Binary correct
        records.append(make_record(
            img_id, "H4", "H4",
            f"Are there exactly {true_count} {obj_name}s in this image?",
            "yes", "medium",
            {"object": obj_name, "true_count": true_count,
             "question_type": "binary_correct"},
        ))
        count += 1

    log.info(f"H4: {count} counting scenarios → "
             f"{len([r for r in records if r['category']=='H4'])} questions")
    return records


# ---------------------------------------------------------------------------
# H5 — Existence Negation (H5a, H5b, H5c, H5d)
# ---------------------------------------------------------------------------

def generate_h5(coco, cooccurrence, cat_ids, id_to_idx, cat_names,
                target=TARGET_PER_SUBCAT, rng=None):
    rng = rng or random.Random(42)
    records = []

    img_to_cats  = build_img_to_cats(coco)
    all_cat_names = list(cat_names.values())
    images        = list(img_to_cats.keys())
    rng.shuffle(images)

    h5a = h5b = h5c = h5d = 0

    # Pre-build: for H5c we need pairs — images that share a category
    # Map: category_name → list of image_ids that contain it
    cat_to_images = defaultdict(list)
    for img_id, cat_ids_present in img_to_cats.items():
        for cid in cat_ids_present:
            cat_to_images[cat_names[cid]].append(img_id)

    for img_id in images:
        present_ids   = img_to_cats[img_id]
        present_names = {cat_names[c] for c in present_ids}
        absent_names  = list(set(all_cat_names) - present_names)
        if not absent_names:
            continue

        # H5a — negative pronoun framing
        if h5a < target:
            absent   = rng.choice(absent_names)
            template = rng.choice(NEGATIVE_PRONOUN_TEMPLATES)
            records.append(make_record(
                img_id, "H5", "H5a",
                template.format(OBJ=absent),
                "yes",   # yes, correct — there IS no X
                "hard",
                {"absent_object": absent, "template": template},
            ))
            h5a += 1

        # H5b — implicit negation: which of A/B is NOT present
        if h5b < target and absent_names and present_names:
            absent  = rng.choice(absent_names)
            present = rng.choice(list(present_names))
            opts    = [absent, present]
            rng.shuffle(opts)
            opt_a, opt_b = opts[0], opts[1]
            correct = "A" if opt_a == absent else "B"
            records.append(make_record(
                img_id, "H5", "H5b",
                f"Which of these is NOT in the image: (A) {opt_a} or (B) {opt_b}?",
                correct, "medium",
                {"option_a": opt_a, "option_b": opt_b,
                 "absent_object": absent},
            ))
            h5b += 1

        # H5c — contrastive pair
        # Image A has object X. Image B does NOT have object X.
        # Question on image B: "Does this image contain the same [X] as the
        #                        previous image?" → no
        if h5c < target and present_names:
            shared_obj = rng.choice(list(present_names))
            candidates = [
                i for i in cat_to_images.get(shared_obj, [])
                if i != img_id
                and shared_obj not in {cat_names[c]
                                       for c in img_to_cats.get(i, set())}
            ]
            # Actually we want image_B where shared_obj is ABSENT
            absent_candidates = [
                i for i in images
                if shared_obj not in {cat_names[c]
                                      for c in img_to_cats.get(i, set())}
                and i != img_id
            ]
            if absent_candidates:
                img_b = rng.choice(absent_candidates[:50])
                records.append(make_record(
                    img_b, "H5", "H5c",
                    f"Does this image contain a {shared_obj}, "
                    f"like the reference image (image ID {img_id})?",
                    "no", "hard",
                    {
                        "reference_image_id": img_id,
                        "target_object":      shared_obj,
                        "reference_has_obj":  True,
                        "target_has_obj":     False,
                    },
                ))
                h5c += 1

        # H5d — unsolvable probe
        # Question presupposes an absent object — correct response is to deny
        if h5d < target and absent_names:
            absent = rng.choice(absent_names)
            probe_templates = [
                f"What color is the {absent} in this image?",
                f"Where is the {absent} located in this image?",
                f"What is the {absent} doing in this image?",
                f"How many {absent}s are there in this image?",
                f"Describe the {absent} in this image.",
            ]
            question = rng.choice(probe_templates)
            records.append(make_record(
                img_id, "H5", "H5d",
                question,
                f"there is no {absent} in this image",
                "hard",
                {"absent_object": absent,
                 "expected_refusal": True},
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

def generate_h7c(coco, target=TARGET_PER_SUBCAT, rng=None):
    """
    Questions embed a false presupposition about an object that is NOT
    in the image. A well-grounded model should reject the presupposition.
    A hallucinating model will answer as if the object is present.

    Ground truth: the correct response is to deny the presupposition.
    Evaluation: compare model output to expected_denial in metadata.
    """
    rng = rng or random.Random(42)
    records = []

    img_to_cats  = build_img_to_cats(coco)
    cat_names    = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    all_cat_names = list(cat_names.values())
    images        = list(img_to_cats.keys())
    rng.shuffle(images)

    count = 0
    for img_id in images:
        if count >= target:
            break

        present_names = {cat_names[c] for c in img_to_cats[img_id]}
        absent_names  = list(set(all_cat_names) - present_names)
        if not absent_names:
            continue

        # Pick a semantically plausible absent object (use popular absent logic)
        # to make the interference harder — more likely to fool the model
        false_obj = rng.choice(absent_names[:20] if len(absent_names) > 20
                               else absent_names)

        template, denial = rng.choice(INTERFERENCE_TEMPLATES)
        question = template.format(FALSE_OBJ=false_obj)
        expected_denial = denial.format(FALSE_OBJ=false_obj)

        records.append(make_record(
            img_id, "H7", "H7c",
            question,
            expected_denial,
            "hard",
            {
                "false_presupposition_object": false_obj,
                "present_objects":             list(present_names),
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


def filter_ambiguous(records: list):
    clean  = [r for r in records if not r["metadata"].get("needs_verification")]
    review = [r for r in records if r["metadata"].get("needs_verification")]
    if review:
        log.info(f"{len(review)} records flagged for manual verification (H2b).")
    return clean, review


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
    instance_counts = {}
    for cat in coco.loadCats(coco.getCatIds()):
        instance_counts[cat["name"]] = len(coco.getAnnIds(catIds=cat["id"]))

    # Load findings
    findings = load_json(args.findings)
    log.info(f"Loaded findings: {findings['total_coco_val_images']} COCO images, "
             f"{findings['vg_coco_overlap_count']} VG/COCO overlap")

    # Build co-occurrence matrix
    cooccurrence, cat_ids, id_to_idx, _ = build_cooccurrence(coco)

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
        target=args.target, rng=rng
    )

    log.info("--- H7c: Text-to-Image Interference ---")
    all_records += generate_h7c(
        coco, target=args.target, rng=rng
    )

    # Deduplication + quality filter
    all_records        = deduplicate(all_records)
    clean, needs_review = filter_ambiguous(all_records)

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
