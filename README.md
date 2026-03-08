# VLM Hallucination Benchmark

A structured benchmark for evaluating hallucination behaviour across Vision-Language Models (VLMs), covering 7 hallucination categories with fine-grained diagnostic breakdowns.

---

## Overview

Vision-Language Models frequently hallucinate — they confidently describe objects, attributes, or relationships that don't exist in the image. Existing benchmarks like POPE and CHAIR measure one or two hallucination types in isolation. This benchmark provides **unified, per-category scoring across 7 hallucination types** using a consistent image base, enabling direct cross-model comparison with fine-grained diagnostic breakdowns.

---

## Hallucination Taxonomy

The benchmark evaluates 7 primary hallucination categories grounded in 32 papers across evaluation, detection, mitigation, and survey literature.

| ID | Category | Description | Subcategories |
|---|---|---|---|
| H1 | Object Existence | Model asserts presence of an absent object | H1a random, H1b popular, H1c adversarial |
| H2 | Object Attribute | Object present but properties described incorrectly | H2a color, H2b shape/size, H2c material |
| H3 | Relational | Incorrect spatial or action relationship between objects | H3a spatial, H3b action/interaction |
| H4 | Counting | Incorrect count of present objects | — |
| H5 | Existence Negation | Fails on negation-framed questions | H5a negative pronoun, H5b implicit negation, H5c contrastive pair, H5d unsolvable probe |
| H6 | Cross-modal Consistency | Inconsistent answers across question phrasings | H6a phrasing contradiction, H6b description-answer inconsistency |
| H7 | Bias and Interference | Language prior or prompt overrides visual evidence | H7a region/OCR bias, H7b factual prior bias, H7c text/image interference |

Full taxonomy with definitions, question templates, metrics, and references: [`docs/taxonomy.md`](docs/taxonomy.md)

### What This Improves Over Prior Work

| Benchmark | H1 | H2 | H3 | H4 | H5 | H6 | H7 |
|---|---|---|---|---|---|---|---|
| CHAIR (2018) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| POPE (2023) | ✅ | ❌ | ❌ | ❌ | Partial | ❌ | ❌ |
| MME (2023) | ✅ | Partial | ❌ | ✅ | ❌ | ❌ | ❌ |
| HallusionBench (2024) | ✅ | Partial | ✅ | Partial | ❌ | ❌ | Partial |
| **This Benchmark** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Repository Structure

```
vlm-hallucination-benchmark/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── processed/               ← Generated QA pairs (benchmark_v1.json)
│   └── README.md
│
├── docs/
│   └── taxonomy.md              ← Full hallucination taxonomy (7 categories, 32 papers)
│
├── scripts/
│   ├── generate_dataset.py      ← Dataset generation from COCO + Visual Genome
│   ├── run_evaluation.py        ← Run VLMs on benchmark (coming soon)
│   └── score_results.py         ← Per-category scoring and analysis (coming soon)
│
├── notebooks/
│   └── README.md                ← Links to Kaggle notebooks
│
└── results/
    ├── exploration_findings.json  ← COCO annotation exploration summary
    └── cooccurrence_heatmap.png   ← Co-occurrence matrix for top 30 COCO categories
```

---

## Progress

### ✅ Week 1–2 — Literature Review + Taxonomy
- Reviewed 32 papers across evaluation benchmarks, detection, mitigation, and surveys
- Defined 7-category hallucination taxonomy in [`docs/taxonomy.md`](docs/taxonomy.md)
- H7 (Bias & Interference) introduced as a new category not present in any single prior benchmark
- Root cause classification table linking each category to its failure mechanism

### ✅ Week 3 — COCO Annotation Exploration
- Audited MS-COCO val2017 annotation structure (5,000 images, 80 categories)
- Built co-occurrence matrix for adversarial H1 sampling
- Measured Visual Genome / COCO overlap for H2/H3 question generation

**Key findings:**

| Finding | Value | Impact on Dataset Design |
|---|---|---|
| Total COCO val2017 images | 5,000 | Base image pool for H1, H4, H5 |
| Categories with >500 instances | 10 / 80 | Target capped at available instances per category |
| Categories with <300 instances | 50 / 80 | Sparse categories grouped or capped |
| VG / COCO overlap | 2,170 images (43.4%) | H2/H3 questions restricted to this subset |
| Dominant adversarial pair | `person` (50+ categories) | De-duplicated — `person` skipped as adversarial target |

**Top meaningful adversarial pairs (H1c):**

| Object in image | Adversarial absent object |
|---|---|
| microwave | oven |
| keyboard | mouse |
| zebra | giraffe |
| fork | dining table |
| toilet | sink |
| cat | bed |

### 🔄 Week 3–5 — Dataset Construction (In Progress)
- [`scripts/generate_dataset.py`](scripts/generate_dataset.py) written — generates H1–H5 from COCO + Visual Genome
- Pending: run script, validate output, generate H6/H7 questions

---

## Dataset

### Sources
- **MS-COCO val2017** — 5,000 images, 80 object categories, instance annotations
- **Visual Genome** — attribute and relationship annotations for H2/H3 questions

### Format
Each record in `data/processed/benchmark_v1.json`:

```json
{
  "image_id":     123456,
  "category":     "H1",
  "subcategory":  "H1c",
  "question":     "Is there a microwave in this image?",
  "ground_truth": "no",
  "difficulty":   "hard",
  "metadata": {
    "absent_object":   "microwave",
    "sampling":        "adversarial",
    "present_objects": ["oven", "sink", "refrigerator"]
  }
}
```

### Planned Size
~2,000–5,000 questions across all categories once generation is complete.

---

## Models to Evaluate

| Model | Size | Source | Status |
|---|---|---|---|
| LLaVA-1.5 | 7B | HuggingFace | ⏳ Planned |
| InstructBLIP | 7B | HuggingFace | ⏳ Planned |
| GPT-4V | — | OpenAI API | ⏳ Planned |
| Gemini Vision | — | Google API | ⏳ Planned |

---

## Setup

```bash
git clone https://github.com/Navyasri12355/vlm-hallucination-benchmark.git
cd vlm-hallucination-benchmark
pip install -r requirements.txt
```

### Download Data

```bash
# COCO val2017 annotations
wget https://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# Visual Genome
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip
wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip
unzip attributes.json.zip -d visual_genome/
unzip relationships.json.zip -d visual_genome/
unzip image_data.json.zip
```

### Generate Dataset

```bash
python scripts/generate_dataset.py \
    --coco_ann  annotations/instances_val2017.json \
    --vg_attrs  visual_genome/attributes.json \
    --vg_rels   visual_genome/relationships.json \
    --vg_imgs   image_data.json \
    --findings  results/exploration_findings.json \
    --output    data/processed/benchmark_v1.json \
    --target    400 \
    --seed      42
```

---

## GPU Resources

All experiments run on free-tier cloud GPUs. No local GPU required.

| Platform | GPU | Quota | Used For |
|---|---|---|---|
| Kaggle | T4 x2 | 30 hrs/week | Main inference runs |
| Google Colab | T4 | ~4 hrs/session | Quick prototyping |
| Lightning.ai | T4 | 22 hrs/month | Overflow experiments |
| HuggingFace Spaces | A100 (shared) | Free ZeroGPU | Final demo |

LLaVA-7B runs comfortably on a single T4 with 4-bit quantization (~6GB VRAM).

---

## Key Papers

| Paper | Venue | Relevance |
|---|---|---|
| CHAIR — Rohrbach et al. | EMNLP 2018 | Object hallucination in captioning |
| POPE — Li et al. | EMNLP 2023 | Binary object existence evaluation |
| MME — Fu et al. | 2023 | Comprehensive MLLM evaluation |
| HallusionBench — Guan et al. | CVPR 2024 | Language hallucination + visual illusion |
| NOPE — Lovenia et al. | 2023 | Negative object presence evaluation |
| RAH-Bench — Chen et al. | 2023 | Attribute and relation hallucination |
| Bingo — Cui et al. | 2023 | Bias and interference in GPT-4V |
| AMBER — Wang et al. | 2023 | LLM-free multi-dimensional benchmark |
| FAITHSCORE — Jing et al. | 2023 | Fine-grained hallucination evaluation |
| CAST — Gautier et al. | 2024 | Cross-modal consistency |

Full reference list with arXiv links: [`docs/taxonomy.md`](docs/taxonomy.md)

---

## Pull Request History

| PR | Branch | Description |
|---|---|---|
| #1 | `docs/taxonomy-v1` | Hallucination taxonomy v1 — 7 categories, 32 papers |
| #2 | `data/coco-annotation-exploration` | COCO annotation exploration — co-occurrence matrix, VG overlap |
| #3 | `data/qa-generation-v1` | Dataset generation script (in progress) |

---

## License

MIT
