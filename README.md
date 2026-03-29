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

---

## Progress

### ✅ Week 1–2 — Literature Review + Taxonomy
- Reviewed 32 papers across evaluation benchmarks, detection, mitigation, and surveys
- Defined 7-category hallucination taxonomy in [`docs/taxonomy.md`](docs/taxonomy.md)
- H7 (Bias & Interference) introduced as a new category not present in any single prior benchmark
- Root cause classification table linking each category to its failure mechanism

### ✅ Week 3 — COCO Annotation Exploration
- Audited MS-COCO val2017 annotation structure (5,000 images, 80 categories)
- Built co-occurrence matrix for adversarial H1c sampling
- Measured Visual Genome / COCO overlap (2,170 images, 43.4%) for H2/H3 generation

### ✅ Week 3–5 — Dataset Construction
- Generated 5,860 QA pairs across 12 subcategories from COCO val2017 + Visual Genome
- Covers H1a/b/c, H2a, H3a/b, H4, H5a/b/c/d, H7c
- Adversarial sampling, contrastive pairs, unsolvable probes, and interference questions included

**Dataset composition:**

| Subcategory | Questions | Type |
|---|---|---|
| H1a | 800 | Binary yes/no (random absent + present pairs) |
| H1b | 397 | Binary yes/no (popular absent) |
| H1c | 337 | Binary yes/no (adversarial co-occurrence) |
| H2a | 667 | Binary yes/no (color attribute) |
| H3a | 587 | Binary yes/no (spatial relations) |
| H3b | 168 | Binary yes/no (action/interaction) |
| H4  | 1,200 | Free-form + binary counting |
| H5a | 400 | Binary yes/no (negative pronoun) |
| H5b | 400 | A/B choice (implicit negation) |
| H5c | 400 | Binary yes/no (contrastive pair) |
| H5d | 400 | Open-ended refusal (unsolvable probe) |
| H7c | 400 | Open-ended refusal (text-image interference) |
| **Total** | **5,860** | |

### ✅ Week 6–7 — Model Evaluation (LLaVA-1.6)
- Ran LLaVA-1.6-Mistral-7B (4-bit quantized) on all 5,860 benchmark records
- Overall accuracy: **70.1%** across scored subcategories

**LLaVA-1.6-Mistral-7B results:**

| Subcategory | Accuracy | Notes |
|---|---|---|
| H1a | 95.2% | Strong on basic existence |
| H1b | 91.7% | Popular objects well handled |
| H1c | 77.9% | Adversarial pairs cause noticeable drop |
| H2a | 75.1% | Color attributes moderately accurate |
| H3a | 58.0% | Spatial relations a clear weakness |
| H3b | 91.9% | Action relations handled well |
| H4  | 57.2% | Counting consistently difficult |
| H5a | 75.6% | Negative pronoun handled moderately |
| H5b | 52.1% | Implicit negation near chance |
| H5c | 97.5% | Contrastive pairs — model defaults to "no" |
| H5d | 54.5% | Unsolvable probes — model often answers rather than refusing |
| H7c | 26.0% | Text-image interference — lowest score, strongest finding |

**Key finding:** `binary_wrong` accuracy (26.5%) vs `binary_correct` accuracy (86.0%) reveals a strong sycophancy pattern — the model tends to agree with whatever the question asserts regardless of visual evidence.

### 🔄 Week 7–8 — Model Evaluation (InstructBLIP, In Progress)
- Running InstructBLIP-Vicuna-7B (8-bit quantized) on all 5,860 records
- Results pending

> All Kaggle notebooks (COCO exploration, Model inferencing (Llava and InstructBLIP)) are linked in [`notebooks/README.md`](notebooks/README.md).

---

## Dataset

### Sources
- **MS-COCO val2017** — 5,000 images, 80 object categories, instance annotations
- **Visual Genome** — attribute and relationship annotations for H2/H3 questions

### Format

Each record in `data/processed/benchmark_v1.json`:

---

## Models Evaluated

| Model | Size | Quantization | VRAM | Overall Accuracy | Status |
|---|---|---|---|---|---|
| LLaVA-1.6-Mistral | 7B | 4-bit (nf4) | ~6GB | 70.1% | ✅ Complete |
| InstructBLIP-Vicuna | 7B | 8-bit | ~9GB | — | 🔄 In Progress |

---

## GPU Resources

All experiments run on free-tier cloud GPUs. No local GPU required.

| Platform | GPU | Quota | Used For |
|---|---|---|---|
| Kaggle | T4 x2 | 30 hrs/week | Main inference runs |
| Google Colab | T4 | ~4 hrs/session | Quick prototyping |
| Lightning.ai | T4 | 22 hrs/month | Overflow experiments |
| HuggingFace Spaces | A100 (shared) | Free ZeroGPU | Final demo |

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
| #3 | `data/qa-generation-v1` | Dataset generation — 5,860 QA pairs across 12 subcategories |
| #4 | `eval/model-inference` | LLaVA-1.6-Mistral-7B inference + scoring (70.1% overall); InstructBLIP in progress |

---

## License

MIT