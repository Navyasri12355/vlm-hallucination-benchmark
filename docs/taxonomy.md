# Hallucination Taxonomy for VLM Benchmark

## Definition

Hallucination in LVLMs refers to the generation of text with information that is not present in, or is inconsistent with, the visual input. This encompasses hallucinated objects, inaccurate attributes and relationships, unfaithful descriptions, and responses driven by language priors rather than actual visual evidence. Root causes include language prior dominance, insufficient visual context, biases and misinformation in training data, and misalignment between visual encoders and language decoders.

---

## Taxonomy Overview

This benchmark organises hallucinations into **7 primary categories**, each grounded in the peer-reviewed literature. The structure integrates findings from evaluation benchmarks, detection methods, mitigation research, and survey papers to produce the most comprehensive diagnostic coverage to date.

```
VLM Hallucinations
├── H1: Object Existence
│   ├── H1a: Nonexistent object (never in image)
│   ├── H1b: Existent object manipulation (modified present object)
│   └── H1c: Knowledge manipulation (object exists, context fabricated)
├── H2: Object Attribute
│   ├── H2a: Color
│   ├── H2b: Shape / Size
│   └── H2c: Material / Texture
├── H3: Relational
│   ├── H3a: Spatial (positional)
│   └── H3b: Action / Interaction
├── H4: Counting / Numerical
├── H5: Existence Negation
├── H6: Cross-modal Consistency
│   ├── H6a: Self-contradiction across question phrasings
│   └── H6b: Description-answer inconsistency
└── H7: Bias and Interference
    ├── H7a: Region / OCR bias
    ├── H7b: Factual bias (spurious image-language correlations)
    └── H7c: Text-to-image and image-to-image interference
```

---

## H1 — Object Existence

### Definition
The model asserts the presence of an object that does not appear in the image at all, or misrepresents properties of a present object based on fabricated context.

### Background
CHAIR (EMNLP 2018) introduced the object hallucination problem on MS-COCO captioning, defining CHAIR_i and CHAIR_s metrics over 80 COCO object categories. POPE (EMNLP 2023) formalised it as a binary Yes/No task with three sampling regimes — random, popular, and adversarial — demonstrating that co-occurrence priors in training data are the primary driver of hallucination. HaELM (2023) identified a strong "Yes-bias" in LVLMs: models tend to respond affirmatively to judgement-type queries regardless of visual content, independent of the object in question. LRV-Instruction / GAVIE provides the most refined typology of negative instruction semantics, which maps directly onto the three subtypes below. AMBER further introduces a category of "hallucinatory target objects" — objects that are disproportionately likely to be imagined by LVLMs due to their frequency in training corpora — which should be prioritised in adversarial sampling. CCEval (HallE-Switch) extended analysis to detailed captioning, investigating how alignment of the language decoder, volume of instruction data, and input image resolution each independently influence hallucination rate in H1.

MERLIM (2023) additionally identified that IT-LVLMs (Instruction-Tuned LVLMs) show particularly poor performance with multiple failure cases in visual grounding, producing hallucinatory events with sensitivity to input query phrasing — meaning identical visual content produces different H1 outcomes depending on how the question is framed.

### Subtypes
| Subtype | Description | Sampling Strategy |
|---|---|---|
| H1a — Nonexistent object | Object has never been in the image | Random → Popular → Adversarial |
| H1b — Existent object manipulation | A present object is described with fabricated modifications | Targeted based on present objects |
| H1c — Knowledge manipulation | Object exists but context/knowledge about it is wrong | Semantic distractor |

**Adversarial sampling logic:** For a given image, the adversarial absent object is the one that most frequently co-occurs with the present objects in MS-COCO training annotations. This is the hardest setting and the primary driver of high FPR.

### Question Templates
```
Binary:      "Is there a [OBJECT] in this image?"  →  yes / no
Open-ended:  "What objects can you see in this image?"  →  verify against COCO GT
Negated:     "There is no [OBJECT] here, correct?"  →  yes / no
```

### Key Metrics
- **Hallucination Rate**: % of all responses containing a hallucinated object
- **False Positive Rate (FPR)**: % of absent-object questions where model says "yes", broken down by random / popular / adversarial
- **Yes-bias score**: overall tendency to say "yes" regardless of content (from HaELM)
- **CHAIR_i / CHAIR_s**: for open-ended captioning evaluation

### Primary References
- Rohrbach et al., *CHAIR*, EMNLP 2018 — arxiv.org/abs/1809.02156
- Li et al., *POPE*, EMNLP 2023 — arxiv.org/abs/2305.10355
- Wang et al., *HaELM*, 2023 — arxiv.org/abs/2308.15126
- Liu et al., *LRV-Instruction / GAVIE*, 2023 — arxiv.org/abs/2306.14565
- Wang et al., *AMBER*, 2023 — arxiv.org/abs/2311.07397
- Yang et al., *CCEval / HallE-Switch*, 2023 — arxiv.org/abs/2310.01779
- Barros et al., *MERLIM*, 2023 — arxiv.org/abs/2312.02219

---

## H2 — Object Attribute

### Definition
The object is correctly identified as present, but one or more of its properties — color, size, shape, or material — are described incorrectly.

### Background
RAH-Bench (2023) explicitly defines Attribute Hallucination as one of three core hallucination types (alongside Categorical and Relation), built on the Panoptic Scene Graph Dataset (PSG) and reporting FPR as the primary evaluation metric. FGHE (2023) introduces 200 manually annotated binary questions across 50 images covering multi-object, attribute, and behaviour dimensions — directly extending POPE-style evaluation to fine-grained attribute space. MME's Fine-Grained Perception subtasks cover attribute recognition across 5 dimensions, benchmarking 30 models and finding attribute recognition to be systematically weaker than coarse-grained perception. AMBER's AMBERSCORE metric also covers attribute hallucination as one of its three dimensions (alongside existence and relation). VALOR-EVAL (2024) evaluates both coverage and faithfulness of attribute descriptions, introducing a holistic metric that penalises both missing and incorrect attribute mentions.

### Subtypes

#### H2a — Color
**Templates:**
```
"What color is the [OBJECT] in this image?"
"Is the [OBJECT] [COLOR]?"  →  yes / no
```
**Example:** Image shows a blue bicycle. Model says "red." → Hallucination.

#### H2b — Shape / Size
**Templates:**
```
"Is the [OBJECT1] larger than [OBJECT2]?"
"What shape is the [OBJECT] on the table?"
```

#### H2c — Material / Texture
**Templates:**
```
"What is the [OBJECT] made of?"
"Is the floor wood or tile?"
```

### Key Metrics
- Per-attribute accuracy (H2a, H2b, H2c scored separately)
- FPR per attribute type (from RAH-Bench)
- AMBERSCORE attribute dimension

### Ground Truth Source
Visual Genome attribute annotations; MS-COCO; PSG dataset

### Primary References
- Chen et al., *RAH-Bench*, 2023 — arxiv.org/abs/2311.16479
- Wang et al., *FGHE*, 2023 — arxiv.org/abs/2312.01701
- Fu et al., *MME*, 2023 — arxiv.org/abs/2306.13394
- Wang et al., *AMBER*, 2023 — arxiv.org/abs/2311.07397
- Hao et al., *VALOR-EVAL*, 2024 — arxiv.org/abs/2404.13874

---

## H3 — Relational

### Definition
Objects are correctly identified as present, but the model incorrectly describes the spatial relationship between them or incorrectly describes an action or interaction.

### Background
RAH-Bench defines Relation Hallucination as its second core type, built on PSG, using FPR as the evaluation metric. HallusionBench (CVPR 2024) provides the most challenging relational questions to date — covering geometry, spatial reasoning, and consecutive image comparisons — and finds that even GPT-4V achieves only 31.42% question-pair accuracy, with all open-source models below 16%. MMRel (2024) introduces a dedicated relation understanding dataset for the MLLM era, covering semantic, geometric, and comparative relations. MERLIM identified visual grounding failures in IT-LVLMs as a primary cause of relational hallucination — the model knows the objects but cannot spatially ground them relative to each other.

### Subtypes

#### H3a — Spatial Relations
Model swaps or misassigns positional relationships: left/right, above/below, in front of/behind, inside/outside.

**Templates:**
```
"Is the [OBJECT A] to the left of [OBJECT B]?"
"What is on top of the [OBJECT]?"
"Is the [OBJECT] inside or outside the [CONTAINER]?"
```

#### H3b — Action / Interaction
Model incorrectly describes what is happening between objects or persons.

**Templates:**
```
"Is the person holding the [OBJECT]?"
"Is the dog sitting or standing?"
"What is the person doing with the [OBJECT]?"
```

### Key Metrics
- Binary accuracy per question
- Separate scores for H3a and H3b
- FPR by relation type (spatial vs. action)
- Confusion matrix for direction swaps (left↔right, above↔below)

### Primary References
- Chen et al., *RAH-Bench*, 2023 — arxiv.org/abs/2311.16479
- Guan et al., *HallusionBench*, CVPR 2024 — arxiv.org/abs/2310.14566
- Nie et al., *MMRel*, 2024 — arxiv.org/abs/2406.09121
- Barros et al., *MERLIM*, 2023 — arxiv.org/abs/2312.02219

---

## H4 — Counting / Numerical

### Definition
The model produces an incorrect count of objects that are present in the image, or makes an incorrect numerical comparison between quantities.

### Background
MME includes counting as an explicit Fine-Grained Perception subtask and finds it among the weakest capabilities across 30 benchmarked models. HallusionBench includes counting within its statistics and geometry reasoning tasks. A dedicated 2024 paper on number hallucinations approaches counting from a **consistency perspective** — testing the same numerical fact in different phrasings and measuring whether the model contradicts itself, not just whether it gets the count wrong. This consistency-aware evaluation is a meaningful extension over single-question accuracy. MAD-Bench (2024) further demonstrated how deceptive prompts — questions that imply a wrong count — cause models to capitulate to the prompt even when the visual evidence is clear, representing a distinct failure mode from pure enumeration error.

### Design Notes
- Target counts between 1–10 (larger counts are visually ambiguous)
- Include **off-by-one** and **off-by-two** difficulty levels
- Include consistency pairs: ask the same count in two different framings, check for self-contradiction
- Exclude images with heavy occlusion or ambiguous object boundaries

### Question Templates
```
"How many [OBJECTS] are in this image?"          →  free-form number
"Are there exactly [N] [OBJECTS] in this image?" →  yes / no
"Are there more than [N] [OBJECTS]?"             →  yes / no
Consistency pair: Ask both "How many?" and "Are there [GT] [OBJECTS]?" → check agreement
```

### Key Metrics
- **Exact match accuracy**: prediction == ground truth
- **Off-by-one tolerance**: |pred − gt| ≤ 1 (lenient, reported separately)
- **Mean Absolute Error (MAE)**
- **Numerical consistency score**: % of paired questions where model answers are internally consistent

### Primary References
- Fu et al., *MME*, 2023 — arxiv.org/abs/2306.13394
- Guan et al., *HallusionBench*, CVPR 2024 — arxiv.org/abs/2310.14566
- Xiong et al., *Number Hallucination Consistency*, 2024 — arxiv.org/abs/2403.01373
- Luo et al., *MAD-Bench*, 2024 — arxiv.org/abs/2402.13220

---

## H5 — Existence Negation

### Definition
The model fails to correctly respond when the question is phrased as a negation, or when asked about objects that are scene-relevant but absent.

### Background
NOPE (2023) is the primary paper for this category. It constructs 29.5k synthetic negative-pronoun questions ("none", "no one", "nobody", "nowhere", "neither") and finds that VLMs hallucinate significantly more on images with higher lexical diversity, more scene-relevant co-occurring objects, and larger answer scopes. HaELM independently corroborates this through its analysis of Yes-bias, showing the problem is amplified when negation framing conflicts with the model's default affirmative response tendency. M-HalDetect (AAAI 2024) provides fine-grained annotations at the accurate / inaccurate / analysis level, with a reward model that correlates highly with human judgement — this can serve as an auxiliary scorer for H5 in cases where binary accuracy is insufficient. VQAv2-IDK (2024) introduces "I don't know" as a valid response class, directly relevant to existence negation: a well-calibrated model should express uncertainty or deny rather than hallucinate a positive answer.

### Subtypes
| Subtype | Description |
|---|---|
| Negative pronoun | Uses "no", "none", "neither", "nobody" in the question |
| Implicit negation | "What is missing from this scene?" |
| Contrastive pair | Two images — one with object, one without — model must differentiate |
| Unsolvable probe | Question presupposes an object that doesn't exist — correct answer is "not applicable" |

### Question Templates
```
"Is there no [OBJECT] in this image?"              →  yes (correct) / no (hallucination)
"Does this image NOT contain a [OBJECT]?"          →  true / false
"None of the [OBJECTS] are visible here, correct?" →  yes / no
"What type of [OBJECT] is in this image?" (none present) → correct: "there is no [OBJECT]"
```

### Key Metrics
- **Binary accuracy** on negation-framed questions
- **Negation bias score**: Δ(FPR negated − FPR positive-framing of same content)
- **IDK rate**: % of cases where model appropriately abstains or expresses uncertainty

### Primary References
- Lovenia et al., *NOPE*, 2023 — arxiv.org/abs/2310.05338
- Wang et al., *HaELM*, 2023 — arxiv.org/abs/2308.15126
- Hendryx et al., *M-HalDetect*, AAAI 2024 — arxiv.org/abs/2308.06394
- NC Soft, *VQAv2-IDK*, 2024 — arxiv.org/abs/2402.09717
- Miyai et al., *Unsolvable Problem Detection*, 2024 — arxiv.org/abs/2403.20331

---

## H6 — Cross-modal Consistency

### Definition
The model gives inconsistent answers to semantically equivalent questions when modality or phrasing changes, or contradicts itself between free-form descriptions and closed-form answers about the same image.

### Background
CAST (2024) proposes a two-stage evaluation where the model first generates similarity statements comparing two inputs, then judges its own output for truthfulness — inconsistencies reveal that language generation is not faithfully anchored to the visual representation. FAITHSCORE (2023) introduces a reference-free, fine-grained pipeline: a Recogniser LLM identifies descriptive content from the model's prediction, a Decomposer LLM generates atomic facts from that content, and a Verifier (visual entailment model, e.g. OFA) checks each atomic fact against the input image. CIEM (NeurIPS 2023 Workshop) demonstrates through contrastive instruction tuning that models give contradictory answers to factual and contrastive QA pairs about the same image. VALOR-EVAL (2024) combines coverage and faithfulness into a single holistic metric, penalising both hallucinated content and omitted true content in descriptive outputs.

H6 tests a higher-order property: a model could score perfectly on H1–H5 individually while still being internally contradictory across question types on the same image — making this the strongest diagnostic for deployment reliability.

### Subtypes

#### H6a — Self-contradiction Across Phrasings
Same factual question, different wording → model gives different answers.

**Design:**
```
Q1 (direct):     "Is there a red chair in this image?"  →  "Yes"
Q2 (paraphrased): "Can you see a chair that is red?"   →  "No"  ← contradiction
```

#### H6b — Description-Answer Inconsistency
Free-form description contradicts Yes/No answers about the same image.

**Design:**
```
Q: "Describe the furniture in this image."  →  no mention of red chair
Q: "Is there a red chair?"  →  "Yes"  ← contradiction
```

### Key Metrics
- **Consistency score**: % of paired questions with consistent answers
- **Self-contradiction rate**: % of images where free-form description contradicts Yes/No answers
- **FAITHSCORE**: atomic fact verification precision/recall
- **VALOR faithfulness score**: penalises hallucinated and omitted content jointly

### Primary References
- Gautier et al., *CAST*, 2024 — arxiv.org/abs/2409.11007
- Jing et al., *FAITHSCORE*, 2023 — arxiv.org/abs/2311.01477
- Zou et al., *CIEM*, NeurIPS 2023 Workshop — arxiv.org/abs/2309.02301
- Hao et al., *VALOR-EVAL*, 2024 — arxiv.org/abs/2404.13874

---

## H7 — Bias and Interference

### Definition
The model's response is distorted by systematic biases in its training (region bias, OCR bias, factual prior bias) or by interference between visual and textual signals — including cases where a misleading text prompt overrides clear visual evidence, or where a second image interferes with reasoning about the first.

### Background
Bingo (2023) is the primary paper for this category, providing 308 images and 370 QA pairs specifically designed to expose bias and interference failures in GPT-4V. It identifies two orthogonal failure dimensions: Bias (Region, OCR, Factual) and Interference (Image-to-Image, Text-to-Image). CorrelationQA (2024) extends this by constructing spurious image-language correlation pairs — images that are statistically associated with certain answers in training data even when the visual content contradicts them. MAD-Bench (2024) specifically tests how easily models are fooled by deceptive prompts, finding that explicit false presuppositions in questions cause models to capitulate even when visual evidence is clear. VLind-Bench (2024) introduces a dedicated benchmark for measuring language prior strength in LVLMs — operationalising exactly how much the model's textual prior overrides its visual signal. HallusionBench extends into visual illusion territory (famous illusions, geometric paradoxes), representing the extreme case where language prior is not just competing with but actively inverting the visual evidence.

IVL-Hallu / PhD (2024) introduces a prompted hallucination evaluation dataset that specifically targets intrinsic vs. prompt-induced hallucinations — directly relevant to distinguishing H7 from H1, since both involve absent objects but H7 is driven by prompt manipulation rather than co-occurrence priors.

### Subtypes

#### H7a — Region / OCR Bias
Model focuses on salient regions (faces, text) and hallucinates content in peripheral regions, or misreads text overlaid on images.

**Templates:**
```
"What does the sign in the background say?"
"What is written on the object in the bottom-left corner?"
```

#### H7b — Factual Prior Bias (Spurious Correlations)
Model uses real-world knowledge priors to answer instead of visual evidence — giving the "typically expected" answer rather than the visually correct one.

**Templates:**
```
"What color is the sky?" (shown as green in an edited image)
"Which team's jersey is this?" (logo obscured)
```

**Design:** Use CorrelationQA-style image-question pairs where the visual answer contradicts the statistical prior.

#### H7c — Text-to-Image and Image-to-Image Interference
A false premise in the question prompt overrides visual evidence (Text-to-Image), or a second image in the context interferes with reasoning about the first (Image-to-Image).

**Templates (Text-to-Image):**
```
"Since the cat is sitting on the mat, what color is the mat?" (no cat present)
"Given that it's raining in this photo, what are people holding?" (sunny image)
```

**Templates (Image-to-Image):**
```
Show two images. Reference object from image 1 when asking about image 2.
"Is the same red car visible here as in the previous image?"
```

### Key Metrics
- **Bias susceptibility score**: % of bias-type questions where model gives prior-driven rather than visually-grounded answer
- **Interference rate**: % of questions where a false text premise or second image changes the model's answer
- **Language prior dominance score** (from VLind-Bench): quantifies how strongly the textual prior overrides visual evidence per model

### Primary References
- Cui et al., *Bingo*, 2023 — arxiv.org/abs/2311.03287
- Han et al., *CorrelationQA*, 2024 — arxiv.org/abs/2402.03757
- Luo et al., *MAD-Bench*, 2024 — arxiv.org/abs/2402.13220
- Han et al., *VLind-Bench*, 2024 — arxiv.org/abs/2406.08702
- Guan et al., *HallusionBench*, CVPR 2024 — arxiv.org/abs/2310.14566
- Zhu et al., *IVL-Hallu / PhD*, 2024 — arxiv.org/abs/2403.11116

---

## Summary Table

| ID | Category | Ground Truth Source | Question Type | Key Metric | Primary Papers |
|---|---|---|---|---|---|
| H1a | Obj. Existence (nonexistent) | MS-COCO instances | Binary + Open | FPR (random/popular/adversarial) | CHAIR, POPE, AMBER, HallE-Switch |
| H1b | Obj. Existence (manipulation) | MS-COCO + LRV typology | Binary | FPR | LRV/GAVIE, HaELM |
| H2a | Color | Visual Genome attributes | Binary + Open | Per-color Accuracy | RAH-Bench, FGHE, MME |
| H2b | Shape/Size | Visual Genome | Binary | Accuracy | RAH-Bench, AMBER |
| H2c | Material | Visual Genome | Binary | Accuracy | RAH-Bench |
| H3a | Spatial Relations | Visual Genome / PSG | Binary | Accuracy, FPR | RAH-Bench, HallusionBench, MMRel |
| H3b | Action/Interaction | Visual Genome / PSG | Binary | Accuracy | RAH-Bench, MERLIM |
| H4 | Counting / Numerical | MS-COCO instances | Free-form + Binary | Exact Match, MAE, Consistency | MME, HallusionBench, Number Consistency |
| H5 | Existence Negation | MS-COCO instances | Binary (negated) | Accuracy, Negation Bias, IDK Rate | NOPE, HaELM, M-HalDetect, IDK |
| H6a | Cross-modal Consistency (phrasings) | Self-generated pairs | Paired Yes/No | Consistency Score | CAST, CIEM |
| H6b | Description-Answer Consistency | Self-generated pairs | Open + Binary | FAITHSCORE, VALOR | FAITHSCORE, VALOR-EVAL |
| H7a | Region / OCR Bias | Bingo annotations | Binary | Bias Susceptibility Score | Bingo |
| H7b | Factual Prior Bias | CorrelationQA pairs | Binary | Prior Override Rate | CorrelationQA, VLind-Bench |
| H7c | Text/Image Interference | MAD-Bench / Bingo | Binary | Interference Rate | MAD-Bench, Bingo, HallusionBench |

---

## What This Taxonomy Improves Over Prior Work

| Benchmark | H1 Obj. | H2 Attr. | H3 Rel. | H4 Count | H5 Neg. | H6 Consistency | H7 Bias |
|---|---|---|---|---|---|---|---|
| CHAIR (2018) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| POPE (2023) | ✅ | ❌ | ❌ | ❌ | Partial | ❌ | ❌ |
| MME (2023) | ✅ | Partial | ❌ | ✅ | ❌ | ❌ | ❌ |
| RAH-Bench (2023) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| HallusionBench (2024) | ✅ | Partial | ✅ | Partial | ❌ | ❌ | Partial |
| NOPE (2023) | Partial | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Bingo (2023) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| AMBER (2023) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| CAST / FAITHSCORE | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **This Benchmark** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## Dataset Construction Notes

- **Image source**: MS-COCO val2017 (5,000 images) + Visual Genome (attributes/relations) + Bingo-style edited images (for H7)
- **Target**: 300–500 questions per top-level category (H1–H7), balanced across subtypes
- **Difficulty labels**: easy / medium / hard per question based on occlusion level, co-occurrence strength, negation complexity, or degree of prompt manipulation
- **Quality control**: each question-answer pair verified against ground truth annotation; adversarial questions validated to ensure the absent object genuinely co-occurs with present ones in training data
- **Avoid**: images with bounding boxes < 32×32 px, heavy motion blur, or ambiguous annotation

---

## Root Cause Classification

Understanding which root cause drives each hallucination type guides both dataset design and future mitigation work:

| Root Cause | Hallucination Categories Affected | Key Papers |
|---|---|---|
| Language prior dominance | H1 (adversarial), H7b, H7c | POPE, VLind-Bench, MAD-Bench |
| Insufficient visual grounding | H2, H3a, H3b | RAH-Bench, MERLIM, MMRel |
| Yes-bias / affirmation tendency | H1, H5 | HaELM, NOPE, M-HalDetect |
| Training data co-occurrence bias | H1a (adversarial), H4, H7b | POPE, AMBER, CorrelationQA |
| Prompt manipulation susceptibility | H5, H7c | MAD-Bench, IVL-Hallu, HallusionBench |
| Representation misalignment | H6a, H6b | CAST, FAITHSCORE, CIEM |
| Region / saliency bias | H7a | Bingo, MERLIM |

---

## Full Reference List

### Evaluation Benchmarks
1. Rohrbach et al. — *CHAIR*, EMNLP 2018 — arxiv.org/abs/1809.02156
2. Li et al. — *POPE*, EMNLP 2023 — arxiv.org/abs/2305.10355
3. Fu et al. — *MME*, 2023 — arxiv.org/abs/2306.13394
4. Hendryx et al. — *M-HalDetect*, AAAI 2024 — arxiv.org/abs/2308.06394
5. Wang et al. — *HaELM*, 2023 — arxiv.org/abs/2308.15126
6. Zou et al. — *CIEM*, NeurIPS 2023 Workshop — arxiv.org/abs/2309.02301
7. Gautier et al. — *CAST*, 2024 — arxiv.org/abs/2409.11007
8. Sun et al. — *MMHAL-BENCH / Fact-RLHF*, 2023 — arxiv.org/abs/2309.14525
9. Liu et al. — *LRV-Instruction / GAVIE*, 2023 — arxiv.org/abs/2306.14565
10. Lovenia et al. — *NOPE*, 2023 — arxiv.org/abs/2310.05338
11. Guan et al. — *HallusionBench*, CVPR 2024 — arxiv.org/abs/2310.14566
12. Jing et al. — *FAITHSCORE*, 2023 — arxiv.org/abs/2311.01477
13. Cui et al. — *Bingo*, 2023 — arxiv.org/abs/2311.03287
14. Wang et al. — *AMBER*, 2023 — arxiv.org/abs/2311.07397
15. Chen et al. — *RAH-Bench*, 2023 — arxiv.org/abs/2311.16479
16. Barros et al. — *MERLIM*, 2023 — arxiv.org/abs/2312.02219
17. Yang et al. — *CCEval / HallE-Switch*, 2023 — arxiv.org/abs/2310.01779
18. Wang et al. — *FGHE*, 2023 — arxiv.org/abs/2312.01701
19. Han et al. — *CorrelationQA*, 2024 — arxiv.org/abs/2402.03757
20. Tong et al. — *VQAv2-IDK*, 2024 — arxiv.org/abs/2402.09717
21. Chen et al. — *MHaluBench / UNIHD*, 2024 — arxiv.org/abs/2402.03190
22. Luo et al. — *MAD-Bench*, 2024 — arxiv.org/abs/2402.13220
23. Xiong et al. — *Number Hallucination Consistency*, 2024 — arxiv.org/abs/2403.01373
24. Zhu et al. — *IVL-Hallu / PhD*, 2024 — arxiv.org/abs/2403.11116
25. Miyai et al. — *Unsolvable Problem Detection*, 2024 — arxiv.org/abs/2403.20331
26. Hao et al. — *VALOR-EVAL*, 2024 — arxiv.org/abs/2404.13874
27. Nie et al. — *MMRel*, 2024 — arxiv.org/abs/2406.09121
28. Han et al. — *VLind-Bench*, 2024 — arxiv.org/abs/2406.08702

### Surveys
29. Liu et al. — *Survey on Hallucination in LVLMs*, 2024 — arxiv.org/abs/2402.00253
30. Bai et al. — *Hallucination of MLLMs: A Survey*, 2024
31. Huang et al. — *Visual Hallucination: Definition, Quantification, and Prescriptive Remediations*, 2024
32. Kamruzzaman et al. — *Unveiling Hallucination in Text, Image, Video, and Audio Foundation Models*, 2024
