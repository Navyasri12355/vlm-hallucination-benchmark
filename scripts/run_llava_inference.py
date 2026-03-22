"""
run_llava_inference.py
----------------------
Runs LLaVA-1.5-7B-hf on the VLM hallucination benchmark dataset.
Produces: results/llava_raw_responses.json

Designed for Kaggle T4 (16GB VRAM). Uses 4-bit quantization (~6GB VRAM).
Supports checkpoint/resume — safe to restart after session timeout.

Usage:
    python scripts/run_llava_inference.py \
        --benchmark  data/processed/benchmark_v1.json \
        --images_dir /kaggle/input/coco-2017-dataset/val2017 \
        --output     results/llava_raw_responses.json \
        --checkpoint results/llava_checkpoint.json

Output record format (one per benchmark item):
    {
        "image_id":     123456,
        "category":     "H1",
        "subcategory":  "H1c",
        "question":     "Is there a microwave in this image?",
        "ground_truth": "no",
        "difficulty":   "hard",
        "metadata":     {...},
        "model":        "llava-hf/llava-1.5-7b-hf",
        "raw_response": "No, there is no microwave visible in this image.",
        "error":        null
    }
"""

import json
import argparse
import logging
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)

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
MODEL_ID       = "llava-hf/llava-1.5-7b-hf"
MAX_NEW_TOKENS = 128
IMAGE_EXTS     = [".jpg", ".jpeg", ".png"]
CHECKPOINT_EVERY = 100   # save checkpoint every N records


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


def find_image(image_id: int, images_dir: str) -> str | None:
    """
    Locate the image file for a COCO image_id.
    COCO filenames are zero-padded to 12 digits: e.g. 000000123456.jpg
    """
    stem = str(image_id).zfill(12)
    for ext in IMAGE_EXTS:
        path = os.path.join(images_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str) -> dict:
    """
    Returns a dict keyed by (image_id, question) of already-completed records.
    Empty dict if no checkpoint exists yet.
    """
    if os.path.exists(checkpoint_path):
        log.info(f"Resuming from checkpoint: {checkpoint_path}")
        records = load_json(checkpoint_path)
        done = {(r["image_id"], r["question"]): r for r in records}
        log.info(f"  {len(done)} records already completed.")
        return done
    return {}


def save_checkpoint(done: dict, checkpoint_path: str) -> None:
    save_json(list(done.values()), checkpoint_path)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """
    Load LLaVA-1.5-7B with 4-bit quantization (BitsAndBytes nf4).
    Uses ~6GB VRAM — fits comfortably on a single T4.
    """
    log.info(f"Loading model: {MODEL_ID}")
    log.info("Applying 4-bit quantization (nf4, float16 compute)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    log.info("Model ready.")
    return processor, model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def build_prompt(question: str) -> str:
    """
    LLaVA-1.5 instruction-following format.
    The <image> token is replaced by image patch embeddings at runtime.
    """
    return f"USER: <image>\n{question}\nASSISTANT:"


def run_single(processor, model, image_path: str, question: str) -> str:
    """
    Run one (image, question) pair through LLaVA-1.5-7B.
    Returns the raw generated text (prompt tokens stripped).

    Note: LlavaNextForConditionalGeneration requires image_sizes to be passed
    to generate(). We move each tensor individually rather than calling
    .to(device) on the whole BatchFeature to avoid dropping image_sizes,
    which is a plain Python list and has no .to() method.
    """
    image  = Image.open(image_path).convert("RGB")
    prompt = build_prompt(question)

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    )

    # Move tensor values to device; leave non-tensor values (e.g. image_sizes) as-is
    inputs_on_device = {
        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs_on_device,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,           # greedy decoding — deterministic
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    input_len = inputs_on_device["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # ── Load benchmark ─────────────────────────────────────────────────────
    log.info(f"Loading benchmark: {args.benchmark}")
    benchmark = load_json(args.benchmark)
    log.info(f"  {len(benchmark)} total records.")

    # ── Load checkpoint ────────────────────────────────────────────────────
    done = load_checkpoint(args.checkpoint)
    remaining = [
        r for r in benchmark
        if (r["image_id"], r["question"]) not in done
    ]
    log.info(f"  {len(remaining)} records to process.")

    if not remaining:
        log.info("All records already processed. Writing final output.")
        save_json(list(done.values()), args.output)
        return

    # ── Verify images directory ────────────────────────────────────────────
    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(
            f"Images directory not found: {args.images_dir}\n\n"
            f"On Kaggle, add the COCO 2017 dataset to your notebook:\n"
            f"  Notebook → Add Data → Search 'COCO 2017' → Add dataset\n"
            f"  Then check the exact path with: os.listdir('/kaggle/input/')"
        )

    # ── Load model ─────────────────────────────────────────────────────────
    processor, model = load_model()

    # ── Run inference ──────────────────────────────────────────────────────
    errors     = 0
    start_time = time.time()

    for i, record in enumerate(remaining):
        image_id = record["image_id"]
        question = record["question"]
        key      = (image_id, question)

        image_path = find_image(image_id, args.images_dir)

        if image_path is None:
            log.warning(f"[{i+1}] Image not found: ID {image_id} — skipping.")
            done[key] = {
                **record,
                "model":        MODEL_ID,
                "raw_response": None,
                "error":        "image_not_found",
            }
            errors += 1
            continue

        try:
            raw_response = run_single(processor, model, image_path, question)
            done[key] = {
                **record,
                "model":        MODEL_ID,
                "raw_response": raw_response,
                "error":        None,
            }
        except Exception as e:
            log.error(f"[{i+1}] Inference error on image {image_id}: {e}")
            done[key] = {
                **record,
                "model":        MODEL_ID,
                "raw_response": None,
                "error":        str(e),
            }
            errors += 1

        # ── Progress + checkpoint ──────────────────────────────────────────
        if (i + 1) % CHECKPOINT_EVERY == 0:
            elapsed       = time.time() - start_time
            per_record    = elapsed / (i + 1)
            est_remaining = per_record * (len(remaining) - i - 1)
            log.info(
                f"  Progress: {i+1}/{len(remaining)}  |  "
                f"elapsed {elapsed/60:.1f}m  |  "
                f"est. remaining {est_remaining/60:.1f}m  |  "
                f"errors {errors}"
            )
            save_checkpoint(done, args.checkpoint)
            log.info(f"  Checkpoint saved → {args.checkpoint}")

    # ── Final save ─────────────────────────────────────────────────────────
    save_json(list(done.values()), args.output)
    total_time = (time.time() - start_time) / 60
    log.info(f"Done. {len(done)} records saved → {args.output}")
    log.info(f"Total time: {total_time:.1f}m  |  Errors: {errors}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLaVA-1.5-7B inference on the VLM hallucination benchmark"
    )
    parser.add_argument(
        "--benchmark",
        default="data/processed/benchmark_v1.json",
        help="Path to benchmark_v1.json",
    )
    parser.add_argument(
        "--images_dir",
        default="/kaggle/input/coco-2017-dataset/coco2017/val2017",
        help="Directory containing COCO val2017 images",
    )
    parser.add_argument(
        "--output",
        default="results/llava_raw_responses.json",
        help="Path to save raw model responses",
    )
    parser.add_argument(
        "--checkpoint",
        default="results/llava_checkpoint.json",
        help="Checkpoint file for resume support (auto-created)",
    )
    args = parser.parse_args()
    main(args)