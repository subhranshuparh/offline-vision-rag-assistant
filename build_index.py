"""
build_index.py
----------------
Builds TWO FAISS indexes:
  1. image_index.faiss  -- vectors from YOUR trained CNN (embedding_model.py)
                            used for image-to-image search.
  2. text_index.faiss    -- vectors from a local MiniLM text encoder, one
                            per image's predicted class name (caption),
                            used so text queries can still find images.

Only the IMAGE vectorization (#1) needs to be your own trained model.
The text index (#2) is just for convenience so "search by typing a
sentence" keeps working; it never touches image pixels.

Usage:
    python build_index.py --images_dir ./sample_images --out_dir ./index_store
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import argparse
from pathlib import Path

import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from embedding_model import load_encoder

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # local, text-only


def collect_image_paths(images_dir: str):
    paths = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def build_index(images_dir: str, out_dir: str, checkpoint_path: str):
    os.makedirs(out_dir, exist_ok=True)

    print("Loading your trained image encoder...")
    encoder = load_encoder(checkpoint_path)

    print(f"Loading local text encoder ({TEXT_MODEL_NAME}) for caption matching...")
    text_model = SentenceTransformer(TEXT_MODEL_NAME)

    image_paths = collect_image_paths(images_dir)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    print(f"Found {len(image_paths)} images. Encoding...")

    image_vectors = []
    captions = []
    metadata = []

    for p in tqdm(image_paths):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue

        vec, label, confidence = encoder.get_embedding_and_label(img)
        image_vectors.append(vec)
        captions.append(label)
        metadata.append(
            {
                "path": os.path.abspath(p),
                "filename": os.path.basename(p),
                "caption": label,
                "confidence": round(confidence, 4),
            }
        )

    image_vectors = np.vstack(image_vectors).astype("float32")
    faiss.normalize_L2(image_vectors)
    image_index = faiss.IndexFlatIP(image_vectors.shape[1])
    image_index.add(image_vectors)

    print("Encoding captions with the local text model...")
    text_vectors = text_model.encode(captions, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(text_vectors)
    text_index = faiss.IndexFlatIP(text_vectors.shape[1])
    text_index.add(text_vectors)

    faiss.write_index(image_index, os.path.join(out_dir, "image_index.faiss"))
    faiss.write_index(text_index, os.path.join(out_dir, "text_index.faiss"))
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(
            {
                "checkpoint_path": os.path.abspath(checkpoint_path),
                "text_model_name": TEXT_MODEL_NAME,
                "image_dim": int(image_vectors.shape[1]),
                "text_dim": int(text_vectors.shape[1]),
                "count": len(metadata),
            },
            f,
            indent=2,
        )

    print(f"Done. Indexed {len(metadata)} images.")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./index_store")
    parser.add_argument("--checkpoint_path", type=str, default="embedding_model.pth")
    args = parser.parse_args()

    build_index(args.images_dir, args.out_dir, args.checkpoint_path)
