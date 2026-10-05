"""
retrieval.py
------------
Loads the two FAISS indexes built by build_index.py:
  - image_index.faiss (from YOUR trained CNN) -> used by query_by_image()
  - text_index.faiss  (from local MiniLM on captions) -> used by query_by_text()
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json

import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer

from embedding_model import load_encoder


class VisionRetriever:
    def __init__(self, index_dir: str = "./index_store"):
        self.index_dir = index_dir

        with open(os.path.join(index_dir, "config.json")) as f:
            self.config = json.load(f)
        with open(os.path.join(index_dir, "metadata.json")) as f:
            self.metadata = json.load(f)

        self.image_index = faiss.read_index(os.path.join(index_dir, "image_index.faiss"))
        self.text_index = faiss.read_index(os.path.join(index_dir, "text_index.faiss"))

        self.encoder = load_encoder(self.config["checkpoint_path"])
        self.text_model = SentenceTransformer(self.config["text_model_name"])

    def _search(self, index, query_vec: np.ndarray, k: int):
        query_vec = query_vec.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vec)
        k = min(k, index.ntotal)
        scores, indices = index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def query_by_image(self, image_path: str, k: int = 5):
        img = Image.open(image_path).convert("RGB")
        vec = self.encoder.get_embedding(img)
        return self._search(self.image_index, vec, k)

    def query_by_text(self, text: str, k: int = 5):
        """
        Note: this does NOT search the image vectors directly (your
        trained CNN has no concept of text). Instead it matches your
        query against each image's predicted class name using a local
        text model. Good for queries like "cat" or "flower" that relate
        to CIFAR-100 class names; won't understand arbitrary free-form
        descriptions the way CLIP would.
        """
        vec = self.text_model.encode([text], convert_to_numpy=True)[0]
        return self._search(self.text_index, vec, k)


if __name__ == "__main__":
    retriever = VisionRetriever("./index_store")
    results = retriever.query_by_text("dog", k=5)
    for r in results:
        print(f"{r['score']:.3f}  {r['filename']}  (predicted: {r['caption']})")
