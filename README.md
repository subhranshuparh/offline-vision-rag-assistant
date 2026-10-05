# Retrieval-Augmented Vision AI Assistant — Custom Trained Model Version

This version replaces CLIP with **your own CNN, trained from scratch**
on CIFAR-100, for converting images into vectors. Everything else
(FAISS retrieval, offline answer generation) stays the same and
still runs with zero API calls.

## Why this satisfies "the model must be trained on a dataset"

`train_embedding_model.py` trains `EmbeddingCNN` (defined in
`embedding_model.py`) from scratch — random initial weights, no
pretrained checkpoint — on CIFAR-100 (100 classes, 60,000 images).
The resulting `embedding_model.pth` is entirely your own trained
model. It is this model, not any third-party API or pretrained
network, that converts every image into the vector stored in FAISS.

## Pipeline

```
CIFAR-100 dataset
       │
       ▼
train_embedding_model.py  ──▶  embedding_model.pth  (YOUR trained CNN)
                                        │
your images/ ──────────────────────────┤
                                        ▼
                              build_index.py
                                   │        │
                    image_index.faiss   text_index.faiss
                    (from your CNN)     (local MiniLM on
                                         predicted labels)
                                        │
                        query (image or text) ──▶ retrieval.py
                                        │
                              top-k results
                                        │
                                        ▼
                          generate_answer.py (local TinyLlama)
                                        │
                                        ▼
                                  Final answer
```

## Setup

```bash
cd rag_vision_assistant_v2
pip install -r requirements.txt
```

## Step 1 — Train your own image embedding model

```bash
python train_embedding_model.py --epochs 15 --batch_size 128
```

- Downloads CIFAR-100 automatically the first time (needs internet
  once, for this download only).
- Trains the CNN from scratch. On CPU this may take a while (roughly
  30–90 minutes for 15 epochs depending on your laptop); reduce
  `--epochs` if needed, or use a GPU if available (auto-detected).
- Produces `embedding_model.pth` and `training_report.json` — include
  the report numbers (train/test accuracy) in your project report as
  proof of training.

## Step 2 — Build the FAISS index over your image collection

Put any images you want to search over into a folder (they do NOT
need to be CIFAR-100 images — any photos work, since the model just
needs to produce a vector for them):

```bash
python build_index.py --images_dir ./sample_images --out_dir ./index_store
```

## Step 3 — Query it

```python
from retrieval import VisionRetriever
from generate_answer import generate_answer

retriever = VisionRetriever("./index_store")

# Image-to-image search (uses YOUR trained CNN)
results = retriever.query_by_image("./query_photo.jpg", k=5)

# Text search (keyword-style, matches against predicted class names)
results = retriever.query_by_text("dog", k=5)

answer = generate_answer("What do these results show?", results)
print(answer)
```

## Step 4 — Run the demo UI

```bash
streamlit run app.py
```

## Honest limitations to mention in your report/viva

- **Text search is not true semantic search.** Because your CNN only
  understands images, text queries are matched against each image's
  *predicted CIFAR-100 class name* using a local text model — not
  against the image content directly. It works well for queries close
  to CIFAR-100 categories (animals, vehicles, household objects,
  etc.) but won't understand open-ended descriptions the way CLIP
  would.
- **Domain mismatch.** CIFAR-100 is a general-purpose dataset. If you
  point `build_index.py` at very different images (e.g. medical
  X-rays), the model will still produce a vector for them, but its
  predicted "class name" will be meaningless (it will guess one of
  the 100 CIFAR categories). Image-to-image similarity search still
  works reasonably in this case since it only needs relative visual
  similarity, not correct labels — but call this out as a limitation.
- **CIFAR-100 images are naturally small/low-detail** (resized to
  64×64 here), so fine-grained visual differences may be lost.

## Possible extensions for your report

- Fine-tune on a domain-specific labeled dataset instead of CIFAR-100
  once you have one, for much better relevance in that domain.
- Replace the classifier-based embedding with a **triplet-loss /
  contrastive-loss** training objective, which directly optimizes for
  "similar images = close vectors" rather than classification
  accuracy — the more "correct" approach for a pure retrieval system.
- Report embedding quality using retrieval metrics (Recall@k) on a
  small held-out labeled query set.
