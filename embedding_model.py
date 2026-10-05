"""
embedding_model.py
--------------------
Defines the CNN architecture used to convert an image into a vector
(embedding), and shared helper functions to load the trained model
and extract embeddings from any image. This is used by
train_embedding_model.py (to train it) and by build_index.py /
retrieval.py (to use it).

This model is trained FROM SCRATCH on CIFAR-100 by
train_embedding_model.py -- it is NOT a pretrained/downloaded
embedding model like CLIP.
"""

import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 64          # all images are resized to this before encoding
EMBEDDING_DIM = 256      # size of the output vector stored in FAISS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match training-time preprocessing exactly
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


class EmbeddingCNN(nn.Module):
    """
    Simple CNN: 4 conv blocks -> flatten -> embedding layer -> classifier head.
    During training we use the classifier head (cross-entropy on class labels).
    At inference time (for indexing/retrieval) we only use the embedding
    layer's output -- the vector BEFORE the classifier head.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, num_classes: int = 100):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   # 64x64 -> 32x32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), # 32x32 -> 16x16
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), # 16x16 -> 8x8
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), # 8x8 -> 4x4
        )

        self.embedding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, embedding_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding: bool = False):
        x = self.features(x)
        embedding = self.embedding_layer(x)
        if return_embedding:
            return embedding
        logits = self.classifier(embedding)
        return logits, embedding


class ImageEncoder:
    """
    Loads the trained checkpoint once and exposes a simple
    get_embedding() / get_embedding_and_label() API for the rest
    of the project.
    """

    def __init__(self, checkpoint_path: str = "embedding_model.pth"):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Trained model not found at {checkpoint_path}. "
                "Run train_embedding_model.py first."
            )

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        self.embedding_dim = checkpoint["embedding_dim"]
        self.class_names = checkpoint["class_names"]

        self.model = EmbeddingCNN(
            embedding_dim=self.embedding_dim, num_classes=len(self.class_names)
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(DEVICE)
        self.model.eval()

    def _preprocess(self, image: Image.Image):
        return TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    @torch.no_grad()
    def get_embedding(self, image: Image.Image) -> np.ndarray:
        x = self._preprocess(image)
        embedding = self.model(x, return_embedding=True)
        return embedding.cpu().numpy()[0].astype("float32")

    @torch.no_grad()
    def get_embedding_and_label(self, image: Image.Image):
        """Returns (embedding_vector, predicted_class_name, confidence)."""
        x = self._preprocess(image)
        logits, embedding = self.model(x, return_embedding=False)
        probs = F.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
        label = self.class_names[pred_idx.item()]
        return (
            embedding.cpu().numpy()[0].astype("float32"),
            label,
            float(conf.item()),
        )


def load_encoder(checkpoint_path: str = "embedding_model.pth") -> ImageEncoder:
    return ImageEncoder(checkpoint_path)
