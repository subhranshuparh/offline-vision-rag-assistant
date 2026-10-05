"""
train_embedding_model.py
--------------------------
Trains the EmbeddingCNN (defined in embedding_model.py) FROM SCRATCH
on CIFAR-100, a general-purpose 100-class image dataset (60,000
images total). This is the model your teacher asked for: a model
YOU trained on a dataset, used to convert images into vectors.

CIFAR-100 downloads automatically the first time you run this
(needs internet ONCE, for the download only -- not needed afterwards).

Usage:
    python train_embedding_model.py --epochs 15 --batch_size 128
"""

import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from embedding_model import EmbeddingCNN, IMAGE_SIZE, EMBEDDING_DIM, DEVICE

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

TEST_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def train(epochs: int, batch_size: int, lr: float, out_path: str):
    print(f"Using device: {DEVICE}")

    print("Loading CIFAR-100 (auto-downloads on first run)...")
    train_set = datasets.CIFAR100(root="./data", train=True, download=True, transform=TRAIN_TRANSFORM)
    test_set = datasets.CIFAR100(root="./data", train=False, download=True, transform=TEST_TRANSFORM)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = train_set.classes  # list of 100 class name strings

    model = EmbeddingCNN(embedding_dim=EMBEDDING_DIM, num_classes=len(class_names)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(images, return_embedding=False)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_set)

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits, _ = model(images, return_embedding=False)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  test_accuracy={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": EMBEDDING_DIM,
                    "class_names": class_names,
                    "test_accuracy": test_acc,
                },
                out_path,
            )
            print(f"  -> Saved new best checkpoint to {out_path} (acc={test_acc:.4f})")

    print(f"\nTraining complete. Best test accuracy: {best_acc:.4f}")
    print(f"Final model saved at: {out_path}")

    # Save training report for your project documentation
    with open("training_report.json", "w") as f:
        json.dump(
            {
                "dataset": "CIFAR-100",
                "num_classes": len(class_names),
                "num_train_images": len(train_set),
                "num_test_images": len(test_set),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "embedding_dim": EMBEDDING_DIM,
                "best_test_accuracy": best_acc,
            },
            f,
            indent=2,
        )
    print("Saved training_report.json (include this in your project report)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_path", type=str, default="embedding_model.pth")
    args = parser.parse_args()

    train(args.epochs, args.batch_size, args.lr, args.out_path)
