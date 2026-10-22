
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "result"
MODEL_DIR = RESULT_DIR / "models"
DEVICE = "cpu"

EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

torch.manual_seed(42)
np.random.seed(42)


class DimensionRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.network(x)


def load_dataset(split_name):
    path = (
        RESULT_DIR
        / split_name
        / f"{split_name}_image_dataset.npz"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run prepare_image_dataset.py first."
        )

    with np.load(path) as data:
        images = data["images"].copy()
        labels = data["labels"].copy()
        part_ids = data["part_ids"].tolist()

    if len(images) == 0:
        raise ValueError(f"{split_name} dataset is empty.")

    if len(images) != len(labels) or len(images) != len(part_ids):
        raise ValueError(f"Inconsistent data lengths in {path}")

    print(f"Loaded {split_name}: {len(images)} samples")
    return images, labels, part_ids


def main():
    # Load the existing splits independently.
    train_images, train_labels, _ = load_dataset("train")
    val_images, val_labels, _ = load_dataset("validation")

    # Normalize labels using training data only.
    target_mean = train_labels.mean(axis=0)
    target_std = train_labels.std(axis=0)
    target_std[target_std == 0] = 1.0

    x_train = torch.tensor(
        train_images[:, None], dtype=torch.float32
    )
    y_train = torch.tensor(
        (train_labels - target_mean) / target_std,
        dtype=torch.float32,
    )

    x_val = torch.tensor(
        val_images[:, None], dtype=torch.float32
    )
    y_val = torch.tensor(
        (val_labels - target_mean) / target_std,
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = DimensionRegressor().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "dimension_baseline.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(xb)
            loss = loss_fn(predictions, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_predictions = model(x_val.to(DEVICE))
            val_loss = loss_fn(
                val_predictions, y_val.to(DEVICE)
            ).item()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss: {np.mean(train_losses):.4f} | "
                f"val loss: {val_loss:.4f}"
            )

        # Save the model with the best validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_mean": target_mean.tolist(),
                    "target_std": target_std.tolist(),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                model_path,
            )

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Saved best model: {model_path}")
    print("Test data was not used during training.")


if __name__ == "__main__":
    main()
