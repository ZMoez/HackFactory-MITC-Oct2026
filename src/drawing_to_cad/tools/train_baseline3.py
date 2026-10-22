
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_DIR = PROJECT_ROOT / "result"
MODEL_DIR = RESULT_DIR / "models"
MODEL_PATH = MODEL_DIR / "multimodal_baseline.pt"

DEVICE = "cpu"
EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
PATIENCE = 30

torch.manual_seed(42)
np.random.seed(42)


class MultiModalRegressor(nn.Module):
    """Predict bounding-box dimensions from an image and numeric features."""

    def __init__(self, num_features: int):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
        )

        self.regressor = nn.Sequential(
            nn.Linear(32 * 4 * 4 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, images, features):
        image_embedding = self.image_encoder(images)
        feature_embedding = self.feature_encoder(features)

        combined = torch.cat(
            [image_embedding, feature_embedding],
            dim=1,
        )
        return self.regressor(combined)


def load_dataset(split_name):
    path = (
        RESULT_DIR
        / split_name
        / f"{split_name}_feature_dataset.npz"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run prepare_feature_dataset.py first."
        )

    with np.load(path, allow_pickle=False) as data:
        images = data["images"].copy()
        features = data["features"].copy()
        labels = data["labels"].copy()
        part_ids = data["part_ids"].tolist()
        feature_names = data["feature_names"].tolist()

    if not (len(images) == len(features) == len(labels) == len(part_ids)):
        raise ValueError(f"Inconsistent dataset lengths: {path}")

    if not np.isfinite(images).all():
        raise ValueError(f"Non-finite image values in {path}")

    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError(f"Non-finite features or labels in {path}")

    print(
        f"Loaded {split_name}: {len(part_ids)} samples, "
        f"{features.shape[1]} features"
    )

    return images, features, labels, part_ids, feature_names


def main():
    # Load only train and validation. Test remains untouched.
    train_images, train_features, train_labels, _, train_names = (
        load_dataset("train")
    )
    val_images, val_features, val_labels, _, val_names = (
        load_dataset("validation")
    )

    if train_names != val_names:
        raise ValueError(
            "Train and validation feature columns do not match."
        )

    # Calculate normalization statistics from training data only.
    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0

    target_mean = train_labels.mean(axis=0)
    target_std = train_labels.std(axis=0)
    target_std[target_std < 1e-8] = 1.0

    train_features_scaled = (train_features - feature_mean) / feature_std
    val_features_scaled = (val_features - feature_mean) / feature_std

    train_labels_scaled = (train_labels - target_mean) / target_std
    val_labels_scaled = (val_labels - target_mean) / target_std

    x_train_images = torch.tensor(
        train_images[:, None], dtype=torch.float32
    )
    x_train_features = torch.tensor(
        train_features_scaled, dtype=torch.float32
    )
    y_train = torch.tensor(
        train_labels_scaled, dtype=torch.float32
    )

    x_val_images = torch.tensor(
        val_images[:, None], dtype=torch.float32
    )
    x_val_features = torch.tensor(
        val_features_scaled, dtype=torch.float32
    )
    y_val = torch.tensor(
        val_labels_scaled, dtype=torch.float32
    )

    loader = DataLoader(
        TensorDataset(x_train_images, x_train_features, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = MultiModalRegressor(
        num_features=train_features.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    loss_fn = nn.MSELoss()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for image_batch, feature_batch, labels_batch in loader:
            image_batch = image_batch.to(DEVICE)
            feature_batch = feature_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(image_batch, feature_batch)
            loss = loss_fn(predictions, labels_batch)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_predictions = model(
                x_val_images.to(DEVICE),
                x_val_features.to(DEVICE),
            )
            val_loss = loss_fn(
                val_predictions,
                y_val.to(DEVICE),
            ).item()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss: {np.mean(train_losses):.4f} | "
                f"val loss: {val_loss:.4f}"
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_features": train_features.shape[1],
                    "feature_names": train_names,
                    "feature_mean": feature_mean.tolist(),
                    "feature_std": feature_std.tolist(),
                    "target_mean": target_mean.tolist(),
                    "target_std": target_std.tolist(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                },
                MODEL_PATH,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved model: {MODEL_PATH}")
    print("Test data was not used during training.")


if __name__ == "__main__":
    main()
