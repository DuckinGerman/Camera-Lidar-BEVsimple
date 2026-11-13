
import os
import csv
import random
import numpy as np
import torch

# Windows spconv DLL support
if os.name == "nt":
    torch_lib = os.path.join(
        os.path.dirname(torch.__file__), "lib"
    )
    os.add_dll_directory(torch_lib)

from nuscenes.nuscenes import NuScenes

from models.bevfusion import MultiModalBEV
from losses.det_loss import DetectionLoss

from datasets.nuscenes_dataset import (
    NuScenesBEVDataset,
    find_available_samples,
    split_by_scene,
)


DATAROOT = r"D:\bevtrain\data\nuscenes"
SAVE_DIR = r"D:\bevtrain\outputs\multi_train"

VERSION = "v1.0-trainval"
DEVICE = "cuda"

EPOCHS = 10
LR = 1e-4
MAX_TRAIN_SAMPLES = 100
MAX_VAL_SAMPLES = 30

SEED = 42


def prepare_targets(targets, device):
    return {
        key: value.unsqueeze(0).to(device)
        for key, value in targets.items()
        if isinstance(value, torch.Tensor)
    }


def run_epoch(model, dataset, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    # Keep frozen backbone in eval mode
    if hasattr(model, "camera_bev"):
        camera = model.camera_bev
        if hasattr(camera, "backbone"):
            camera.backbone.eval()

    totals = []

    # for idx in range(len(dataset)):
    #     data = dataset[idx]
    indices = list(range(len(dataset)))

    if training :
        random.shuffle(indices)

    for step, idx in enumerate(indices, start=1):
        data = dataset[idx]
        targets = prepare_targets(
            data["targets"], DEVICE
        )

        with torch.set_grad_enabled(training):
            preds = model(
                dataset.nusc,
                data["sample"],
                data["points"]
            )

            losses = criterion(preds, targets)
            loss = losses["total"]

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at sample {idx}"
                )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 10.0
                )

                optimizer.step()

        totals.append(loss.item())

        if training and step % 10 == 0:
            print(
                f"  Step {step}/{len(dataset)} "
                f"Loss {np.mean(totals[-10:]):.4f}",
                flush=True
            )

    return float(np.mean(totals))


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    nusc = NuScenes(
        version=VERSION,
        dataroot=DATAROOT,
        verbose=False
    )

    print("Scanning available sensor files...", flush=True)
    available = find_available_samples(nusc)

    train_samples, val_samples = split_by_scene(
        available, seed=SEED
    )

    print("Complete samples:", len(available))
    print("Train available:", len(train_samples))
    print("Val available:", len(val_samples))

    random.Random(SEED).shuffle(train_samples)
    random.Random(SEED + 1).shuffle(val_samples)

    if MAX_TRAIN_SAMPLES is not None:
        train_samples = train_samples[:MAX_TRAIN_SAMPLES]

    if MAX_VAL_SAMPLES is not None:
        val_samples = val_samples[:MAX_VAL_SAMPLES]

    train_dataset = NuScenesBEVDataset(
        nusc, train_samples
    )
    val_dataset = NuScenesBEVDataset(
        nusc, val_samples
    )

    model = MultiModalBEV().to(DEVICE)
    criterion = DetectionLoss()

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR,
        weight_decay=1e-4
    )

    best_val = float("inf")
    csv_path = os.path.join(SAVE_DIR, "history.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss"
        ])

        for epoch in range(1, EPOCHS + 1):
            print(f"\nEpoch {epoch}/{EPOCHS}", flush=True)

            train_loss = run_epoch(
                model, train_dataset, criterion, optimizer
            )

            val_loss = run_epoch(
                model, val_dataset, criterion
            )

            print(
                f"Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f}",
                flush=True
            )

            writer.writerow([
                epoch, train_loss, val_loss
            ])
            f.flush()

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "version": VERSION,
                "seed": SEED,
            }

            torch.save(
                checkpoint,
                os.path.join(SAVE_DIR, "last.pt")
            )

            if val_loss < best_val:
                best_val = val_loss

                torch.save(
                    checkpoint,
                    os.path.join(SAVE_DIR, "best.pt")
                )

    print("\nTraining completed.")
    print("Best validation loss:", best_val)


if __name__ == "__main__":
    main()
