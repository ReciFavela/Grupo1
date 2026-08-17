import os
import sys
import multiprocessing
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from collections import Counter

def get_auto_config():
    cfg = {}

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1e9

        if vram < 4:
            batch = 16
        elif vram < 8:
            batch = 32
        elif vram < 16:
            batch = 64
        else:
            batch = 128

        cfg["device"] = torch.device("cuda")
        cfg["batch"] = batch
        cfg["use_amp"] = True
        cfg["pin_memory"] = True

        torch.backends.cudnn.benchmark = True

        print("GPU:", props.name)
        print(f"VRAM: {vram:.1f} GB")

    else:
        cfg["device"] = torch.device("cpu")
        cfg["batch"] = 8
        cfg["use_amp"] = False
        cfg["pin_memory"] = False
        print("Rodando em CPU")

    # Windows-safe
    cfg["workers"] = 0

    return cfg


def build_class_weights(train_dataset, device):
    labels = [label for _, label in train_dataset.samples]
    counts = Counter(labels)
    num_classes = len(train_dataset.classes)

    weights = torch.tensor(
        [1.0 / counts[i] for i in range(num_classes)],
        dtype=torch.float32,
    )
    weights = weights / weights.sum() * num_classes

    print("Pesos por classe:", dict(zip(train_dataset.classes, weights.tolist())))
    return weights.to(device)


def build_train_sampler(train_dataset):
    labels = [label for _, label in train_dataset.samples]
    counts = Counter(labels)
    sample_weights = torch.tensor(
        [1.0 / counts[label] for label in labels],
        dtype=torch.float32,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def main():

    cfg = get_auto_config()

    device = cfg["device"]
    BATCH_SIZE = cfg["batch"]
    NUM_WORKERS = cfg["workers"]
    USE_AMP = cfg["use_amp"]
    PIN_MEMORY = cfg["pin_memory"]

    LR = 3e-4

    CHECKPOINT_PATH = config.CHECKPOINT_PATH
    BEST_MODEL_PATH = config.BEST_MODEL_PATH

    config.MODELS_DIR.mkdir(exist_ok=True)

    # ================= TRANSFORMS =================

    train_transform = transforms.Compose([
        transforms.Lambda(
            lambda img: img.convert("RGB")
        ),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(
            lambda img: img.convert("RGB")
        ),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])

    # ================= DATA =================

    if not config.TRAIN_DIR.exists() or not config.VAL_DIR.exists():
        print(
            "Erro: data/train ou data/val não encontrados.\n"
            "Execute primeiro: python scripts/split_data.py"
        )
        sys.exit(1)

    train_dataset = datasets.ImageFolder(
        str(config.TRAIN_DIR),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        str(config.VAL_DIR),
        transform=val_transform
    )

    if train_dataset.classes != val_dataset.classes:
        print("Erro: classes diferentes entre train e val.")
        sys.exit(1)

    train_sampler = build_train_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print("Classes:", train_dataset.classes)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print("Batch:", BATCH_SIZE)

    # ================= MODEL =================

    model = models.resnet18(
        weights="DEFAULT"
    )

    for p in model.parameters():
        p.requires_grad=False

    model.fc = nn.Linear(
        model.fc.in_features,
        len(train_dataset.classes)
    ) 

    model=model.to(device)

    class_weights = build_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LR
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    # ================= RESUME =================

    epoch=0
    best_f1=0

    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(
                CHECKPOINT_PATH,
                map_location=device
            )

            if isinstance(ckpt,dict):
                model.load_state_dict(
                    ckpt["model_state"]
                )

                optimizer.load_state_dict(
                    ckpt["optimizer_state"]
                )

                scaler.load_state_dict(
                    ckpt["scaler_state"]
                )

                epoch = ckpt["epoch"]
                best_f1 = ckpt["best_f1"]

                print(
                    f"Retomando epoch {epoch}"
                )
        except:
            print("Checkpoint ignorado")


    # ================= TREINO INFINITO =================

    while True:

        epoch +=1

        # -------- TRAIN --------

        model.train()
        total_loss=0

        loop=tqdm(
            train_loader,
            desc=f"Epoch {epoch}"
        )

        for x,y in loop:

            x=x.to(device)
            y=y.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(
                "cuda",
                enabled=USE_AMP
            ):
                out=model(x)
                loss=criterion(out,y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            loop.set_postfix(
                loss=loss.item()
            )

        train_loss = total_loss / len(train_loader)

        print(
            f"Train Loss: {train_loss:.6f}"
        )


        # -------- FINE TUNING --------

        if epoch==2:
            print(
                "Descongelando layer4"
            )

            for name,p in model.named_parameters():
                if (
                    "layer4" in name
                    or "fc" in name
                ):
                    p.requires_grad=True

            optimizer = torch.optim.Adam(
                filter(
                    lambda p:
                    p.requires_grad,
                    model.parameters()
                ),
                lr=LR*0.5
            )


       # -------- VALIDATION --------

        model.eval()

        val_loss = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)
                y = y.to(device)

                out = model(x)

                loss = criterion(out, y)
                val_loss += loss.item()

                preds = torch.argmax(
                    out,
                    dim=1
                )

                all_preds.extend(
                    preds.cpu().numpy()
                )

                all_labels.extend(
                    y.cpu().numpy()
                )

        avg_val_loss = (
            val_loss / len(val_loader)
        )

        # métricas
        acc = accuracy_score(
            all_labels,
            all_preds
        )

        precision = precision_score(
            all_labels,
            all_preds,
            pos_label=1
        )

        recall = recall_score(
            all_labels,
            all_preds,
            pos_label=1
        )

        f1 = f1_score(
            all_labels,
            all_preds,
            pos_label=1
        )

        cm = confusion_matrix(
            all_labels,
            all_preds
        )

        print(
            f"Val Loss: {avg_val_loss:.6f}"
        )

        print(
            f"Val Acc: {acc*100:.2f}%"
        )

        print(
            f"Precision PET: {precision:.4f}"
        )

        print(
            f"Recall PET: {recall:.4f}"
        )

        print(
            f"F1 Score: {f1:.4f}"
        )

        print("Confusion Matrix:")
        print(cm)

        # -------- SAVE BEST --------

        if f1 > best_f1 and avg_val_loss < 0.05:
            best_f1 = f1

            torch.save(
                model.state_dict(),
                BEST_MODEL_PATH
            )

            print(
                "Melhor modelo salvo por F1"
            )

        # -------- CHECKPOINT --------

        torch.save(
            {
                "model_state":model.state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "scaler_state":scaler.state_dict(),
                "epoch":epoch,
                "best_f1": best_f1
            },
            CHECKPOINT_PATH
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()