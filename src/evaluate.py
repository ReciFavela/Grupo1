import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ================= CONFIG =================

MODEL_PATH = config.BEST_MODEL_PATH
TEST_DIR = config.TEST_DIR
BATCH_SIZE = 32

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if not TEST_DIR.exists() or not any(TEST_DIR.iterdir()):
    print(
        f"Erro: {TEST_DIR} não encontrado ou vazio.\n"
        "Execute primeiro: python scripts/split_data.py"
    )
    sys.exit(1)

# ================= TRANSFORMS =================

transform = transforms.Compose([
    transforms.Lambda(
        lambda img: img.convert("RGB")
    ),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ================= DATA =================

dataset = datasets.ImageFolder(
    str(TEST_DIR),
    transform=transform
)

classes = dataset.classes
print("Classes:", classes)
print(f"Imagens de teste: {len(dataset)}")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ================= MODEL =================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()

# ================= EVALUATION =================

all_preds = []
all_labels = []

with torch.no_grad():

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        out = model(x)

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


# ================= RESULTS =================

acc = accuracy_score(
    all_labels,
    all_preds
)

cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\n=== Conjunto de TESTE (nunca visto no treino) ===")
print("\nAccuracy:")
print(f"{acc * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=classes,
        digits=4
    )
)
