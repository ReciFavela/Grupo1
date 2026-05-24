import torch
import torch.nn as nn
import os
from torchvision import models, transforms
from PIL import Image
from collections import Counter
from pathlib import Path

# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"

# modelo binário
classes = [
    "NOT_PET",
    "PET"
]

# classes reais no dataset test
pastas_teste = [
    "HDPE",
    "LDPE",
    "Other",
    "PET",
    "PP",
    "PS",
    "PVC"
]

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# ================= MODEL =================
model = models.resnet18(weights=None)

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


# ================= PREDICT =================
def predict_image(path):

    img = Image.open(path).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)

        probs = torch.softmax(out, dim=1)

        pred_idx = torch.argmax(
            probs,
            dim=1
        ).item()

        confidence = probs[
            0,
            pred_idx
        ].item() * 100

    classe = classes[pred_idx]

    return classe, confidence


# ================= CONTADORES =================
acertos = 0
erros = 0

por_classe = {
    pasta: {
        "total": 0,
        "acertos": 0,
        "erros": 0
    }
    for pasta in pastas_teste
}

# matriz confusão simples
confusao = Counter()


# ================= MAIN =================
if __name__ == "__main__":

    pasta_root = "../imag-test"

    extensoes_validas = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    for pasta_real in pastas_teste:

        pasta_atual = os.path.join(
            pasta_root,
            pasta_real
        )

        if not os.path.exists(pasta_atual):
            print(f"Pasta não encontrada: {pasta_atual}")
            continue

        print(f"\n======== TESTANDO {pasta_real} ========")

        # rótulo real binário
        classe_real = (
            "PET"
            if pasta_real == "PET"
            else "NOT_PET"
        )

        for arquivo in os.listdir(pasta_atual):

            if not arquivo.lower().endswith(
                extensoes_validas
            ):
                continue

            caminho = os.path.join(
                pasta_atual,
                arquivo
            )

            pred, conf = predict_image(
                caminho
            )

            acertou = (
                pred == classe_real
            )

            por_classe[pasta_real]["total"] += 1

            if acertou:
                acertos += 1
                por_classe[pasta_real]["acertos"] += 1
                status = "ACERTOU"
            else:
                erros += 1
                por_classe[pasta_real]["erros"] += 1
                status = "ERROU"

            confusao[
                (classe_real, pred)
            ] += 1

            print(
                f"{arquivo} -> "
                f"Real:{classe_real} "
                f"Pred:{pred} "
                f"{conf:.2f}% "
                f"[{status}]"
            )


# ================= RESUMO =================
total = acertos + erros
acc = (
    acertos / total * 100
    if total > 0 else 0
)

print("\n==============================")
print("RESULTADO FINAL")
print("==============================")
print(f"Acertos: {acertos}")
print(f"Erros: {erros}")
print(f"Acurácia: {acc:.2f}%")

print("\n--- Por Classe ---")
for classe,dados in por_classe.items():

    if dados["total"] == 0:
        continue

    acc_classe = (
        dados["acertos"] /
        dados["total"] * 100
    )

    print(
        f"\n{classe}"
        f"\n Total: {dados['total']}"
        f"\n Acertos: {dados['acertos']}"
        f"\n Erros: {dados['erros']}"
        f"\n Accuracy: {acc_classe:.2f}%"
    )


print("\n--- Matriz Confusão ---")
print(
f"PET -> PET: {confusao[('PET','PET')]}"
)
print(
f"PET -> NOT_PET: {confusao[('PET','NOT_PET')]}"
)
print(
f"NOT_PET -> PET: {confusao[('NOT_PET','PET')]}"
)
print(
f"NOT_PET -> NOT_PET: {confusao[('NOT_PET','NOT_PET')]}"
)