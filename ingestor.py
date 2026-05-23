"""
worker/ingestor.py
Pipeline completo de ingestão:
  1. Lê uma imagem do dataset
  2. YOLO detecta objetos (bounding boxes)
  3. Modelo PyTorch confirma se é PET ou NOT_PET
  4. Salva imagem no MinIO
  5. Grava evento bruto no MongoDB
  6. Grava métrica agregada no InfluxDB

Uso:
    pip install -r requirements.txt
    python worker/ingestor.py --imagens data/PET --camera cam_01 --turno manha
"""

import argparse
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

from ultralytics import YOLO

from pymongo import MongoClient
from influxdb_client import InfluxDBClient, Point, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
from minio import Minio

# ─────────────────────────────────────────
# Configurações
# ─────────────────────────────────────────

MONGO_URI    = "mongodb://admin:recifavela123@localhost:27017/"
MONGO_DB     = "recifavela"
MONGO_COL    = "frames"

INFLUX_URL   = "http://localhost:8086"
INFLUX_TOKEN = "recifavela-super-secret-token"
INFLUX_ORG   = "recifavela"
INFLUX_BUCKET = "deteccoes"

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS   = "admin"
MINIO_SECRET   = "recifavela123"
MINIO_BUCKET   = "frames"

MODELO_PATH    = "models/best_model.pth"
CONFIANCA_MIN  = 0.5   # threshold YOLO
CLASSES        = ["NOT_PET", "PET"]   # índice 0 e 1

# ─────────────────────────────────────────
# Transformações — igual ao treino
# ─────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────
# Carrega modelo PyTorch (classificador)
# ─────────────────────────────────────────

def carregar_modelo_classificador():
    print("[Classificador] Carregando best_model.pth...")
    modelo = models.resnet18(weights=None)
    modelo.fc = torch.nn.Linear(modelo.fc.in_features, 2)
    modelo.load_state_dict(torch.load(MODELO_PATH, map_location="cpu"))
    modelo.eval()
    print("[Classificador] Pronto.")
    return modelo

# ─────────────────────────────────────────
# Classificador: recebe crop PIL → retorna classe e confiança
# ─────────────────────────────────────────

def classificar(modelo, crop: Image.Image):
    tensor = transform(crop).unsqueeze(0)
    with torch.no_grad():
        saida = modelo(tensor)
        probs = torch.softmax(saida, dim=1)[0]
        idx   = probs.argmax().item()
    return CLASSES[idx], round(probs[idx].item(), 4)

# ─────────────────────────────────────────
# YOLO: detecta objetos na imagem inteira
# ─────────────────────────────────────────

def detectar_yolo(yolo, imagem_path: str):
    resultados = yolo(imagem_path, conf=CONFIANCA_MIN, verbose=False)
    boxes = []
    for r in resultados:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = round(float(box.conf[0]), 4)
            boxes.append({"bbox": [x1, y1, x2, y2], "conf_yolo": conf})
    return boxes

# ─────────────────────────────────────────
# MinIO: salva imagem e retorna o path
# ─────────────────────────────────────────

def salvar_minio(minio_client, imagem_path: str, camera_id: str, frame_id: str):
    extensao = Path(imagem_path).suffix
    objeto   = f"{camera_id}/{frame_id}{extensao}"
    minio_client.fput_object(MINIO_BUCKET, objeto, imagem_path)
    return f"{MINIO_BUCKET}/{objeto}"

# ─────────────────────────────────────────
# MongoDB: grava documento do frame
# ─────────────────────────────────────────

def gravar_mongo(col, frame_id, camera_id, turno, imagem_path, deteccoes):
    total_pet     = sum(1 for d in deteccoes if d["classe"] == "PET")
    total_not_pet = sum(1 for d in deteccoes if d["classe"] == "NOT_PET")

    doc = {
        "frame_id"     : frame_id,
        "timestamp"    : datetime.now(timezone.utc).isoformat(),
        "camera_id"    : camera_id,
        "turno"        : turno,
        "imagem_path"  : imagem_path,
        "deteccoes"    : deteccoes,
        "total_pet"    : total_pet,
        "total_not_pet": total_not_pet,
    }
    col.insert_one(doc)
    return doc

# ─────────────────────────────────────────
# InfluxDB: grava métrica agregada
# ─────────────────────────────────────────

def gravar_influx(write_api, camera_id, turno, deteccoes):
    agora = datetime.now(timezone.utc)

    for classe in ["PET", "NOT_PET"]:
        itens = [d for d in deteccoes if d["classe"] == classe]
        if not itens:
            continue
        confianca_media = round(
            sum(d["confianca"] for d in itens) / len(itens), 4
        )
        ponto = (
            Point("deteccao_pet")
            .tag("camera_id", camera_id)
            .tag("turno", turno)
            .tag("classe", classe)
            .field("contagem", len(itens))
            .field("confianca_media", confianca_media)
            .time(agora)
        )
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=ponto)

# ─────────────────────────────────────────
# Pipeline principal — processa uma imagem
# ─────────────────────────────────────────

def processar_frame(
    imagem_path, camera_id, turno,
    yolo, classificador, mongo_col, write_api, minio_client
):
    frame_id = f"{camera_id}_{uuid.uuid4().hex[:8]}"
    imagem   = Image.open(imagem_path).convert("RGB")

    # 1. YOLO detecta regiões
    boxes = detectar_yolo(yolo, imagem_path)

    deteccoes = []

    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            crop = imagem.crop((x1, y1, x2, y2))

            # 2. Classificador confirma PET ou NOT_PET
            classe, confianca = classificar(classificador, crop)

            deteccoes.append({
                "classe"    : classe,
                "confianca" : confianca,
                "conf_yolo" : box["conf_yolo"],
                "bbox"      : box["bbox"],
            })
    else:
        # Sem detecção YOLO — classifica imagem inteira
        classe, confianca = classificar(classificador, imagem)
        deteccoes.append({
            "classe"    : classe,
            "confianca" : confianca,
            "conf_yolo" : None,
            "bbox"      : None,
        })

    # 3. Salva imagem no MinIO
    path_minio = salvar_minio(minio_client, imagem_path, camera_id, frame_id)

    # 4. Grava evento no MongoDB
    doc = gravar_mongo(
        mongo_col, frame_id, camera_id, turno, path_minio, deteccoes
    )

    # 5. Grava métricas no InfluxDB
    gravar_influx(write_api, camera_id, turno, deteccoes)

    total_pet = doc["total_pet"]
    print(f"  ✔ {frame_id} | PET: {total_pet} | detecções: {len(deteccoes)}")
    return doc

# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Worker de ingestão Recifavela")
    parser.add_argument("--imagens", required=True, help="Pasta com imagens a processar")
    parser.add_argument("--camera",  default="cam_01", help="ID da câmera")
    parser.add_argument("--turno",   default="manha",
                        choices=["manha", "tarde", "noite"], help="Turno do processamento")
    args = parser.parse_args()

    pasta = Path(args.imagens)
    if not pasta.exists():
        print(f"[ERRO] Pasta não encontrada: {pasta}")
        return

    extensoes = {".jpg", ".jpeg", ".png"}
    imagens   = [f for f in pasta.iterdir() if f.suffix.lower() in extensoes]

    if not imagens:
        print(f"[ERRO] Nenhuma imagem encontrada em {pasta}")
        return

    print(f"\n{'='*50}")
    print(f"  Worker Recifavela — {len(imagens)} imagens")
    print(f"  Câmera: {args.camera} | Turno: {args.turno}")
    print(f"{'='*50}\n")

    # Inicializa serviços
    print("[Init] Carregando YOLO...")
    yolo = YOLO("yolov8n.pt")   # baixa automaticamente na 1ª execução

    classificador = carregar_modelo_classificador()

    print("[Init] Conectando aos bancos...")
    mongo_client = MongoClient(MONGO_URI)
    mongo_col    = mongo_client[MONGO_DB][MONGO_COL]

    influx_client = InfluxDBClient(
        url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG
    )
    write_api = influx_client.write_api(write_options=SYNCHRONOUS)

    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=False,
    )

    print("[Init] Tudo pronto. Iniciando processamento...\n")

    processadas = 0
    erros       = 0

    for imagem_path in sorted(imagens):
        try:
            processar_frame(
                str(imagem_path), args.camera, args.turno,
                yolo, classificador, mongo_col, write_api, minio_client
            )
            processadas += 1
        except Exception as e:
            print(f"  ✘ Erro em {imagem_path.name}: {e}")
            erros += 1

    # Fecha conexões
    mongo_client.close()
    influx_client.close()

    print(f"\n{'='*50}")
    print(f"  Concluído: {processadas} processadas | {erros} erros")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
