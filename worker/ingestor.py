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
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from tqdm import tqdm

from ultralytics import YOLO

from pymongo import MongoClient, ASCENDING
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from minio import Minio

# ─────────────────────────────────────────
# Configurações
# ─────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MONGO_URI     = "mongodb://admin:recifavela123@localhost:27017/"
MONGO_DB      = "recifavela"
MONGO_COL     = "frames"

INFLUX_URL    = "http://localhost:8086"
INFLUX_TOKEN  = "recifavela-super-secret-token"
INFLUX_ORG    = "recifavela"
INFLUX_BUCKET = "deteccoes"

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS   = "admin"
MINIO_SECRET   = "recifavela123"
MINIO_BUCKET   = "frames"

MODELO_PATH   = os.path.join(BASE_DIR, "models", "best_model.pth")
CONFIANCA_MIN = 0.5
CLASSES       = ["NOT_PET", "PET"]
BATCH_SIZE    = 16
MAX_RETRIES   = 3

# ─────────────────────────────────────────
# Logger — terminal + arquivo
# ─────────────────────────────────────────

def configurar_logger():
    logs_dir = Path(BASE_DIR) / "logs"
    logs_dir.mkdir(exist_ok=True)

    nome_log = f"ingestor_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"
    log_path = logs_dir / nome_log

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger("ingestor")
    logger.info(f"Log salvo em: {log_path}")
    return logger

# ─────────────────────────────────────────
# Detecção automática de dispositivo
# ─────────────────────────────────────────

def detectar_dispositivo(logger):
    logger.info("Verificando hardware disponível...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        nome   = torch.cuda.get_device_name(0)
        mem    = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✔ GPU detectada: {nome} ({mem:.1f} GB VRAM)")

    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("✔ Apple Silicon (MPS) detectado")

    else:
        device = torch.device("cpu")
        n_threads = torch.get_num_threads()
        logger.info(f"✔ CPU ({n_threads} threads disponíveis)")
        torch.set_num_threads(n_threads)

    return device

# ─────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────

def benchmark(modelo, device, total_imagens, logger, n=10):
    logger.info(f"Rodando {n} inferências de teste em {device}...")
    dummy = torch.rand(1, 3, 224, 224).to(device)

    with torch.no_grad():
        for _ in range(3):
            modelo(dummy)

    inicio = time.perf_counter()
    with torch.no_grad():
        for _ in range(n):
            modelo(dummy)
    fim = time.perf_counter()

    por_imagem = (fim - inicio) / n * 1000
    estimativa = por_imagem * total_imagens / 1000 / 60
    logger.info(f"Tempo médio por imagem : {por_imagem:.1f} ms")
    logger.info(f"Estimativa para {total_imagens} imagens: {estimativa:.1f} minutos")

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
# Carrega modelo PyTorch
# ─────────────────────────────────────────

def carregar_modelo_classificador(device, logger):
    logger.info("Carregando best_model.pth...")
    modelo = models.resnet18(weights=None)
    modelo.fc = torch.nn.Linear(modelo.fc.in_features, 2)
    modelo.load_state_dict(torch.load(MODELO_PATH, map_location=device))
    modelo.to(device)
    modelo.eval()
    logger.info(f"Classificador pronto — rodando em {device}")
    return modelo

# ─────────────────────────────────────────
# Classificador em batch
# ─────────────────────────────────────────

def classificar_batch(modelo, device, crops: list):
    tensors = torch.stack([transform(c) for c in crops]).to(device)
    with torch.no_grad():
        saidas = modelo(tensors)
        probs  = torch.softmax(saidas, dim=1)
        idxs   = probs.argmax(dim=1).tolist()
    return [
        (CLASSES[idx], round(probs[i][idx].item(), 4))
        for i, idx in enumerate(idxs)
    ]

# ─────────────────────────────────────────
# YOLO
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
# MinIO com retry
# ─────────────────────────────────────────

def salvar_minio(minio_client, imagem_path: str, camera_id: str, frame_id: str, logger):
    extensao = Path(imagem_path).suffix
    objeto   = f"{camera_id}/{frame_id}{extensao}"

    for tentativa in range(1, MAX_RETRIES + 1):
        try:
            minio_client.fput_object(MINIO_BUCKET, objeto, imagem_path)
            return f"{MINIO_BUCKET}/{objeto}"
        except Exception as e:
            if tentativa < MAX_RETRIES:
                logger.warning(f"MinIO falhou (tentativa {tentativa}/{MAX_RETRIES}): {e}. Tentando novamente...")
                time.sleep(2 ** tentativa)
            else:
                raise

# ─────────────────────────────────────────
# MongoDB
# ─────────────────────────────────────────

def gravar_mongo(col, frame_id, camera_id, turno, arquivo_original, imagem_path, deteccoes):
    total_pet     = sum(1 for d in deteccoes if d["classe"] == "PET")
    total_not_pet = sum(1 for d in deteccoes if d["classe"] == "NOT_PET")

    doc = {
        "frame_id"        : frame_id,
        "arquivo_original": arquivo_original,
        "timestamp"       : datetime.now(timezone.utc).isoformat(),
        "camera_id"       : camera_id,
        "turno"           : turno,
        "imagem_path"     : imagem_path,
        "deteccoes"       : deteccoes,
        "total_pet"       : total_pet,
        "total_not_pet"   : total_not_pet,
    }
    col.insert_one(doc)
    return doc

# ─────────────────────────────────────────
# InfluxDB
# ─────────────────────────────────────────

def gravar_influx(write_api, camera_id, turno, deteccoes):
    agora = datetime.now(timezone.utc)
    for classe in ["PET", "NOT_PET"]:
        itens = [d for d in deteccoes if d["classe"] == classe]
        if not itens:
            continue
        confianca_media = round(sum(d["confianca"] for d in itens) / len(itens), 4)
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
    yolo, classificador, device,
    mongo_col, write_api, minio_client, logger
):
    arquivo_original = Path(imagem_path).name

    if mongo_col.find_one({"arquivo_original": arquivo_original}):
        return "pulado"

    frame_id = f"{camera_id}_{uuid.uuid4().hex[:8]}"
    imagem   = Image.open(imagem_path).convert("RGB")

    boxes = detectar_yolo(yolo, imagem_path)

    deteccoes = []

    if boxes:
        crops = [imagem.crop(b["bbox"]) for b in boxes]
        resultados = classificar_batch(classificador, device, crops)
        for box, (classe, confianca) in zip(boxes, resultados):
            deteccoes.append({
                "classe"    : classe,
                "confianca" : confianca,
                "conf_yolo" : box["conf_yolo"],
                "bbox"      : box["bbox"],
            })
    else:
        classe, confianca = classificar_batch(classificador, device, [imagem])[0]
        deteccoes.append({
            "classe"    : classe,
            "confianca" : confianca,
            "conf_yolo" : None,
            "bbox"      : None,
        })

    path_minio = salvar_minio(minio_client, imagem_path, camera_id, frame_id, logger)

    doc = gravar_mongo(
        mongo_col, frame_id, camera_id, turno,
        arquivo_original, path_minio, deteccoes
    )

    gravar_influx(write_api, camera_id, turno, deteccoes)
    return doc

# ─────────────────────────────────────────
# Relatório final
# ─────────────────────────────────────────

def imprimir_relatorio(stats, tempo_total, logger):
    minutos = int(tempo_total // 60)
    segundos = int(tempo_total % 60)
    conf_media = (
        round(stats["soma_confianca"] / stats["total_deteccoes"], 4)
        if stats["total_deteccoes"] > 0 else 0
    )

    logger.info("=" * 50)
    logger.info(f"  Processadas : {stats['processadas']}")
    logger.info(f"  Puladas     : {stats['puladas']}")
    logger.info(f"  Erros       : {stats['erros']}")
    logger.info(f"  Total PET detectado     : {stats['total_pet']}")
    logger.info(f"  Total NOT_PET detectado : {stats['total_not_pet']}")
    logger.info(f"  Confiança média geral   : {conf_media * 100:.1f}%")
    logger.info(f"  Tempo total             : {minutos}m {segundos}s")
    logger.info("=" * 50)

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

    logger = configurar_logger()

    pasta = Path(args.imagens)
    if not pasta.exists():
        logger.error(f"Pasta não encontrada: {pasta}")
        return

    extensoes = {".jpg", ".jpeg", ".png"}
    imagens   = [f for f in pasta.iterdir() if f.suffix.lower() in extensoes]

    if not imagens:
        logger.error(f"Nenhuma imagem encontrada em {pasta}")
        return

    logger.info("=" * 50)
    logger.info(f"  Worker Recifavela — {len(imagens)} imagens")
    logger.info(f"  Câmera: {args.camera} | Turno: {args.turno}")
    logger.info("=" * 50)

    device        = detectar_dispositivo(logger)
    yolo          = YOLO("yolov8n.pt")
    if str(device) == "cuda":
        yolo.to("cuda")

    classificador = carregar_modelo_classificador(device, logger)
    benchmark(classificador, device, len(imagens), logger)

    logger.info("Conectando aos bancos...")
    mongo_client = MongoClient(MONGO_URI)
    mongo_col    = mongo_client[MONGO_DB][MONGO_COL]
    mongo_col.create_index(
        "arquivo_original",
        unique=True,
        partialFilterExpression={"arquivo_original": {"$exists": True}}
    )
    mongo_col.create_index([("timestamp", ASCENDING)])
    mongo_col.create_index([("camera_id", ASCENDING)])
    mongo_col.create_index([("turno", ASCENDING)])

    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api     = influx_client.write_api(write_options=SYNCHRONOUS)

    minio_client  = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=False,
    )

    logger.info("Tudo pronto. Iniciando processamento...\n")

    stats = {
        "processadas"    : 0,
        "puladas"        : 0,
        "erros"          : 0,
        "total_pet"      : 0,
        "total_not_pet"  : 0,
        "total_deteccoes": 0,
        "soma_confianca" : 0.0,
    }

    inicio = time.perf_counter()

    with tqdm(sorted(imagens), unit="img", dynamic_ncols=True) as barra:
        for imagem_path in barra:
            barra.set_description(imagem_path.name[:30])
            try:
                resultado = processar_frame(
                    str(imagem_path), args.camera, args.turno,
                    yolo, classificador, device,
                    mongo_col, write_api, minio_client, logger
                )

                if resultado == "pulado":
                    stats["puladas"] += 1
                    barra.set_postfix(status="pulado")
                else:
                    stats["processadas"] += 1
                    stats["total_pet"]       += resultado["total_pet"]
                    stats["total_not_pet"]   += resultado["total_not_pet"]
                    for d in resultado["deteccoes"]:
                        stats["total_deteccoes"] += 1
                        stats["soma_confianca"]  += d["confianca"]
                    barra.set_postfix(
                        pet=stats["total_pet"],
                        erros=stats["erros"]
                    )

            except Exception as e:
                stats["erros"] += 1
                logger.error(f"Erro em {imagem_path.name}: {e}")
                barra.set_postfix(status="erro")

    mongo_client.close()
    influx_client.close()

    tempo_total = time.perf_counter() - inicio
    imprimir_relatorio(stats, tempo_total, logger)


if __name__ == "__main__":
    main()