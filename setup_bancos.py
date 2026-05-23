"""
setup_bancos.py
Cria a collection no MongoDB e o bucket no InfluxDB,
e o bucket de imagens no MinIO.
Execute uma vez antes de rodar o worker.

Uso:
    pip install pymongo influxdb-client minio
    python setup_bancos.py
"""

from pymongo import MongoClient, ASCENDING
from influxdb_client import InfluxDBClient
from influxdb_client.client.bucket_api import BucketsService
from minio import Minio
import sys

# ─────────────────────────────────────────
# Configurações (mesmas do docker-compose)
# ─────────────────────────────────────────

MONGO_URI = "MONGODB_URI"
MONGO_DB  = "MONGO_DB"
MONGO_COL = "MONGO_COL"

INFLUX_URL   = "INFLUX_URL"
INFLUX_TOKEN = "INFLUX_TOKEN"
INFLUX_ORG   = "INFLUX_ORG"
INFLUX_BUCKET = "INFLUX_BUCKET"

MINIO_ENDPOINT  = "MINIO_ENDPOINT"
MINIO_ACCESS    = "MINIO_ACCESS"
MINIO_SECRET    = "MINIO_SECRET"
MINIO_BUCKET    = "MINIO_BUCKET"


# ─────────────────────────────────────────
# 1. MongoDB — collection + índices
# ─────────────────────────────────────────

def setup_mongo():
    print("\n[MongoDB] Conectando...")
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]

    # Cria a collection se não existir
    if MONGO_COL not in db.list_collection_names():
        db.create_collection(MONGO_COL)
        print(f"[MongoDB] Collection '{MONGO_COL}' criada.")
    else:
        print(f"[MongoDB] Collection '{MONGO_COL}' já existe.")

    col = db[MONGO_COL]

    # Índices para buscas rápidas por câmera, turno e timestamp
    col.create_index([("timestamp", ASCENDING)])
    col.create_index([("camera_id", ASCENDING)])
    col.create_index([("turno", ASCENDING)])
    print("[MongoDB] Índices criados: timestamp, camera_id, turno")

    # Exemplo do documento que o worker vai gravar:
    print("[MongoDB] Estrutura esperada de cada documento:")
    print("""
    {
        "timestamp": "2026-05-23T08:30:00Z",
        "camera_id": "cam_01",
        "turno": "manha",
        "frame_id": "cam_01_000423",
        "imagem_path": "frames/cam_01/frame_000423.jpg",
        "deteccoes": [
            {
                "classe": "PET",
                "confianca": 0.96,
                "bbox": [x, y, w, h]
            }
        ],
        "total_pet": 2,
        "total_not_pet": 0
    }
    """)
    client.close()
    print("[MongoDB] OK\n")


# ─────────────────────────────────────────
# 2. InfluxDB — confirma bucket
# ─────────────────────────────────────────

def setup_influx():
    print("[InfluxDB] Conectando...")
    client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )

    buckets_api = client.buckets_api()
    buckets = buckets_api.find_buckets().buckets
    nomes = [b.name for b in buckets]

    if INFLUX_BUCKET in nomes:
        print(f"[InfluxDB] Bucket '{INFLUX_BUCKET}' já existe.")
    else:
        buckets_api.create_bucket(bucket_name=INFLUX_BUCKET, org=INFLUX_ORG)
        print(f"[InfluxDB] Bucket '{INFLUX_BUCKET}' criado.")

    print("[InfluxDB] Métricas que o worker vai gravar:")
    print("""
    Measurement : deteccao_pet
    Tags        : camera_id, turno, classe
    Fields      : contagem (int), confianca_media (float)
    Timestamp   : horário do frame
    """)
    client.close()
    print("[InfluxDB] OK\n")


# ─────────────────────────────────────────
# 3. MinIO — bucket de imagens
# ─────────────────────────────────────────

def setup_minio():
    print("[MinIO] Conectando...")
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=False
    )

    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)
        print(f"[MinIO] Bucket '{MINIO_BUCKET}' criado.")
    else:
        print(f"[MinIO] Bucket '{MINIO_BUCKET}' já existe.")

    print(f"[MinIO] Imagens serão salvas em: {MINIO_BUCKET}/camera_id/frame_id.jpg")
    print("[MinIO] OK\n")


# ─────────────────────────────────────────
# Execução
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Setup dos bancos — Recifavela Grupo 1")
    print("=" * 50)

    erros = []

    try:
        setup_mongo()
    except Exception as e:
        erros.append(f"MongoDB: {e}")

    try:
        setup_influx()
    except Exception as e:
        erros.append(f"InfluxDB: {e}")

    try:
        setup_minio()
    except Exception as e:
        erros.append(f"MinIO: {e}")

    if erros:
        print("⚠️  Erros encontrados:")
        for erro in erros:
            print(f"   - {erro}")
        print("\nVerifique se os containers estão rodando: docker compose ps")
        sys.exit(1)
    else:
        print("✅ Tudo configurado! Agora pode rodar o worker de ingestão.")