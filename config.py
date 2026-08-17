"""
Configuração centralizada do projeto Recifavela.
Paths e credenciais dos serviços (podem ser sobrescritas via variáveis de ambiente).
"""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
EXTERNAL_DIR = ROOT_DIR / "data-external"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
IMAG_TEST_DIR = ROOT_DIR / "imag-test"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
CHECKPOINT_PATH = MODELS_DIR / "checkpoint.pth"
SPLIT_MANIFEST_PATH = DATA_DIR / "split_manifest.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# MongoDB
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb://admin:recifavela123@localhost:27017/",
)
MONGO_DB = os.getenv("MONGO_DB", "recifavela")
MONGO_COL = os.getenv("MONGO_COL", "frames")

# InfluxDB
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "recifavela-super-secret-token")
INFLUX_ORG = os.getenv("INFLUX_ORG", "recifavela")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "deteccoes")

# MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS = os.getenv("MINIO_ACCESS", "admin")
MINIO_SECRET = os.getenv("MINIO_SECRET", "recifavela123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "frames")

# ML
CLASSES = ["NOT_PET", "PET"]
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
