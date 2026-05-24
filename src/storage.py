from pymongo import MongoClient
from minio import Minio
from datetime import datetime
import uuid
import cv2

import os
BASE_DIR = os.getcwd()
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(TEMP_DIR, exist_ok=True)

from src.config import MONGO_URL, MONGO_DB

# ---------------- MONGO ----------------
mongo_client = MongoClient(MONGO_URL)
db = mongo_client[MONGO_DB]
collection = db["detections"]

# ---------------- MINIO ----------------
minio = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="password123",
    secure=False
)

BUCKET = "frames"


def save_to_storage(frame, result):

    image_name = f"{uuid.uuid4()}.jpg"
    temp_path = os.path.join(TEMP_DIR, image_name)

     # SAVE FRAME
    success = cv2.imwrite(temp_path, frame)

    if not success:
        print("❌ Failed to write image")
        return

    # UPLOAD TO MINIO
    minio.fput_object(BUCKET, image_name, temp_path)

    # MongoDB
    doc = {
        "camera": "cam_01",
        "timestamp": datetime.utcnow(),
        "result": result,
        "image": image_name
    }

    collection.insert_one(doc)