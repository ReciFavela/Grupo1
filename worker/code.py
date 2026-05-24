import cv2
import time
import uuid

from src.predict import predict_image
from src.storage import save_to_storage
from src.metrics import send_metrics

import os
from dotenv import load_dotenv

load_dotenv()

print("INFLUX:", os.getenv("INFLUX_URL"))

print("Worker started (PET classifier)...")

cap = cv2.VideoCapture(0)

temp_dir = os.path.join(os.getcwd(), "temp")
os.makedirs(temp_dir, exist_ok=True)

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}.jpg")
    
    success = cv2.imwrite(temp_path, frame)

    if not success:
        print("Failed to write frame")
        continue

    label, confidence = predict_image(temp_path)

    result = {
        "class": label,
        "confidence": confidence
    }

    save_to_storage(frame, result)
    send_metrics(result)

    print("Processed:", result)

    time.sleep(1)