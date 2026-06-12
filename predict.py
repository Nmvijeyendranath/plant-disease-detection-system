"""
predict.py — Standalone prediction script
Usage: python predict.py path/to/leaf_image.jpg
"""

import sys
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "best_model.keras"

from app import CLASS_NAMES, DISEASE_INFO

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: '{MODEL_PATH}' not found. Run train_cnn_final.py first.")
        sys.exit(1)
    print("Loading model …")
    return tf.keras.models.load_model(MODEL_PATH)

def predict(img_path: str, model):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    preds = model.predict(arr, verbose=0)[0]
    class_id   = int(np.argmax(preds))
    confidence = float(preds[class_id])
    disease    = CLASS_NAMES[class_id]

    if "healthy" in disease.lower():
        severity = "None"
    elif confidence > 0.85:
        severity = "High"
    elif confidence > 0.65:
        severity = "Medium"
    else:
        severity = "Low"

    return disease, confidence, severity

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"ERROR: Image '{img_path}' not found.")
        sys.exit(1)

    m = load_model()
    disease, conf, severity = predict(img_path, m)

    print("\n" + "="*40)
    print("  PLANT DISEASE DIAGNOSIS REPORT")
    print("="*40)
    print(f"  Image     : {img_path}")
    print(f"  Disease   : {disease.replace('_', ' ').title()}")
    print(f"  Confidence: {conf*100:.1f}%")
    print(f"  Severity  : {severity}")
    print()
    print("  ADVICE:")
    for line in DISEASE_INFO[disease]["treatment"].splitlines():
        print(f"  {line}")
    print("="*40)

    # Show all class probabilities
    m2 = load_model()
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    preds = m2.predict(arr, verbose=0)[0]
    print("\n  Class Probabilities:")
    for name, prob in zip(CLASS_NAMES, preds):
        bar = "█" * int(prob * 30)
        print(f"  {name:<15} {prob*100:5.1f}%  {bar}")
