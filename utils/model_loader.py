import os
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "best_model.keras")

model = None

def get_model():
    global model

    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                model = tf.keras.models.load_model(MODEL_PATH)
                print("[INFO] Model loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
        else:
            print(f"[WARNING] Model file '{MODEL_PATH}' not found.")

    return model