import os
import sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# ─── CONFIG ─────────────────────────────────────────────────────────────────
DATASET_PATH = "."
TRAIN_DIR    = os.path.join(DATASET_PATH, "datasets")
MODEL_OUT    = os.path.join(DATASET_PATH, "best_model.keras")

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 40        # EarlyStopping will cut this short if needed
SEED       = 42

# ─── LOAD & SPLIT DATA ──────────────────────────────────────────────────────
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = train_ds_raw.class_names
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# ─── CLASS-WEIGHT COMPUTATION ────────────────────────────────────────────────
label_counts = np.zeros(num_classes, dtype=np.int64)
for _, labels in train_ds_raw.unbatch():
    label_counts[int(labels.numpy())] += 1

total = label_counts.sum()
class_weight = {i: total / (num_classes * count) for i, count in enumerate(label_counts)}
print(f"Class weights: {class_weight}")

# ─── AUGMENTATION PIPELINE ──────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE

augment_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomContrast(0.15),
], name="augmentation")

def cast_float(image, label):
    return tf.cast(image, tf.float32), label

def augment_fn(image, label):
    return augment_layer(image, training=True), label

train_ds = (
    train_ds_raw
    .map(cast_float, num_parallel_calls=AUTOTUNE)
    .cache()
    .shuffle(1000, seed=SEED)
    .map(augment_fn, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds_raw
    .map(cast_float, num_parallel_calls=AUTOTUNE)
    .cache()
    .prefetch(AUTOTUNE)
)

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
print(f"\nLoading existing model from {MODEL_OUT} ...")
model = tf.keras.models.load_model(MODEL_OUT)

# Stage 2 — unfreeze top layers, fine-tune
print("\n===== STAGE 2 — Fine-tuning Top Layers =====")

# Find the base model (EfficientNetV2S) inside the loaded model
base = None
for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        base = layer
        break

if base is not None:
    base.trainable = True
    # Freeze all layers except the last 30
    for layer in base.layers[:-30]:
        layer.trainable = False
else:
    print("Warning: Could not find base model layer by name 'efficientnet'. Fine-tuning the whole model.")
    model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),   # lower LR for fine-tune
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_stage2 = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_accuracy",
                                        save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                      restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                          patience=3, min_lr=1e-7, verbose=1),
]

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_stage2,
    class_weight=class_weight,
    verbose=1
)

# ─── FINAL EVAL ─────────────────────────────────────────────────────────────
print("\n===== Final Evaluation =====")
val_loss, val_acc = model.evaluate(val_ds, verbose=1)
print(f"Val Accuracy : {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"Val Loss     : {val_loss:.4f}")
print(f"\nBest model saved -> {MODEL_OUT}")

# ─── PLOT TRAINING CURVES ───────────────────────────────────────────────────
try:
    acc  = history2.history["accuracy"]
    vacc = history2.history["val_accuracy"]
    loss = history2.history["loss"]
    vloss= history2.history["val_loss"]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Acc"); plt.plot(vacc, label="Val Acc")
    plt.title("Accuracy (Stage 2)"); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss"); plt.plot(vloss, label="Val Loss")
    plt.title("Loss (Stage 2)"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.join(DATASET_PATH, "static", "heatmaps"), exist_ok=True)
    curve_path = os.path.join(DATASET_PATH, "static", "heatmaps", "training_curves_stage2.png")
    plt.savefig(curve_path, dpi=120)
    print(f"Training curves saved → {curve_path}")
except Exception as e:
    print(f"Could not save training curves: {e}")
