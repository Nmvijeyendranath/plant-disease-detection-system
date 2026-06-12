"""
train_cnn_final.py
==================
Trains an EfficientNetV2S model for crop leaf disease classification.
Supports corn (maize), potato, and tomato disease categories.

Expected folder layout:
  datasets/
    Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
    Corn_(maize)___Common_rust_/
    Corn_(maize)___Northern_Leaf_Blight/
    Corn_(maize)___healthy/
    Potato___Early_blight/
    Potato___Late_blight/
    Potato___healthy/
    Tomato___Bacterial_spot/
    ... (10 tomato classes)

Run:  python train_cnn_final.py
"""

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
# Count samples per class to handle any imbalance
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

# ─── MODEL ──────────────────────────────────────────────────────────────────
# Stage 1 — freeze base, train head only
base = tf.keras.applications.EfficientNetV2S(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
x       = base(inputs, training=False)
x       = tf.keras.layers.GlobalAveragePooling2D()(x)
x       = tf.keras.layers.BatchNormalization()(x)
x       = tf.keras.layers.Dense(256, activation="relu",
                                 kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x       = tf.keras.layers.Dropout(0.45)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs, name="EfficientNetV2S_PlantDisease")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ─── CALLBACKS ──────────────────────────────────────────────────────────────
callbacks_stage1 = [
    tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_accuracy",
                                        save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                      restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-6, verbose=1),
]

print("\n===== STAGE 1 — Head Training =====")
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks_stage1,
    class_weight=class_weight,
    verbose=1
)

# Stage 2 — unfreeze top layers, fine-tune
print("\n===== STAGE 2 — Fine-tuning Top Layers =====")
base.trainable = True
# Freeze all layers except the last 30
for layer in base.layers[:-30]:
    layer.trainable = False

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
print(f"Class order used by model: {class_names}")
print("WARNING: Make sure CLASS_NAMES in app.py and predict.py match this order!")

# ─── PLOT TRAINING CURVES ───────────────────────────────────────────────────
try:
    acc  = history1.history["accuracy"]  + history2.history["accuracy"]
    vacc = history1.history["val_accuracy"] + history2.history["val_accuracy"]
    loss = history1.history["loss"]      + history2.history["loss"]
    vloss= history1.history["val_loss"]  + history2.history["val_loss"]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Acc"); plt.plot(vacc, label="Val Acc")
    plt.title("Accuracy"); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss"); plt.plot(vloss, label="Val Loss")
    plt.title("Loss"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.join(DATASET_PATH, "static", "heatmaps"), exist_ok=True)
    curve_path = os.path.join(DATASET_PATH, "static", "heatmaps", "training_curves.png")
    plt.savefig(curve_path, dpi=120)
    print(f"Training curves saved → {curve_path}")
except Exception as e:
    print(f"Could not save training curves: {e}")
