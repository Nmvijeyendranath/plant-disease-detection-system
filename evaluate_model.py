"""
evaluate_model.py
=================
Evaluates the trained EfficientNetV2S model on the held-out validation split.

Produces:
  • Overall accuracy & loss
  • Per-class precision, recall, F1-score (classification report)
  • Confusion matrix (printed + saved as image)

Run:  python evaluate_model.py
"""

import os
import sys
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"

import tensorflow as tf
import numpy as np

# ─── CONFIG (must match training) ────────────────────────────────────────────
DATASET_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR    = os.path.join(DATASET_PATH, "datasets")
MODEL_PATH   = os.path.join(DATASET_PATH, "best_model.keras")

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42          # same seed as training to get the exact same val split

# ─── CHECK MODEL EXISTS ─────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file '{MODEL_PATH}' not found.")
    print("Please train the model first by running train_cnn_final.py.")
    sys.exit(1)

# ─── LOAD VALIDATION DATA ───────────────────────────────────────────────────
print("Loading validation data …")
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

class_names = val_ds.class_names
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# Cast to float32 (same as training pipeline — no manual normalisation)
def cast_float(image, label):
    return tf.cast(image, tf.float32), label

val_ds = val_ds.map(cast_float, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} …")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.\n")

# ─── EVALUATE (OVERALL METRICS) ─────────────────────────────────────────────
print("=" * 60)
print("  OVERALL MODEL EVALUATION")
print("=" * 60)

val_loss, val_acc = model.evaluate(val_ds, verbose=1)
print(f"\n  Validation Accuracy : {val_acc:.4f}  ({val_acc * 100:.2f}%)")
print(f"  Validation Loss     : {val_loss:.4f}")

# ─── COLLECT ALL PREDICTIONS ────────────────────────────────────────────────
print("\nRunning predictions on all validation images …")
y_true = []
y_pred = []
y_prob = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))
    y_prob.extend(np.max(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

total_samples = len(y_true)
correct = (y_true == y_pred).sum()

# ─── PER-CLASS REPORT ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PER-CLASS CLASSIFICATION REPORT")
print("=" * 60)

# Compute per-class precision, recall, f1 manually (no sklearn dependency)
report_rows = []
for i in range(num_classes):
    tp = ((y_pred == i) & (y_true == i)).sum()
    fp = ((y_pred == i) & (y_true != i)).sum()
    fn = ((y_pred != i) & (y_true == i)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    support   = int((y_true == i).sum())

    report_rows.append((class_names[i], precision, recall, f1, support))

# Print header
header = f"{'Class':<45} {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'Support':>7}"
print(header)
print("-" * len(header))

for name, prec, rec, f1, sup in report_rows:
    print(f"{name:<45} {prec:>9.4f}  {rec:>6.4f}  {f1:>6.4f}  {sup:>7d}")

# Macro / Weighted averages
macro_prec = np.mean([r[1] for r in report_rows])
macro_rec  = np.mean([r[2] for r in report_rows])
macro_f1   = np.mean([r[3] for r in report_rows])

weights    = np.array([r[4] for r in report_rows]) / total_samples
w_prec     = sum(r[1] * r[4] for r in report_rows) / total_samples
w_rec      = sum(r[2] * r[4] for r in report_rows) / total_samples
w_f1       = sum(r[3] * r[4] for r in report_rows) / total_samples

print("-" * len(header))
print(f"{'Macro Average':<45} {macro_prec:>9.4f}  {macro_rec:>6.4f}  {macro_f1:>6.4f}  {total_samples:>7d}")
print(f"{'Weighted Average':<45} {w_prec:>9.4f}  {w_rec:>6.4f}  {w_f1:>6.4f}  {total_samples:>7d}")

# ─── CONFUSION MATRIX ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CONFUSION MATRIX")
print("=" * 60)

cm = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[int(t)][int(p)] += 1

# Short labels for readability
short_names = [n.split("___")[-1][:18] for n in class_names]

# Print matrix header
print(f"\n{'True \\ Pred':<20}", end="")
for sn in short_names:
    print(f" {sn:>18}", end="")
print()
print("-" * (20 + 19 * num_classes))

for i in range(num_classes):
    print(f"{short_names[i]:<20}", end="")
    for j in range(num_classes):
        val = cm[i][j]
        if i == j:
            print(f" \033[92m{val:>18}\033[0m", end="")   # green for diagonal
        elif val > 0:
            print(f" \033[91m{val:>18}\033[0m", end="")   # red for misclass
        else:
            print(f" {val:>18}", end="")
    print()

# ─── CONFIDENCE ANALYSIS ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CONFIDENCE ANALYSIS")
print("=" * 60)

correct_mask = (y_true == y_pred)
wrong_mask   = ~correct_mask

print(f"  Total samples      : {total_samples}")
print(f"  Correct predictions: {correct} ({correct / total_samples * 100:.1f}%)")
print(f"  Wrong predictions  : {total_samples - correct} ({(total_samples - correct) / total_samples * 100:.1f}%)")
print()
print(f"  Avg confidence (correct): {y_prob[correct_mask].mean() * 100:.1f}%" if correct_mask.any() else "  No correct predictions")
print(f"  Avg confidence (wrong)  : {y_prob[wrong_mask].mean() * 100:.1f}%" if wrong_mask.any() else "  No wrong predictions -- perfect!")
print()

# Top misclassifications
if wrong_mask.any():
    print("  Top misclassified pairs (true -> predicted, count):")
    misclass_counts = {}
    for t, p in zip(y_true[wrong_mask], y_pred[wrong_mask]):
        key = (class_names[int(t)], class_names[int(p)])
        misclass_counts[key] = misclass_counts.get(key, 0) + 1

    sorted_misclass = sorted(misclass_counts.items(), key=lambda x: -x[1])
    for (true_name, pred_name), count in sorted_misclass[:10]:
        t_short = true_name.split("___")[-1]
        p_short = pred_name.split("___")[-1]
        print(f"    {t_short:<30} -> {p_short:<30}  ({count})")

# ─── SAVE CONFUSION MATRIX PLOT ─────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix - Validation Set", fontsize=16, fontweight="bold", pad=15)

    tick_marks = np.arange(num_classes)
    display_names = [n.replace("___", "\n").replace("_", " ") for n in class_names]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(display_names, fontsize=8)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i][j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i][j] > thresh else "black",
                    fontsize=9)

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()

    out_dir = os.path.join(DATASET_PATH, "static", "heatmaps")
    os.makedirs(out_dir, exist_ok=True)
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    print(f"Saving confusion matrix to {cm_path} ...")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n  Confusion matrix saved -> {cm_path}")
except Exception as e:
    print(f"\n  Could not save confusion matrix plot: {e}")

# ─── SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Model          : {MODEL_PATH}")
print(f"  Dataset        : {TRAIN_DIR}")
print(f"  Val Split      : 20% (seed={SEED})")
print(f"  Total Val Images: {total_samples}")
print(f"  Overall Accuracy: {val_acc * 100:.2f}%")
print(f"  Macro F1       : {macro_f1:.4f}")
print(f"  Weighted F1    : {w_f1:.4f}")
print("=" * 60)
