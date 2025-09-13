# evaluate.py
import os
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# -------------------
# File paths
# -------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.json")

# -------------------
# Load model + data
# -------------------
with open(METRICS_PATH, "r") as f:
    metrics_info = json.load(f)

target_col = metrics_info["target"]

# Load dataset
df = pd.read_csv(DATA_PATH)
y = df[target_col].astype(int)
X = df.drop(columns=[target_col])

# Load model
model = joblib.load(MODEL_PATH)

# -------------------
# Get predictions
# -------------------
proba = model.predict_proba(X)[:, 1]

# -------------------
# ROC Curve
# -------------------
fpr, tpr, _ = roc_curve(y, proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig(os.path.join(ARTIFACT_DIR, "roc_curve.png"), dpi=150)
plt.close()

# -------------------
# Precision-Recall Curve
# -------------------
prec, rec, _ = precision_recall_curve(y, proba)
pr_auc = auc(rec, prec)

plt.figure()
plt.plot(rec, prec, label=f"PR curve (AUC = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.savefig(os.path.join(ARTIFACT_DIR, "pr_curve.png"), dpi=150)
plt.close()

print(f"ROC curve saved to: {os.path.join(ARTIFACT_DIR, 'roc_curve.png')}")
print(f"PR curve saved to: {os.path.join(ARTIFACT_DIR, 'pr_curve.png')}")