# train.py
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# -------------------
# File paths
# -------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Target column options (dataset dependent)
TARGET_CANDIDATES = ["Class", "isFraud", "Fraud", "label", "is_fraud"]


# -------------------
# Load the dataset
# -------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target is None:
        raise ValueError(f"Target column not found. Available: {list(df.columns)}")

    y = df[target].astype(int)
    X = df.drop(columns=[target])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    return X, y, num_cols, cat_cols, target


# -------------------
# Build model pipelines
# -------------------
def build_pipelines(num_cols, cat_cols, use_smote=False):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ]
    )

    lr = LogisticRegression(max_iter=3000, class_weight="balanced")
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    if use_smote:
        lr_pipe = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=42)), ("clf", lr)])
        rf_pipe = ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=42)), ("clf", rf)])
    else:
        lr_pipe = Pipeline([("pre", pre), ("clf", lr)])
        rf_pipe = Pipeline([("pre", pre), ("clf", rf)])

    return {"logreg": lr_pipe, "rf": rf_pipe}


# -------------------
# Evaluation function
# -------------------
def evaluate(model, Xte, yte):
    proba = model.predict_proba(Xte)[:, 1]
    yhat = model.predict(Xte)

    roc = roc_auc_score(yte, proba)
    pr = average_precision_score(yte, proba)
    report = classification_report(yte, yhat, zero_division=0)
    cm = confusion_matrix(yte, yhat)

    return roc, pr, report, cm


# -------------------
# Main execution
# -------------------
def main():
    X, y, num_cols, cat_cols, target = load_data()

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = build_pipelines(num_cols, cat_cols, use_smote=False)
    results = {}
    best_model_name, best_score, best_model = None, -1, None

    for name, pipe in models.items():
        pipe.fit(Xtr, ytr)
        roc, pr, report, cm = evaluate(pipe, Xte, yte)
        results[name] = {"roc_auc": roc, "pr_auc": pr, "report": report, "confusion_matrix": cm.tolist()}
        print(f"{name}: ROC-AUC={roc:.4f} | PR-AUC={pr:.4f}")
        if pr > best_score:
            best_score, best_model_name, best_model = pr, name, pipe

    # Try with SMOTE
    models_smote = build_pipelines(num_cols, cat_cols, use_smote=True)
    for name, pipe in models_smote.items():
        nm = name + "_smote"
        pipe.fit(Xtr, ytr)
        roc, pr, report, cm = evaluate(pipe, Xte, yte)
        results[nm] = {"roc_auc": roc, "pr_auc": pr, "report": report, "confusion_matrix": cm.tolist()}
        print(f"{nm}: ROC-AUC={roc:.4f} | PR-AUC={pr:.4f}")
        if pr > best_score:
            best_score, best_model_name, best_model = pr, nm, pipe

    # Save best model
    joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "model.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump({"results": results, "best_model": best_model_name, "best_pr_auc": best_score, "target": target}, f,
                  indent=2)

    print("\nBest model:", best_model_name)
    print("Saved model & metrics in artifacts/")


if __name__ == "__main__":
    main()