
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import joblib
from pipeline import build_preprocess
from utils import ensure_dir

def prepare_data(df, target_col):
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].map({'Yes':1,'No':0}).fillna(df[target_col])
    return df

def main(args):
    outdir = Path(args.outdir); ensure_dir(outdir)
    models_dir = Path('models'); ensure_dir(models_dir)

    # load with optional sampling for speed
    df = pd.read_csv(args.data)
    if args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=42).reset_index(drop=True)
    df = prepare_data(df, args.target)

    X = df.drop(columns=[args.target])
    y = df[args.target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()==2 else None
    )

    pre, _ = build_preprocess(df, args.target)
    model = LogisticRegression(max_iter=500, n_jobs=None)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= args.threshold).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    # metrics
    (outdir/"metrics.json").write_text(json.dumps({
        "best_model": "logreg",
        "metrics": {"logreg":{"roc_auc":float(auc),"f1":float(f1),"best_params":{}}}
    }, indent=2))

    # permutation importance on a small subsample
    n = min(len(X_test), args.imp_rows)
    r = permutation_importance(pipe, X_test.iloc[:n], y_test.iloc[:n], n_repeats=2, random_state=42)
    feature_names = list(X_test.columns)
    imp_df = pd.DataFrame({"feature":feature_names,"importance_mean":r.importances_mean,"importance_std":r.importances_std})\
        .sort_values("importance_mean", ascending=False)
    imp_df.to_csv(outdir/"feature_importance.csv", index=False)

    # predictions
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_prob"] = y_prob
    pred_df["y_pred"] = y_pred
    pred_df.to_csv(outdir/"predictions.csv", index=False)
    # save trained pipeline for API
    import joblib
    joblib.dump(pipe, models_dir/"best_model.pkl")

    # text report
    cm = confusion_matrix(y_test, y_pred)
    cls = classification_report(y_test, y_pred, digits=4, zero_division=0)
    (outdir/"report.txt").write_text(
        f"Model: logreg\\nROC-AUC: {auc:.4f}\\n\\nConfusion Matrix:\\n{cm}\\n\\nClassification Report:\\n{cls}\\n"
    )
    print(f"Done. ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--target", type=str, default="Churn")
    ap.add_argument("--outdir", type=str, default="output")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--max_rows", type=int, default=4000, help="Sample size cap for speed (0 = no cap)")
    ap.add_argument("--imp_rows", type=int, default=400, help="Rows used for permutation importance")
    args = ap.parse_args()
    main(args)
