#!/usr/bin/env python
"""Single-command MLOps pipeline:  python main.py [--version v1|v2] [--config configs/config.yaml]
raw CSV -> validation -> preprocessing -> training -> evaluation -> MLflow tracking + registry."""
import argparse
import json
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
import sklearn
import yaml
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.features import TitanicPreprocessor                                     # noqa: E402
from src.pipeline import build_pipeline, evaluate, get_selected_features, save_confusion_matrix  # noqa: E402


def banner(msg):
    print(f"\n{'=' * 8} {msg} {'=' * 8}")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ 1. validation
def stage_validate(cfg):
    banner("1/5 DATA VALIDATION")
    d = cfg["data"]
    path = Path(d["raw_path"])
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    df = pd.read_csv(path)
    errors, checks = [], {}
    missing = [c for c in d["required_columns"] if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
    else:
        checks["rows"] = len(df)
        if len(df) < d["min_rows"]:
            errors.append(f"only {len(df)} rows (< {d['min_rows']})")
        if not set(df[d["target"]].dropna().unique()) <= {0, 1} or df[d["target"]].isna().any():
            errors.append("target must be binary 0/1 without nulls")
        if df["PassengerId"].duplicated().any():
            errors.append("duplicate PassengerId values")
        for col, thr in d["max_null_fraction"].items():
            frac = float(df[col].isna().mean())
            checks[f"null_frac_{col}"] = round(frac, 4)
            if frac > thr:
                errors.append(f"{col}: {frac:.2%} nulls > {thr:.0%}")
        if not df["Pclass"].isin([1, 2, 3]).all():
            errors.append("Pclass outside {1,2,3}")
        if not df["Sex"].isin(["male", "female"]).all():
            errors.append("Sex has unexpected values")
        if (df["Fare"].dropna() < 0).any() or not df["Age"].dropna().between(0, 120).all():
            errors.append("Age/Fare out of range")
    report = {"passed": not errors, "errors": errors, "checks": checks}
    Path(cfg["artifacts"]["validation_path"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise ValueError("Data validation failed: " + "; ".join(errors))
    return df


# ------------------------------------------------------------------ 2. preprocessing
def stage_preprocess(cfg, raw, version):
    banner(f"2/5 PREPROCESSING ({version})")
    t, seed = cfg["data"]["target"], cfg["split"]["random_state"]
    proc = TitanicPreprocessor().fit(raw).transform(raw)           # impute, drop cols, FamilySize/IsAlone/Title
    proc[t] = raw[t].values
    pool, test = train_test_split(proc, test_size=cfg["split"]["test_size"], stratify=proc[t], random_state=seed)
    frac = cfg["versions"][version]["train_fraction"]
    train = pool if frac >= 1 else train_test_split(pool, train_size=frac, stratify=pool[t], random_state=seed)[0]
    train, test = train.sort_index(), test.sort_index()
    out = Path(cfg["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    print(f"train rows={len(train)} (fraction {frac}) | test rows={len(test)} | columns={list(train.columns)}")
    return train, test


# ------------------------------------------------------------------ 3. training
def stage_train(cfg, version, train):
    banner("3/5 TRAINING")
    t = cfg["data"]["target"]
    tree = cfg["versions"][version]["tree"]
    pipe = build_pipeline(tree, cfg["feature_selection"]["alpha"])
    pipe.fit(train.drop(columns=t), train[t])
    sel = get_selected_features(pipe)
    print("tree params:", tree)
    print("chi2-selected features:", sel)
    Path(cfg["artifacts"]["model_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, cfg["artifacts"]["model_path"])
    return pipe, tree, sel


# ------------------------------------------------------------------ 4. evaluation
def stage_evaluate(cfg, version, pipe, train, test, tree, selected):
    banner("4/5 EVALUATION")
    t = cfg["data"]["target"]
    m = evaluate(pipe, test.drop(columns=t), test[t])
    m["train_accuracy"] = float(pipe.score(train.drop(columns=t), train[t]))
    a = cfg["artifacts"]
    Path(a["metrics_path"]).write_text(json.dumps(m, indent=2), encoding="utf-8")
    save_confusion_matrix(m["confusion_matrix"], a["confusion_matrix_path"], f"Decision Tree {version} - test")
    info = {"model_name": cfg["mlflow"]["registered_model_name"], "version": version,
            "algorithm": "DecisionTreeClassifier", "params": tree,
            "chi2_alpha": cfg["feature_selection"]["alpha"], "selected_features": selected,
            "input_columns": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Name"],
            "n_train": len(train), "n_test": len(test),
            "metrics": {k: round(v, 4) for k, v in m.items() if k != "confusion_matrix"},
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__, "python_version": platform.python_version()}
    Path(a["model_info_path"]).write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info["metrics"], indent=2))
    return m, info


# ------------------------------------------------------------------ 5. MLflow
def stage_track(cfg, version, pipe, m, info, config_path):
    banner("5/5 MLFLOW TRACKING + REGISTRY")
    import mlflow
    import mlflow.sklearn
    cfg_m = cfg["mlflow"]
    mlflow.set_tracking_uri(cfg_m["tracking_uri"])
    mlflow.set_experiment(cfg_m["experiment_name"])
    with mlflow.start_run(run_name=f"decision_tree_{version}") as run:
        mlflow.set_tags({"data_version": version, "algorithm": "DecisionTreeClassifier"})
        params = {"data_version": version, "scaler": "MinMaxScaler", "chi2_alpha": info["chi2_alpha"],
                  "n_train": info["n_train"], "n_test": info["n_test"],
                  "n_selected_features": len(info["selected_features"]),
                  **{f"tree_{k}": v for k, v in info["params"].items()}}
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, float)})
        a = cfg["artifacts"]
        mlflow.log_artifact(a["confusion_matrix_path"], artifact_path="plots")
        mlflow.log_artifact(a["metrics_path"], artifact_path="reports")
        mlflow.log_artifact(config_path, artifact_path="config")
        kwargs = dict(sk_model=pipe, registered_model_name=cfg_m["registered_model_name"],
                      serialization_format=cfg_m.get("serialization_format", "cloudpickle"),
                      pip_requirements=[f"scikit-learn=={sklearn.__version__}", "pandas", "numpy", "cloudpickle"])
        try:
            mlflow.sklearn.log_model(**kwargs, name="model")              # MLflow >= 3
        except TypeError:
            mlflow.sklearn.log_model(**kwargs, artifact_path="model")     # MLflow 2.x
        print("MLflow run id:", run.info.run_id)


def main():
    ap = argparse.ArgumentParser(description="Titanic Decision-Tree MLOps pipeline")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--version", default=None, choices=["v1", "v2"], help="overrides run.version in the config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    version = args.version or cfg["run"]["version"]
    print(f"Pipeline version: {version} | config: {args.config}")

    raw = stage_validate(cfg)
    train, test = stage_preprocess(cfg, raw, version)
    pipe, tree, selected = stage_train(cfg, version, train)
    m, info = stage_evaluate(cfg, version, pipe, train, test, tree, selected)
    try:
        stage_track(cfg, version, pipe, m, info, args.config)
    except Exception:                       # tracking problems must not lose the trained model
        traceback.print_exc()
        print("WARNING: MLflow tracking failed (model + metrics were still saved).")
    banner("PIPELINE FINISHED")


if __name__ == "__main__":
    main()
