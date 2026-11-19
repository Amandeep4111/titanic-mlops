"""Model pipeline: preprocess -> One-Hot + MinMax -> Chi-Square selection -> Decision Tree."""
import matplotlib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFpr, chi2
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .features import TitanicPreprocessor

NUM_COLS = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
CAT_COLS = ["Pclass", "Sex", "Embarked", "Title"]
BIN_COLS = ["IsAlone"]


def build_pipeline(tree_params, chi2_alpha=0.05):
    encode = ColumnTransformer(
        [("num", MinMaxScaler(), NUM_COLS),                       # MinMax -> non-negative (needed by chi2)
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
         ("bin", "passthrough", BIN_COLS)],
        verbose_feature_names_out=False)
    params = {"random_state": 42, **tree_params}
    return Pipeline([("prep", TitanicPreprocessor()),
                     ("encode", encode),
                     ("select", SelectFpr(chi2, alpha=chi2_alpha)),   # Chi-Square test, keep p < alpha
                     ("clf", DecisionTreeClassifier(**params))])


def chi2_table(pipe):
    names = pipe.named_steps["encode"].get_feature_names_out()
    sel = pipe.named_steps["select"]
    return (pd.DataFrame({"feature": names, "chi2": sel.scores_, "p_value": sel.pvalues_, "selected": sel.get_support()})
            .sort_values("chi2", ascending=False).reset_index(drop=True))


def get_selected_features(pipe):
    names = pipe.named_steps["encode"].get_feature_names_out()
    return list(names[pipe.named_steps["select"].get_support()])


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    return {"accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "confusion_matrix": confusion_matrix(y, pred).tolist()}


def save_confusion_matrix(cm, path, title="Confusion matrix"):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(pd.DataFrame(cm).values, display_labels=["Died", "Survived"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
