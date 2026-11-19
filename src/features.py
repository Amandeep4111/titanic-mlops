"""Cleaning + feature engineering shared by notebook, main.py and the API."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FINAL_COLS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "IsAlone", "Title"]
TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
COMMON_TITLES = ["Mr", "Mrs", "Miss", "Master"]


class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    """Raw Titanic rows -> clean rows with FamilySize, IsAlone, Title.

    * drops Cabin / Ticket / PassengerId / Name (Name only used to extract Title)
    * Age & Fare -> median, Embarked -> mode (learned in fit)
    * accepts raw rows (with Name) or already-processed rows (with Title)
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.age_median_ = float(pd.to_numeric(X["Age"], errors="coerce").median())
        self.fare_median_ = float(pd.to_numeric(X["Fare"], errors="coerce").median())
        self.embarked_mode_ = X["Embarked"].mode(dropna=True).iloc[0]
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if "Title" not in X.columns:
            title = X["Name"].astype(str).str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
            title = title.replace(TITLE_MAP)
            X["Title"] = title.where(title.isin(COMMON_TITLES), "Rare")
        X["Pclass"] = X["Pclass"].astype(int)
        X["SibSp"] = pd.to_numeric(X["SibSp"], errors="coerce").fillna(0).astype(int)
        X["Parch"] = pd.to_numeric(X["Parch"], errors="coerce").fillna(0).astype(int)
        X["Age"] = pd.to_numeric(X["Age"], errors="coerce").fillna(self.age_median_)
        X["Fare"] = pd.to_numeric(X["Fare"], errors="coerce").fillna(self.fare_median_)
        X["Embarked"] = X["Embarked"].fillna(self.embarked_mode_)
        if "FamilySize" not in X.columns:
            X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        if "IsAlone" not in X.columns:
            X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        return X[FINAL_COLS]
