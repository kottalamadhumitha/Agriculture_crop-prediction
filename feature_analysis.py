"""
Data drill: correlations, RandomForest importances, permutation importance.

Run from project root:
  ./venv/bin/python feature_analysis.py

Uses cleaned_crop_data.csv and the same engineered columns as train_model.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from features import (
    ALL_FEATURE_NAMES,
    BASE_FEATURE_NAMES,
    add_engineered_features,
    print_feature_justification,
)


def main() -> None:
    df = pd.read_csv("cleaned_crop_data.csv")
    df.columns = df.columns.str.strip().str.lower()
    df = df[BASE_FEATURE_NAMES + ["label"]]

    print("=== Dataset ===")
    print(df["label"].value_counts().describe())
    print("rows:", len(df))

    print("\n=== Base feature correlations (|r| >= 0.25) ===")
    c = df[BASE_FEATURE_NAMES].corr().abs()
    pairs = (
        c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    for (a, b), r in pairs[pairs >= 0.25].items():
        print(f"  {a:12s} vs {b:12s}  r={r:.3f}")

    Xb = add_engineered_features(df.drop(columns=["label"]))
    X = Xb[ALL_FEATURE_NAMES].to_numpy(float)
    y = df["label"]

    imputer = SimpleImputer(strategy="mean")
    X_i = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_i)

    X_train, X_test, y_train, y_test = train_test_split(
        X_s, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=20, random_state=42, n_jobs=1
    )
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print("\n=== Hold-out accuracy (engineered features) ===")
    print(f"  {acc:.4f}")

    print("\n=== Gini importance (mean decrease impurity) ===")
    order = np.argsort(clf.feature_importances_)[::-1]
    for j in order:
        name = ALL_FEATURE_NAMES[j]
        print(f"  {name:22s} {clf.feature_importances_[j]:.4f}")

    print("\n=== Permutation importance (held-out, n_repeats=8) ===")
    perm = permutation_importance(
        clf, X_test, y_test, n_repeats=8, random_state=42, n_jobs=1
    )
    order_p = np.argsort(perm.importances_mean)[::-1]
    for j in order_p:
        name = ALL_FEATURE_NAMES[j]
        mean = perm.importances_mean[j]
        sd = perm.importances_std[j]
        print(f"  {name:22s} mean={mean:+.4f} ± {sd:.4f}")

    if acc >= 0.999 and perm.importances_mean.max() < 0.02:
        print(
            "\n  (Note: With near-perfect accuracy, many features look redundant "
            "to the forest; permutation drops can read as ~0 even when MDI ranks "
            "differ. Use correlation structure + MDI rank together.)"
        )

    print("\n=== How we justify each feature (base + engineered) ===")
    print_feature_justification()

    print("Notes:")
    print(
        "  • Interpret importances relatively (rank), not as causal effects."
    )
    print(
        "  • Engineered columns encode clusters hinted by correlations "
        "(e.g. K–rainfall, N–temperature, P–humidity)."
    )


if __name__ == "__main__":
    main()
