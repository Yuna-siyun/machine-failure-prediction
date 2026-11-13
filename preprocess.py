"""
AI4I 2020 Predictive Maintenance - Preprocessing Script
DS 301 Midterm Project

Two modes:
  paper    : reproduce the paper  -> 70/30 split, MinMaxScaler, original features only,
                                     Type encoded as in the paper (H=0, L=1, M=2)
  improved : our extended version -> 80/20 split, StandardScaler, + 3 engineered features,
                                     Type encoded by quality order (L=0, M=1, H=2)

Run:
  python preprocess.py                  # default: paper mode
  python preprocess.py --mode improved

Input:  data/ai4i2020.csv
Output: data/processed/<mode>/
          cleaned.csv                         full cleaned dataset (unscaled, for inspection)
          X_train.csv, X_test.csv             UNSCALED features
          y_train.csv, y_test.csv             target
          test_info.csv                       UDI, Product ID and failure modes (TWF-RNF) of each
                                              test row, same row order as X_test.csv.
                                              For error analysis ONLY - never use as features.

Scaling is NOT applied to the saved files. To avoid leakage during cross-validation,
the scaler must be fitted inside each CV fold. Use build_pipeline() in the modeling code:

  from preprocess import build_pipeline
  pipe = build_pipeline(LogisticRegression(max_iter=1000), mode="paper")
  cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5), scoring="f1")

To reproduce the paper's SMOTE step, pass smote=True:

  pipe = build_pipeline(DecisionTreeClassifier(), mode="paper", smote=True)

SMOTE then runs inside the pipeline (scaler -> SMOTE -> model), so it is applied only to
the training data in fit() and to the training folds in cross-validation. predict() on
the test set never resamples anything.
"""
import argparse
import os

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

RAW_PATH = "data/ai4i2020.csv"
OUT_ROOT = "data/processed"
TARGET_RAW = "Machine failure"
TARGET = "machine_failure"   # name after column cleanup
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
RANDOM_STATE = 42   # fixed so every team member gets the same split

TYPE_ENCODINGS = {
    "paper":   {"H": 0, "L": 1, "M": 2},   # as in the paper (alphabetical, like LabelEncoder)
    "ordinal": {"L": 0, "M": 1, "H": 2},   # quality order: low < medium < high
}

CONFIGS = {
    "paper":    {"test_size": 0.3, "scaler": MinMaxScaler,   "add_features": False,
                 "type_encoding": "paper"},
    "improved": {"test_size": 0.2, "scaler": StandardScaler, "add_features": True,
                 "type_encoding": "ordinal"},
}


def load_data(path):
    df = pd.read_csv(path)
    print(f"[load] shape = {df.shape}")
    print(f"[check] missing values = {df.isna().sum().sum()}, duplicate rows = {df.duplicated().sum()}")
    return df


def check_label_consistency(df):
    """Report rows where 'Machine failure' disagrees with the failure modes (TWF-RNF).
    - Failure mode present but Machine failure = 0 : 18 rows (all RNF)
    - Machine failure = 1 but no failure mode       : 9 rows
    These rows are kept (not removed); the team decides how to handle them."""
    n_modes = df[FAILURE_MODES].sum(axis=1)
    a = ((df[TARGET_RAW] == 0) & (n_modes > 0)).sum()
    b = ((df[TARGET_RAW] == 1) & (n_modes == 0)).sum()
    print(f"[check] failure mode but failure=0: {a} rows / failure=1 but no mode: {b} rows")


def clean_columns(df):
    # 1) Drop identifiers: UDI (row number) and Product ID (serial number) carry no predictive info
    df = df.drop(columns=["UDI", "Product ID"])

    # 2) Drop failure-mode columns: they directly reveal the target -> data leakage
    df = df.drop(columns=FAILURE_MODES)

    # 3) Clean column names: remove brackets/units and spaces (XGBoost fails on '[' ']' in names)
    df.columns = (df.columns.str.replace(r"\s*\[.*?\]", "", regex=True)
                            .str.strip().str.replace(" ", "_").str.lower())
    return df


def encode_type(df, encoding="ordinal"):
    """Encode product Type with a fixed rule (not learned from data -> no leakage).
    paper  : H=0, L=1, M=2  (matches the paper)
    ordinal: L=0, M=1, H=2  (quality order)"""
    df["type"] = df["type"].map(TYPE_ENCODINGS[encoding])
    return df


def add_features(df):
    """Domain-based engineered features (improved mode only)."""
    df["temp_diff"] = df["process_temperature"] - df["air_temperature"]          # heat dissipation (HDF)
    df["power"] = df["torque"] * df["rotational_speed"] * 2 * 3.141592653589793 / 60  # watts (PWF)
    df["strain"] = df["tool_wear"] * df["torque"]                                  # overstrain (OSF)
    return df


def build_pipeline(model, mode="paper", smote=False, random_state=RANDOM_STATE):
    """Scaler (+ optional SMOTE) + model in one pipeline.

    When passed to cross_val_score / GridSearchCV, the scaler (and SMOTE) are re-fitted
    on the training folds only, so validation folds never leak into preprocessing.
    'type' is label-encoded and passed through without scaling.

    smote=False -> scikit-learn Pipeline:  scaler -> model
    smote=True  -> imblearn Pipeline:      scaler -> SMOTE -> model
                   SMOTE only resamples during fit(); predict() / predict_proba()
                   on validation or test data are never resampled.
                   SMOTE runs after scaling because it uses k-nearest-neighbor distances."""
    scaler_cls = CONFIGS[mode]["scaler"]
    preprocessor = ColumnTransformer(
        transformers=[("scale", scaler_cls(), make_column_selector(pattern=r"^(?!type$)"))],
        remainder="passthrough",
    )
    if smote:
        return ImbPipeline([("preprocess", preprocessor),
                            ("smote", SMOTE(random_state=random_state)),
                            ("model", model)])
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def preprocess(mode="paper"):
    cfg = CONFIGS[mode]
    out_dir = f"{OUT_ROOT}/{mode}"
    print(f"[mode] {mode}: test_size={cfg['test_size']}, scaler={cfg['scaler'].__name__}, "
          f"add_features={cfg['add_features']}, type_encoding={cfg['type_encoding']}")

    raw = load_data(RAW_PATH)
    check_label_consistency(raw)

    df = clean_columns(raw)
    df = encode_type(df, cfg["type_encoding"])
    if cfg["add_features"]:
        df = add_features(df)

    # Export the full cleaned dataset (before split/scaling) so it can be inspected directly
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/cleaned.csv", index=False)
    print(f"[save] cleaned data -> {out_dir}/cleaned.csv  shape = {df.shape}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"[target] failure rate = {y.mean():.2%}  (highly imbalanced)")

    # Stratified split: keeps the ~3.4% failure rate equal in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=RANDOM_STATE)

    # Saved UNSCALED: scaling happens inside build_pipeline() during model training
    X_train.to_csv(f"{out_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{out_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv", index=False)

    # Row IDs and failure modes of the test rows, for error analysis only (not features).
    # The split keeps the original row index, so we can look the rows up in the raw data.
    test_info = raw.loc[X_test.index, ["UDI", "Product ID"] + FAILURE_MODES]
    test_info.to_csv(f"{out_dir}/test_info.csv", index=False)

    print(f"[split] train = {X_train.shape}, test = {X_test.shape}")
    print(f"[split] failure rate train = {y_train.mean():.2%}, test = {y_test.mean():.2%}")
    print(f"[save] saved to {out_dir}/")
    # Note: apply oversampling (e.g., SMOTE) only inside the training folds during modeling
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=CONFIGS.keys(), default="paper",
                        help="paper = reproduce the paper, improved = our extended version")
    args = parser.parse_args()
    preprocess(args.mode)
