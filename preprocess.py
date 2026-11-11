"""
AI4I 2020 Predictive Maintenance - Preprocessing Script
DS 301 Midterm Project

Run:    python preprocess.py
Input:  data/raw/ai4i2020.csv
Output: train/test CSVs + scaler.pkl in data/processed/
"""
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/raw/ai4i2020.csv"
OUT_DIR = "data/processed"
TARGET_RAW = "Machine failure"
TARGET = "machine_failure"   # name after column cleanup
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
RANDOM_STATE = 42   # fixed so every team member gets the same split
TEST_SIZE = 0.2


def load_data(path):
    df = pd.read_csv(path)
    print(f"[load] shape = {df.shape}")
    print(f"[check] missing values = {df.isna().sum().sum()}, duplicate rows = {df.duplicated().sum()}")
    return df


def check_label_consistency(df):
    """Check rows where 'Machine failure' disagrees with the failure modes (TWF-RNF).
    - Failure mode present but Machine failure = 0 : 18 rows (all RNF)
    - Machine failure = 1 but no failure mode       : 9 rows
    These rows are only reported here, not removed (the team decides how to handle them)."""
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


def encode_type(df):
    # Type has a natural order: L (low quality) < M (medium) < H (high) -> ordinal encoding
    df["type"] = df["type"].map({"L": 0, "M": 1, "H": 2})
    return df


def add_features(df):
    """Domain-based engineered features (optional). Set ADD_FEATURES = False if the paper doesn't use them."""
    df["temp_diff"] = df["process_temperature"] - df["air_temperature"]          # heat dissipation (HDF)
    df["power"] = df["torque"] * df["rotational_speed"] * 2 * 3.141592653589793 / 60  # watts (PWF)
    df["strain"] = df["tool_wear"] * df["torque"]                                  # overstrain (OSF)
    return df


ADD_FEATURES = True


def main():
    df = load_data(RAW_PATH)
    check_label_consistency(df)

    df = clean_columns(df)
    df = encode_type(df)
    if ADD_FEATURES:
        df = add_features(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"[target] failure rate = {y.mean():.2%}  (highly imbalanced)")

    # Stratified split: keeps the ~3.4% failure rate equal in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # Scaling: fit on train only, then transform test (prevents leakage)
    num_cols = [c for c in X.columns if c != "type"]
    scaler = StandardScaler()
    X_train_s, X_test_s = X_train.copy(), X_test.copy()
    X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_s[num_cols] = scaler.transform(X_test[num_cols])

    os.makedirs(OUT_DIR, exist_ok=True)
    X_train_s.to_csv(f"{OUT_DIR}/X_train.csv", index=False)
    X_test_s.to_csv(f"{OUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUT_DIR}/y_test.csv", index=False)
    with open(f"{OUT_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"[split] train = {X_train_s.shape}, test = {X_test_s.shape}")
    print(f"[split] failure rate train = {y_train.mean():.2%}, test = {y_test.mean():.2%}")
    print(f"[save] saved to {OUT_DIR}/")
    # Note: apply oversampling (e.g., SMOTE) only to the training data during the modeling step


if __name__ == "__main__":
    main()
