# Machine Failure Prediction

DS 301 Midterm Project - Team A

## Project overview

This project predicts machine failure using the **AI4I 2020 Predictive Maintenance Dataset** and compares our baseline results with Sengupta, Mehta & Rana (2023).

The three models used in this project are:

- Decision Tree
- K-Nearest Neighbors (KNN, K=5)
- Support Vector Machine (SVM, RBF kernel)

Our current baseline follows the paper's split and preprocessing settings **without SMOTE**. The paper uses SMOTE and additional classifiers, so this is not a complete reproduction.

## Reproduce the results (four steps)

Run these steps **from the project root**. We recommend using the same Python environment for preprocessing and both notebooks.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the data

```bash
python preprocess.py --mode paper
```

This creates the unscaled train/test files under `data/processed/paper/`, including `test_info.csv` for future failure-mode analysis. Do not use `test_info.csv` as a model feature.

### 3. Train the models

Open `notebooks/02_model_training.ipynb` in Jupyter Notebook and select **Restart & Run All**.

Outputs:
- `reports/predictions_paper.csv`
- `reports/knn_cv_results.csv`

### 4. Evaluate the models

Open `notebooks/03_evaluation.ipynb` in Jupyter Notebook and select **Restart & Run All**.

Outputs include:
- `reports/metrics_paper.csv`
- `reports/paper_comparison.csv`
- Figures in `reports/figures/`

## Reproducibility notes

- Data: 10,000 rows; binary target `machine_failure` (339 failures).
- Split: stratified 70/30 train/test, using `random_state=42` (7,000 training and 3,000 test examples).
- `Type` is already encoded in `preprocess.py` in paper mode as **H=0, L=1, M=2**; the training notebook does not encode it again.
- The five sensor features are scaled inside `build_pipeline()`, so the scaler is refitted in each cross-validation fold and does not see test data when fitting.
- The stored experiment uses **no SMOTE**, and ROC-AUC / PR-AUC are not calculated until predicted scores are saved.

## Data and reference

- Dataset: [AI4I 2020 Predictive Maintenance Dataset (UCI)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- Paper: P. Sengupta, A. Mehta and P. S. Rana, *Predictive Maintenance of Armoured Vehicles using Machine Learning Approaches* (2023), [arXiv:2307.14453](https://arxiv.org/abs/2307.14453)
