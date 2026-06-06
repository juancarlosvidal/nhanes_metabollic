# train_models_publication_ready.py

"""
NHANES Metabolic Syndrome - Publication-Grade ML Pipeline

Features:
- 7 model configurations (paper-aligned)
- Logistic Regression, Random Forest, XGBoost
- Proper preprocessing (scaling + encoding)
- Stratified Train /  Test split (680/20)
- 5-fold Cross-Validation
- Survey weights (WTMEC2YR)
- ROC Curves (saved)
- Full metrics export
"""
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ===================== CONFIG =====================
INPUT_FILE = r"..\output_data\nhanes_with_metabolic_syndrome_adults.csv"
RESULT_DIR = r"..\output_result"

os.makedirs(RESULT_DIR, exist_ok=True)
for filename in os.listdir(RESULT_DIR):
    file_path = os.path.join(RESULT_DIR, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # delete file
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # delete folder
    except Exception as e:
        print(f"Failed to delete {file_path}: {e}")
TARGET = 'metabolic_syndrome'
WEIGHTS = 'WTMEC2YR'

# ===================== MODELS =====================
models_map = {
    "Model 1": ['RIDAGEYR', 'RIAGENDR'],
    "Model 2": ['BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS'],
    "Model 3": ['RIDAGEYR', 'RIAGENDR', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS'],
    "Model 4": ['RIDAGEYR', 'RIAGENDR', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBXSCH', 'LBXSTR'],
    "Model 5": ['RIDAGEYR', 'RIAGENDR', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBXSCH', 'LBXSTR', 'LBXGH'],
    "Model 6": ['RIDAGEYR', 'RIAGENDR', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBXSCH', 'LBXSTR', 'LBXGLU'],
    "Model 7": ['RIDAGEYR', 'RIAGENDR', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST', 'BPXDI1', 'BPXSY1', 'BPXPLS', 'LBXSCH', 'LBXSTR', 'LBXGLU', 'LBXGH']
}

# ===================== MAIN PIPELINE =====================
def run_pipeline():
    df = pd.read_csv(INPUT_FILE)

    all_results = []

    for model_name, features in models_map.items():
        print(f"\nProcessing {model_name}...")

        df_clean = df.dropna(subset=features + [TARGET, WEIGHTS])

        X = df_clean[features]
        y = df_clean[TARGET]
        w = df_clean[WEIGHTS]

        # ================= SPLIT (80/20) =================
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, w, test_size=0.20, stratify=y, random_state=42
        )


        # ================= PREPROCESSING =================
        num_features = X.select_dtypes(include=np.number).columns.tolist()
        cat_features = ['RIAGENDR'] if 'RIAGENDR' in X.columns else []

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first'), cat_features)
        ])

        # ================= CLASSIFIERS =================

        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                n_jobs=-1,
                random_state=42,
                eval_metric='logloss'
            )
        }

        for algo_name, model in classifiers.items():
            print(f"Training {algo_name}...")

            pipe = Pipeline([
                ('prep', preprocessor),
                ('model', model)
            ])

            # ================= CROSS-VALIDATION =================
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []

            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                w_tr, w_val = w_train.iloc[train_idx], w_train.iloc[val_idx]

                pipe.fit(X_tr, y_tr, model__sample_weight=w_tr)
                y_val_prob = pipe.predict_proba(X_val)[:, 1]

                score = roc_auc_score(y_val, y_val_prob, sample_weight=w_val)
            cv_scores.append(score)

            cv_auc_mean = np.mean(cv_scores)
            # ================= TRAIN =================
            pipe.fit(X_train, y_train, model__sample_weight=w_train)

            # ================= EVALUATION =================
            y_prob = pipe.predict_proba(X_test)[:, 1]
            y_pred = pipe.predict(X_test)

            auc = roc_auc_score(y_test, y_prob, sample_weight=w_test)
            acc = accuracy_score(y_test, y_pred, sample_weight=w_test)
            prec = precision_score(y_test, y_pred, sample_weight=w_test)
            rec = recall_score(y_test, y_pred, sample_weight=w_test)
            f1 = f1_score(y_test, y_pred, sample_weight=w_test)
            ce = log_loss(y_test, y_prob, sample_weight=w_test)

            # ================= ROC CURVE =================
            fpr, tpr, _ = roc_curve(y_test, y_prob, sample_weight=w_test)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            plt.plot([0,1],[0,1],'--')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC - {model_name} - {algo_name}")
            plt.legend()
            plt.savefig(os.path.join(RESULT_DIR, f"ROC_{model_name}_{algo_name}.png"), dpi=300)
            plt.close()

            all_results.append({
                "Model": model_name,
                "Algorithm": algo_name,
                "AUC": round(auc,4),
                "Accuracy": round(acc,4),
                "Precision": round(prec,4),
                "Recall": round(rec,4),
                "F1": round(f1,4),
                "LogLoss": round(ce,4),
                "CV_AUC_Mean": round(cv_auc_mean.mean(),4)
            })
    ## Final Results
    results_df = pd.DataFrame(all_results)
    results_df.sort_values(by="AUC", ascending=False)
    plt.figure(figsize=(12,6))
    sns.barplot(data=results_df, x='Model', y='AUC', hue='Algorithm')
    plt.title("AUC Comparison (Weighted)")
    plt.ylim(0.5, 1)
    plt.savefig(os.path.join(RESULT_DIR, f"ROC_{model_name}_{algo_name}_auc.png"), dpi=300)

    plt.close()
    # ================= SAVE RESULTS =================
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{RESULT_DIR}/final_results.csv", index=False)
    results_df.to_latex(f"{RESULT_DIR}/final_results.tex", index=False)
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()
# %%
