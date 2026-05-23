import os
import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN

from data_preprocessing import prepare_data, get_preprocessor

# Set plotting style
sns.set_theme(style="whitegrid")

def evaluate_model(y_true, y_pred, y_prob):
    """
    Calculate and return key classification metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob))
    }
    return metrics

def train_and_select_model():
    # 1. Load and split data
    print("Step 1: Loading and splitting dataset...")
    X_train, X_test, y_train, y_test = prepare_data()
    
    # 2. Get and fit preprocessor
    print("Step 2: Fitting preprocessing pipeline...")
    preprocessor, num_cols, cat_cols = get_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Try to get feature names after transformation
    # OneHotEncoder categories
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_feature_names
    
    # 3. Handle class imbalance using SMOTEENN
    print("Step 3: Handling class imbalance using SMOTEENN...")
    smoteenn = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_transformed, y_train)
    
    print(f"Resampled training set shape: {X_train_resampled.shape}")
    print(f"Resampled target class distribution:\n{y_train_resampled.value_counts(normalize=True)}")
    
    # 4. Compare multiple baseline models
    print("\nStep 4: Training and comparing baseline models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0)
    }
    
    baseline_results = {}
    best_baseline_name = None
    best_baseline_f1 = -1
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        
        # Predict on test set
        y_pred = model.predict(X_test_transformed)
        y_prob = model.predict_proba(X_test_transformed)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_prob)
        baseline_results[name] = metrics
        print(f"    F1-Score: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
        
        if metrics['f1_score'] > best_baseline_f1:
            best_baseline_f1 = metrics['f1_score']
            best_baseline_name = name

    print(f"\nBest baseline model: {best_baseline_name} with F1-Score of {best_baseline_f1:.4f}")
    
    # 5. Hyperparameter Optimization using Optuna for the best model
    print(f"\nStep 5: Optimizing {best_baseline_name} hyperparameters using Optuna...")
    
    # Define optimization target
    def objective(trial):
        if best_baseline_name == "Random Forest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            clf = RandomForestClassifier(**params)
            
        elif best_baseline_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            clf = XGBClassifier(**params)
            
        elif best_baseline_name == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 15, 255),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': 42,
                'verbose': -1
            }
            clf = LGBMClassifier(**params)
            
        elif best_baseline_name == "CatBoost":
            params = {
                'iterations': trial.suggest_int('iterations', 50, 300),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_state': 42,
                'verbose': 0
            }
            clf = CatBoostClassifier(**params)
            
        else: # Fallback / Logistic Regression
            params = {
                'C': trial.suggest_float('C', 0.001, 10.0, log=True),
                'max_iter': 1000,
                'random_state': 42
            }
            clf = LogisticRegression(**params)
            
        clf.fit(X_train_resampled, y_train_resampled)
        y_pred = clf.predict(X_test_transformed)
        # We optimize for F1-score to balance precision and recall
        return f1_score(y_test, y_pred)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    print(f"  Best trial F1-Score: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    
    # 6. Train final optimized model
    print("\nStep 6: Training final optimized model...")
    best_params = study.best_params
    
    if best_baseline_name == "Random Forest":
        final_model = RandomForestClassifier(**best_params, random_state=42)
    elif best_baseline_name == "XGBoost":
        final_model = XGBClassifier(**best_params, eval_metric='logloss', random_state=42)
    elif best_baseline_name == "LightGBM":
        final_model = LGBMClassifier(**best_params, verbose=-1, random_state=42)
    elif best_baseline_name == "CatBoost":
        final_model = CatBoostClassifier(**best_params, verbose=0, random_state=42)
    else:
        final_model = LogisticRegression(**best_params, max_iter=1000, random_state=42)
        
    final_model.fit(X_train_resampled, y_train_resampled)
    
    # 7. Evaluate and save metrics & plots
    os.makedirs("models", exist_ok=True)
    
    y_pred = final_model.predict(X_test_transformed)
    y_prob = final_model.predict_proba(X_test_transformed)[:, 1]
    
    final_metrics = evaluate_model(y_test, y_pred, y_prob)
    print("\n--- Final Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    
    # Save metrics JSON
    all_results = {
        "best_model_name": best_baseline_name,
        "best_hyperparameters": best_params,
        "baseline_comparison": baseline_results,
        "final_metrics": final_metrics
    }
    with open("models/metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    # Generate Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_baseline_name}')
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    plt.close()
    
    # Generate ROC Curve Plot
    plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {final_metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {best_baseline_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("models/roc_curve.png", dpi=150)
    plt.close()
    
    # Feature Importance (if supported)
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Take top 15 features
        top_n = min(15, len(feature_names))
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importances - {best_baseline_name}')
        plt.bar(range(top_n), importances[top_indices], align='center', color='royalblue')
        plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("models/feature_importance.png", dpi=150)
        plt.close()
        print("Feature importance plot saved successfully!")
        
    # 8. Save Model & Preprocessor Pipeline
    # We save a dictionary containing the fitted preprocessor and the best fitted model
    # That way we can easily load them and transform user inputs inside Streamlit
    saved_pipeline = {
        "preprocessor": preprocessor,
        "model": final_model,
        "feature_names": feature_names,
        "model_name": best_baseline_name
    }
    joblib.dump(saved_pipeline, "models/best_churn_model.joblib")
    print(f"\nModel pipeline successfully saved to models/best_churn_model.joblib!")
    
if __name__ == "__main__":
    train_and_select_model()
