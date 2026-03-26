import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

def train_and_evaluate_models(X_train, y_train, X_test, y_test, verbose=True):
    """
    Trains Logistic Regression, Random Forest, XGBoost, SVM, and Neural Network models.
    Evaluates them using Stratified K-Fold Cross-Validation, then tests on the hold-out set.
    """
    if verbose:
        print("--- Starting Model Training & Evaluation ---")

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    xgb_scale_weight = neg_count / pos_count

    # 1. Initialize ALL Models
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=10000),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=1000),
        "XGBoost": XGBClassifier(scale_pos_weight=xgb_scale_weight, random_state=42, eval_metric='logloss'),
        # SVM requires probability=True to calculate ROC-AUC and do threshold tuning later
        "Support Vector Machine": SVC(class_weight='balanced', probability=True, random_state=42),
        # Neural Network (2 hidden layers: 64 neurons, then 32 neurons)
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    trained_models = {}

    for name, model in models.items():
        if verbose:
            print(f"\nEvaluating: {name}...")
            
        # Cross-Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        if verbose:
            print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Final Hold-Out Test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] 
        
        test_auc = roc_auc_score(y_test, y_prob)
        
        if verbose:
            print(f"  Test ROC-AUC: {test_auc:.4f}")
            print("  Classification Report on Hold-Out Test Data:\n")
            print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)'], zero_division=0))
            print("-" * 50)
            
        trained_models[name] = model

    return trained_models