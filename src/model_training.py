import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

def train_and_evaluate_models(X_train, y_train, X_test, y_test, verbose=True):
    """
    Trains Logistic Regression, Random Forest, and XGBoost models using class weights.
    Evaluates them using Stratified K-Fold Cross-Validation, then tests them on the hold-out set.
    """
    if verbose:
        print("--- Starting Model Training & Evaluation ---")

    # Calculate the exact imbalance ratio for XGBoost (Negative count / Positive count)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    xgb_scale_weight = neg_count / pos_count

    # 1. Initialize the Models (Option A: Algorithmic Weights)
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=10000),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
        "XGBoost": XGBClassifier(scale_pos_weight=xgb_scale_weight, random_state=42, eval_metric='logloss')
    }

    # 2. Set up the Cross-Validation strategy you suggested
    # 5 splits means it trains on 80% of the train set and tests on 20% of the train set, rotating 5 times
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Dictionary to store the fitted models so we can return them
    trained_models = {}

    for name, model in models.items():
        if verbose:
            print(f"\nEvaluating: {name}...")
            
        # --- Cross-Validation (The Training Phase) ---
        # We use ROC-AUC as the scoring metric during CV because of the imbalance
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
        if verbose:
            print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # --- Final Hold-Out Test (The Reality Check) ---
        # Now we train it on the ENTIRE training set and grade it on the untouched X_test vault
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] # Get probabilities for ROC-AUC
        
        test_auc = roc_auc_score(y_test, y_prob)
        
        if verbose:
            print(f"  Test ROC-AUC: {test_auc:.4f}")
            print("  Classification Report on Hold-Out Test Data:\n")
            print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))
            print("-" * 50)
            
        trained_models[name] = model

    return trained_models