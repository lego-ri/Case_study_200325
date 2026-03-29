import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings

def train_and_evaluate_models(X_train, y_train, X_test, y_test, apply_smote=False, verbose=True):
    """
    Executes Nested Cross-Validation with dynamic pipeline execution.
    - Inner Loop: Hyperparameter tuning via GridSearchCV (n_jobs=1 to prevent deadlock).
    - Outer Loop: Unbiased performance estimation (n_jobs=-2 for parallel speed).
    - Pipeline: Ensures SMOTE is only applied to training folds, preventing data leaks.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    if verbose:
        mode = "(WITH SMOTE Pipeline)" if apply_smote else "(STANDARD)"
        print(f"--- Starting Nested CV Model Training {mode} ---")

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    xgb_scale_weight = 1 if apply_smote else (neg_count / pos_count)

    # 1. Define Models and Hyperparameter Grids
    model_configs = {
        "Logistic Regression": {
            "model": LogisticRegression(class_weight='balanced', random_state=42, max_iter=10000, solver='saga'),
            "params": {
                'classifier__penalty': ['elasticnet'], # combines L1 (Lasso) and L2 (Ridge) penalties
                'classifier__C': [0.01, 0.1, 1, 10], # Inverse of regularization strength
                'classifier__l1_ratio': [0.0, 0.5, 1.0] # L1/L2 ratio
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-2),
            "params": {
                'classifier__n_estimators': [100, 300], # Size of the forest
                'classifier__max_depth': [None, 5, 10], # Maximum depth (consecutive questions)
                'classifier__min_samples_split': [2, 5] # Minimum number of patients in a node before a split is permitted
            }
        },
        "XGBoost": {
            "model": XGBClassifier(scale_pos_weight=xgb_scale_weight, random_state=42, eval_metric='logloss', n_jobs=-2),
            "params": {
                'classifier__n_estimators': [100, 200], # Depth of the "forest" (trees in sequence)
                'classifier__learning_rate': [0.01, 0.1],
                'classifier__max_depth': [3, 5] # Maximum depth (consecutive questions)
            }
        },
        "Support Vector Machine": {
            "model": SVC(class_weight='balanced', probability=True, random_state=42),
            "params": {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            }
        }
    }

    # 2. Define Cross-Validation Geometry
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    trained_models = {}

    for name, config in model_configs.items():
        if verbose:
            print(f"\n>>> Executing Nested CV for: {name} <<<")
            
        # 3. Construct the Mathematical Pipeline
        steps = []
        if apply_smote:
            steps.append(('smote', SMOTE(random_state=42)))
        steps.append(('classifier', config['model']))
        
        pipeline = ImbPipeline(steps)

        # 4. Inner Loop: Hyperparameter Search
        clf = GridSearchCV(
            estimator=pipeline, 
            param_grid=config['params'], 
            cv=inner_cv, 
            scoring='roc_auc', 
            n_jobs=1  # MUST remain 1 to prevent multiprocessing deadlock with the outer loop
        )

        # 5. Outer Loop: Unbiased Evaluation
        cv_scores = cross_val_score(clf, X_train, y_train, cv=outer_cv, scoring='roc_auc', n_jobs=-2)
        
        if verbose:
            print(f"  Outer CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # 6. Final Fit
        clf.fit(X_train, y_train)
        
        # INJECT THE CV SCORE INTO THE OBJECT FOR DOWNSTREAM LOGGING
        clf.cv_auc_score = cv_scores.mean()
        
        if verbose:
            print(f"  Best Tuned Parameters: {clf.best_params_}")

        # 7. Final Hold-Out Test Evaluation
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] 
        test_auc = roc_auc_score(y_test, y_prob)
        
        if verbose:
            print(f"  Hold-Out Test ROC-AUC: {test_auc:.4f}")
            print("  Classification Report (Hold-Out Data):\n")
            print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)'], zero_division=0))
            print("-" * 60)
            
        trained_models[name] = {
                    'model': clf,
                    'cv_auc': float(cv_scores.mean())
                }
        
    return trained_models