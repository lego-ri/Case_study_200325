import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def test_model_thresholds(trained_models, X_test, y_test, exp_name, thresholds=[0.15, 0.25, 0.35, 0.50]):
    """
    Iterates through a dictionary of trained models and evaluates their performance 
    across multiple probability decision thresholds, now capturing hyperparameters.
    """
    results = []

    for name, model_data in trained_models.items():
        clf = model_data['model']
        cv_auc = model_data['cv_auc']
        
        # Rigorously extract hyperparameters (falling back to N/A if it wasn't tuned)
        best_params = str(clf.best_params_) if hasattr(clf, 'best_params_') else "N/A"
        
        # Clean up the 'classifier__' prefix for cleaner CSV logging
        best_params = best_params.replace("'classifier__", "'")
        
        y_prob = clf.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_prob)
        
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            
            rec = recall_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)

            results.append({
                'Experiment': exp_name,
                'Model': name,
                'Threshold': thresh,
                'CV_AUC': round(cv_auc, 4),
                'Test_AUC': round(test_auc, 4),
                'Recall': round(rec, 4),
                'Precision': round(prec, 4),
                'F1_Score': round(f1, 4),
                'Parameters': best_params # INJECTED HERE
            })
    return pd.DataFrame(results)