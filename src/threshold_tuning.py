import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def test_model_thresholds(trained_models, X_test, y_test, exp_name, thresholds=[0.15, 0.25, 0.35, 0.50]):
    """
    Iterates through a dictionary of trained models and evaluates their performance 
    across multiple probability decision thresholds.
    """
    results = []

    for name, model_data in trained_models.items():
        clf = model_data['model']
        cv_auc = model_data['cv_auc']
        
        # Calculate continuous probability for the positive class (Cancer)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        # Calculate the single overarching Test ROC-AUC
        test_auc = roc_auc_score(y_test, y_prob)
        
        # Test discrete clinical thresholds
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
                'F1_Score': round(f1, 4)
            })
    return pd.DataFrame(results)