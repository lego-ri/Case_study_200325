import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score

def test_model_thresholds(trained_models_dict, X_test, y_test, exp_name, thresholds=[0.50, 0.30, 0.20, 0.15]):
    """
    Tests models against custom probability thresholds and returns a structured DataFrame of results.
    """
    results = []
    
    for model_name, model in trained_models_dict.items():
        # Get the RAW probabilities for the Positive class
        probabilities = model.predict_proba(X_test)[:, 1]
        
        for threshold in thresholds:
            custom_predictions = (probabilities >= threshold).astype(int)
            
            # zero_division=0 prevents NaN math errors if the model predicts 0 positive cases
            recall = recall_score(y_test, custom_predictions, pos_label=1, zero_division=0)
            precision = precision_score(y_test, custom_predictions, pos_label=1, zero_division=0)
            
            # Calculate F1 Score manually to ensure zero_division safety
            f1 = 0.0
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            # Append to our structured log
            results.append({
                'Experiment': exp_name,
                'Model': model_name,
                'Threshold': threshold,
                'Recall': recall,
                'Precision': precision,
                'F1_Score': f1
            })
            
    return pd.DataFrame(results)