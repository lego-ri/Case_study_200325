import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, recall_score, precision_score

def test_model_thresholds(trained_models_dict, X_test, y_test, thresholds=[0.50, 0.30, 0.20, 0.15]):
    """
    Takes a dictionary of trained models and tests them against custom probability thresholds.
    Focuses specifically on the Recall and Precision of the Positive (1) class.
    """
    print("\n==========================================")
    print("PHASE 6: MULTI-MODEL THRESHOLD TUNING")
    print("==========================================")

    for name, model in trained_models_dict.items():
        print(f"\n--- Model: {name} ---")
        
        # Get the RAW probabilities for the Positive class
        probabilities = model.predict_proba(X_test)[:, 1]
        
        for threshold in thresholds:
            # Create new True/False predictions based on the custom threshold
            custom_predictions = (probabilities >= threshold).astype(int)
            
            # We specifically extract the metrics for the Positive class (index 1)
            # zero_division=0 prevents warnings if the model predicts 0 positive cases
            recall = recall_score(y_test, custom_predictions, pos_label=1, zero_division=0)
            precision = precision_score(y_test, custom_predictions, pos_label=1, zero_division=0)
            
            # Format the output for easy reading
            print(f"Threshold {threshold*100:2.0f}% | Recall (Caught Cancer): {recall:.2f} | Precision (True Alarms): {precision:.2f}")