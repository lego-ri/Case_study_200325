from src.data_prep import clean_and_prepare_data
from src.pre_processing import preprocess_data
from src.pre_processing_smote import preprocess_data_smote
from src.model_training import train_and_evaluate_models
from sklearn.metrics import classification_report
from src.threshold_tuning import test_model_thresholds



print("==========================================")
print("CERVICAL CANCER RISK PREDICTION PIPELINE")
print("==========================================\n")

# 1. Clean the Data
input_path = "data/risk_factors_cervical_cancer.csv"
clean_df = clean_and_prepare_data(input_path, verbose=False) # Turned off to keep terminal readable

print("==========================================")
print("EXPERIMENT A: ALGORITHMIC WEIGHTS (NO SMOTE)")
print("==========================================")
X_train_A, X_test_A, y_train_A, y_test_A, scaler_A = preprocess_data(clean_df, verbose=False)
models_A = train_and_evaluate_models(X_train_A, y_train_A, X_test_A, y_test_A, verbose=True)


print("\n\n==========================================")
print("EXPERIMENT B: SYNTHETIC DATA (WITH SMOTE)")
print("==========================================")
X_train_B, X_test_B, y_train_B, y_test_B, scaler_B = preprocess_data_smote(clean_df, verbose=True)
models_B = train_and_evaluate_models(X_train_B, y_train_B, X_test_B, y_test_B, verbose=True)

# Run the tuner using the models from Experiment B (SMOTE) and their respective test data
test_model_thresholds(models_B, X_test_B, y_test_B, thresholds=[0.50, 0.30, 0.20, 0.15])
# print("\n==========================================")
# print("PHASE 6: THRESHOLD TUNING (LOGISTIC REGRESSION)")
# print("==========================================")

# # 1. Grab our trained Logistic Regression model from Experiment A (the one without SMOTE)
# # We use the non-SMOTE one because it represents the purest mathematical baseline
# # log_reg_model = models_A["Logistic Regression"]
# log_reg_model = models_A["Random Forest"]

# # 2. Get the RAW probabilities (not the final 0 or 1 predictions) for the Test Set
# # [:, 1] grabs the probability specifically for the Positive (1) class
# probabilities = log_reg_model.predict_proba(X_test_A)[:, 1]

# # 3. Define the custom thresholds we want to test
# custom_thresholds = [0.50, 0.30, 0.20, 0.15]

# for threshold in custom_thresholds:
#     print(f"\n--- Testing Threshold: {threshold * 100}% ---")
    
#     # Create new predictions: True (1) if probability >= threshold, else False (0)
#     # We convert the True/False boolean array directly into integers (1s and 0s)
#     custom_predictions = (probabilities >= threshold).astype(int)
    
#     # Print the reality check
#     print(classification_report(y_test_A, custom_predictions, target_names=['Negative (0)', 'Positive (1)']))
#     print("-" * 50)