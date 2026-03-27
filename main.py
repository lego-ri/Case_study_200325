from src.data_prep import clean_and_prepare_data
from src.pre_processing import preprocess_data
from src.pre_processing_smote import preprocess_data_smote
from src.model_training import train_and_evaluate_models
from sklearn.metrics import classification_report
from src.threshold_tuning import test_model_thresholds
import sys



print("==========================================")
print("CERVICAL CANCER RISK PREDICTION PIPELINE")
print("==========================================\n")

# 1. Clean the Data
# Input and output paths
input_path = "data/risk_factors_cervical_cancer.csv"
output_path_knn = "data/clean_cervical_cancer_data_knn.csv"
output_path_median = "data/clean_cervical_cancer_data_median.csv"
# Redundant and leaking columns
redundant_and_leaking_cols = [
        'STDs',                         # Redundant to STDs (number)
        'STDs:condylomatosis',          # Redundant master column
        'Schiller',                     # Target leakage
        'Hinselmann'                    # Target leakage
]
# Generate both versions to test downstream ML model sensitivity
clean_df_knn = clean_and_prepare_data(input_path, output_path_knn, imputation_method='knn', verbosity=0, cols_to_drop_extra=redundant_and_leaking_cols)
clean_df_median = clean_and_prepare_data(input_path, output_path_median, imputation_method='median', verbosity=0, cols_to_drop_extra=redundant_and_leaking_cols)

print("==========================================")
print("EXPERIMENT A: ALGORITHMIC WEIGHTS (NO SMOTE)")
print("==========================================")
X_train_A, X_test_A, y_train_A, y_test_A, scaler_A = preprocess_data(clean_df_knn, verbose=False)
models_A = train_and_evaluate_models(X_train_A, y_train_A, X_test_A, y_test_A, verbose=True)


print("\n\n==========================================")
print("EXPERIMENT B: SYNTHETIC DATA (WITH SMOTE)")
print("==========================================")
X_train_B, X_test_B, y_train_B, y_test_B, scaler_B = preprocess_data_smote(clean_df_knn, verbose=True)
models_B = train_and_evaluate_models(X_train_B, y_train_B, X_test_B, y_test_B, verbose=True)

print("\n==========================================")
print("TUNING EXPERIMENT A: NO SMOTE (Algorithmic Weights)")
print("==========================================")
test_model_thresholds(models_A, X_test_A, y_test_A, thresholds=[0.50, 0.30, 0.20, 0.15])

print("\n==========================================")
print("TUNING EXPERIMENT B: WITH SMOTE (Synthetic Data)")
print("==========================================")
test_model_thresholds(models_B, X_test_B, y_test_B, thresholds=[0.50, 0.30, 0.20, 0.15])
