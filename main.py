import pandas as pd
from src.data_prep import clean_and_prepare_data
from src.pre_processing import preprocess_data
from src.pre_processing_smote import preprocess_data_smote
from src.model_training import train_and_evaluate_models
from src.threshold_tuning import test_model_thresholds

print("==========================================")
print("CERVICAL CANCER RISK PREDICTION HARNESS")
print("==========================================\n")

# 1. Define the Grid of Experiments
imputation_methods = ['knn', 'median']

# Define different feature dropping strategies to test
feature_drop_strategies = {
    "Pre_Screen_standard": [
        'STDs', 'STDs:condylomatosis', 'Schiller', 'Hinselmann'
    ],
    "Pre_Screen_STDs": [
        'STDs:condylomatosis', 'Schiller', 'Hinselmann'
    ],
    "Pre_Screen_condy": [
        'STDs', 'Schiller', 'Hinselmann'
    ]
    # "Post_test_STDs": [
    #     'STDs:condylomatosis'
    # ],
    # "Post_test_condy": [
    #     'STDs'
    # ]
}

input_path = "data/risk_factors_cervical_cancer.csv"
all_experiment_results = []

# 2. Execute the Experimental Loop
for imputer in imputation_methods:
    for strategy_name, drop_list in feature_drop_strategies.items():
        
        print(f"\n>>> RUNNING PIPELINE: Imputer={imputer.upper()} | Drops={strategy_name} <<<")
        
        # A. Prepare the specific dataset
        df_clean = clean_and_prepare_data(
            input_path, 
            output_filepath=None, # No need to save to disk during rapid testing
            imputation_method=imputer, 
            verbosity=0, 
            cols_to_drop_extra=drop_list
        )
        
        # B. Test Without SMOTE
        exp_name_base = f"{imputer.upper()}_{strategy_name}_NoSMOTE"
        X_train, X_test, y_train, y_test, _ = preprocess_data(df_clean, verbose=False)
        models = train_and_evaluate_models(X_train, y_train, X_test, y_test, verbose=False)
        
        res_base = test_model_thresholds(models, X_test, y_test, exp_name=exp_name_base)
        all_experiment_results.append(res_base)
        
        # C. Test With SMOTE
        exp_name_smote = f"{imputer.upper()}_{strategy_name}_SMOTE"
        X_train_s, X_test_s, y_train_s, y_test_s, _ = preprocess_data_smote(df_clean, verbose=False)
        models_smote = train_and_evaluate_models(X_train_s, y_train_s, X_test_s, y_test_s, verbose=False)
        
        res_smote = test_model_thresholds(models_smote, X_test_s, y_test_s, exp_name=exp_name_smote)
        all_experiment_results.append(res_smote)

# 3. Compile and Analyze the Master Ledger
master_results = pd.concat(all_experiment_results, ignore_index=True)

# Sort by Recall (to catch the most cancers), then by F1-Score (to minimize false alarms)
master_results = master_results.sort_values(by=['Recall', 'F1_Score'], ascending=[False, False])

print("\n==========================================")
print("TOP 10 BEST MODEL CONFIGURATIONS")
print("==========================================")
print(master_results.head(10).to_string(index=False))

# Save the full ledger for offline analysis
master_results.to_csv("data/master_experiment_ledger.csv", index=False)
print("\n[Success] Full results saved to 'data/master_experiment_ledger.csv'.")