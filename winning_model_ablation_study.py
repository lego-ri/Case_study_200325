import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Assuming your custom modules are still accessible
from src.data_prep import clean_and_prepare_data
from src.pre_processing import preprocess_data

print("==========================================")
print("ABLATION STUDY: CHAMPION XGBOOST vs. NO COLPOSCOPY")
print("==========================================\n")

# 1. The Ablated Feature Space
# We take the champion drop list and surgically remove the colposcopy tests
ablation_drop_list = [
    'STDs:condylomatosis', 
    'STDs: Number of diagnosis', 
    'Hinselmann', 
    'Schiller'
]
input_path = "data/risk_factors_cervical_cancer.csv"

# 2. Extract and Clean Data
df_clean = clean_and_prepare_data(
    input_path, 
    output_filepath=None, 
    imputation_method='median', 
    verbosity=0, 
    cols_to_drop_extra=ablation_drop_list
)

# 3. Preprocess without SMOTE
X_train, X_test, y_train, y_test, _ = preprocess_data(df_clean, use_smote=False, verbose=False)

# Re-extract true feature names to prevent metadata loss
true_feature_names = df_clean.drop('Biopsy', axis=1).columns.tolist()
X_train = pd.DataFrame(X_train, columns=true_feature_names)
X_test = pd.DataFrame(X_test, columns=true_feature_names)

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_weight = neg_count / pos_count

# 4. Instantiate the Champion XGBoost Architecture
champion_xgb_ablated = XGBClassifier(
    scale_pos_weight=scale_weight, 
    random_state=42, 
    eval_metric='logloss',
    max_depth=5,          
    learning_rate=0.1, 
    n_estimators=100,
    n_jobs=1
)

# 5. Train and Evaluate
print("--- Training Ablated Model ---")
champion_xgb_ablated.fit(X_train, y_train)

# Calculate predictions on the Hold-Out Test set
y_prob = champion_xgb_ablated.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.50).astype(int) # Standard 0.50 threshold for Post-Test parity

test_auc = roc_auc_score(y_test, y_prob)

print(f"\n>>> ABLATED TEST ROC-AUC: {test_auc:.4f} <<<")
print("(Compare this against the original 0.9238)\n")

print("--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)'], zero_division=0))