import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Assuming your custom modules are still accessible in the environment
from src.data_prep import clean_and_prepare_data
from src.pre_processing import preprocess_data

print("==========================================")
print("CLINICAL ORACLE: XGBOOST FEATURE IMPORTANCE")
print("==========================================\n")

# 1. Recreate the exact champion environment (Post_test_STDs)
drop_list = ['STDs','STDs:condylomatosis', 'STDs: Number of diagnosis',
            #   'Schiller', 'Hinselmann'
              ]
input_path = "data/risk_factors_cervical_cancer.csv"

# Suppress output to focus on the final graph
df_clean = clean_and_prepare_data(
    input_path, 
    output_filepath=None, 
    imputation_method='median', 
    verbosity=0, 
    cols_to_drop_extra=drop_list
)

# 2. Extract TRUE feature names directly from the clean DataFrame
# We drop 'Biopsy' because it is the target, leaving only the clinical predictors
true_feature_names = df_clean.drop('Biopsy', axis=1).columns.tolist()

# Preprocess without SMOTE (ignoring the broken feature_names output)
X_train, X_test, y_train, y_test, _ = preprocess_data(df_clean, use_smote=False, verbose=False)

# Convert to Pandas DataFrames using our extracted true names to prevent metadata loss
X_train = pd.DataFrame(X_train, columns=true_feature_names)
X_test = pd.DataFrame(X_test, columns=true_feature_names)

# Recombine for the final oracle training
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])

neg_count = (y_full == 0).sum()
pos_count = (y_full == 1).sum()
scale_weight = neg_count / pos_count

# 3. Instantiate and train the Champion Model
champion_xgb = XGBClassifier(
    scale_pos_weight=scale_weight, 
    random_state=42, 
    eval_metric='logloss',
    max_depth=5,          
    learning_rate=0.1, 
    n_estimators=100,
    n_jobs=1
)

print("--- Training Champion Model on Full Data Matrix ---")
champion_xgb.fit(X_full, y_full)

# 4. Extract the Mathematical Gain
importances = champion_xgb.feature_importances_

# Bind the true biological names to the calculated Information Gain
fi_df = pd.DataFrame({
    'Feature': true_feature_names,
    'Importance (Gain)': importances
})

# Filter out features that the model mathematically ignored (Gain == 0)
fi_df = fi_df[fi_df['Importance (Gain)'] > 0].sort_values(by='Importance (Gain)', ascending=False)

# 5. Plot the Biological Hierarchy (with Seaborn warning fixed)
plt.figure(figsize=(14, 10))
sns.barplot(
    data=fi_df, 
    x='Importance (Gain)', 
    y='Feature', 
    hue='Feature',      # Resolves the deprecation warning
    legend=False,       # Prevents a massive, redundant legend from rendering
    palette='magma'
)

plt.title('XGBoost Target Feature Importances (Clinical Risk Hierarchy)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Relative Information Gain', fontsize=14)
plt.ylabel('Clinical Feature', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\n--- Top 5 Absolute Risk Factors ---")
print(fi_df.head(5).to_string(index=False))