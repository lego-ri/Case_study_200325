import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Assuming your custom modules are still accessible in the environment
from src.data_prep import clean_and_prepare_data
from src.pre_processing import preprocess_data

print("==========================================")
print("CLINICAL ORACLE: RANDOM FOREST FEATURE IMPORTANCE")
print("==========================================\n")

# 1. Recreate the exact champion environment (Median Imputation, Standard Drops, No SMOTE)
# Update this drop_list to match exactly what "Post_test_standard" meant in your pipeline
drop_list = [
    'STDs', 'STDs:condylomatosis', 'STDs: Number of diagnosis'
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

# Preprocess without SMOTE
X_train, X_test, y_train, y_test, _ = preprocess_data(df_clean, use_smote=False, verbose=False)

# Convert to Pandas DataFrames using our extracted true names to prevent metadata loss
X_train = pd.DataFrame(X_train, columns=true_feature_names)
X_test = pd.DataFrame(X_test, columns=true_feature_names)

# Recombine for the final oracle training (Training on all data for the final feature importance extraction)
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])

# 3. Instantiate and train the Champion Random Forest Model
# Using your grid search winners: max_depth=None, min_samples_split=5, n_estimators=100
champion_rf = RandomForestClassifier(
    class_weight='balanced', # Crucial for Imbalanced Medical Data
    random_state=42, 
    max_depth=None,           
    min_samples_split=5, 
    n_estimators=100,
    n_jobs=-1
)

print("--- Training Champion Random Forest on Full Data Matrix ---")
champion_rf.fit(X_full, y_full)

# 4. Extract the Mathematical Importance (Mean Decrease in Impurity)
importances = champion_rf.feature_importances_

# Bind the true biological names to the calculated Importance
fi_df = pd.DataFrame({
    'Feature': true_feature_names,
    'Importance (Gini)': importances
})

# Filter out features that the model mathematically ignored (Importance == 0) 
# and sort descending to ensure we grab the absolute highest values
fi_df = fi_df[fi_df['Importance (Gini)'] > 0].sort_values(by='Importance (Gini)', ascending=False)

# Isolate the Top 10 most important features
top_n = 10 
fi_df_top = fi_df.head(top_n)

# --- THE FIX FOR PRESENTATION ---
# Reverse the order of our Top 10 subset so the absolute highest value plots at the bottom
fi_df_top = fi_df_top.sort_values(by='Importance (Gini)', ascending=True)


# 5. Plot the Biological Hierarchy
plt.figure(figsize=(10, 6))

sns.barplot(
    data=fi_df_top, 
    x='Importance (Gini)', 
    y='Feature', 
    hue='Feature',      # Resolves the deprecation warning
    legend=False,       # Prevents redundant legend
    palette='viridis'   # Matching your EDA 'viridis' color scheme
)

plt.title(f"Top {top_n} Random Forest Predictors for Positive Biopsy", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=12)
plt.ylabel("Clinical Feature", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Print the top risk factors in standard descending order for your own reference
print(f"\n--- Top {top_n} Absolute Risk Factors (Random Forest) ---")
print(fi_df.head(top_n).to_string(index=False))