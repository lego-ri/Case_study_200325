import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data_smote(df, target_col='Biopsy', test_size=0.2, random_state=42, verbose=True):
    """
    Separates features/target, performs stratified split, applies SMOTE to the 
    TRAINING data only, and scales features using StandardScaler.
    """
    if verbose:
        print("--- Starting ML Pre-Processing Pipeline (WITH SMOTE) ---")

    # 1. Define Features (X) and Target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 3. Apply SMOTE to Training Data ONLY
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    if verbose:
        print("[Step 1] Performed Stratified Split.")
        print(f"[Step 2] Applied SMOTE. Original Train Positives: {y_train.sum()} -> New Train Positives: {y_train_smote.sum()}")
        print(f"         - New Train size: {X_train_smote.shape[0]} patients (Perfect 50/50 balance)")
        print(f"         - Test size (UNTOUCHED): {X_test.shape[0]} patients ({y_test.sum()} positive)")

    # 4. Feature Scaling
    scaler = StandardScaler()
    
    # Fit strictly on the NEW SMOTE training data, then transform both
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_smote), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    if verbose:
        print("[Step 3] Applied StandardScaler to features.")
        print("--- SMOTE Pre-Processing Complete ---\n")

    return X_train_scaled, X_test_scaled, y_train_smote, y_test, scaler