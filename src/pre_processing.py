import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(df, target_col='Biopsy', test_size=0.2, random_state=42, use_smote=False, verbose=True):
    """
    Separates features and target, performs a stratified train/test split, 
    scales the continuous features, and optionally applies SMOTE to the training set.
    
    CRITICAL ORDER OF OPERATIONS: Train/Test Split -> Scale -> SMOTE.
    """
    if verbose:
        mode_str = "(WITH SMOTE)" if use_smote else "(STANDARD)"
        print(f"--- Starting ML Pre-Processing Pipeline {mode_str} ---")

    # 1. Define Features (X) and Target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if verbose:
        print("[Step 1] Performed Stratified 80/20 Train/Test Split.")

    # 3. Feature Scaling (MUST happen BEFORE SMOTE)
    scaler = StandardScaler()
    
    # Fit strictly on training data to prevent data leakage, then transform both
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    if verbose:
        print("[Step 2] Applied StandardScaler to features.")

    # 4. Conditional SMOTE Application
    if use_smote:
        # Apply SMOTE mathematically *after* the space is standardized
        smote = SMOTE(random_state=random_state)
        X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
        
        if verbose:
            print(f"[Step 3] Applied SMOTE. Original Train Positives: {y_train.sum()} -> New Train Positives: {y_train_final.sum()}")
            print(f"         - New Train size: {X_train_final.shape[0]} patients (Perfect 50/50 balance)")
            print(f"         - Test size (UNTOUCHED): {X_test_scaled.shape[0]} patients ({y_test.sum()} positive)")
            
    else:
        # Pass the scaled data through untouched
        X_train_final, y_train_final = X_train_scaled, y_train
        
        if verbose:
            print(f"         - Train size: {X_train_final.shape[0]} patients ({y_train_final.sum()} positive)")
            print(f"         - Test size:  {X_test_scaled.shape[0]} patients ({y_test.sum()} positive)")

    if verbose:
        print("--- Pre-Processing Complete ---\n")

    return X_train_final, X_test_scaled, y_train_final, y_test, scaler