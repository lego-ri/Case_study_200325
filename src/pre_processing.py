import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col='Biopsy', test_size=0.2, random_state=42, verbose=True):
    """
    Separates features and target, performs a stratified train/test split, 
    and scales the continuous features using StandardScaler.
    """
    if verbose:
        print("--- Starting ML Pre-Processing Pipeline ---")

    # 1. Define Features (X) and Target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    if verbose:
        print("[Step 1] Performed Stratified 80/20 Train/Test Split.")
        print(f"         - Train size: {X_train.shape[0]} patients ({y_train.sum()} positive)")
        print(f"         - Test size:  {X_test.shape[0]} patients ({y_test.sum()} positive)")

    # 3. Feature Scaling
    scaler = StandardScaler()
    
    # Fit strictly on training data to prevent data leakage, then transform both
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    if verbose:
        print("[Step 2] Applied StandardScaler to features.")
        print("--- Pre-Processing Complete ---\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler