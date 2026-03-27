import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def clean_and_prepare_data(input_filepath, output_filepath=None, imputation_method='knn', verbose=True):
    """
    Loads raw cervical cancer data, cleans it, performs specified imputation (KNN or Median), 
    enforces rigorous clinical consistency, removes redundant/leaking features, 
    and returns an ML-ready dataframe.
    
    Parameters:
    - imputation_method (str): 'knn' or 'median'
    """
    if imputation_method not in ['knn', 'median']:
        raise ValueError("imputation_method must be either 'knn' or 'median'.")

    if verbose:
        print("==========================================")
        print(f"--- Starting Clinical Data Prep Pipeline (Mode: {imputation_method.upper()}) ---")
        print("==========================================")

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    df = pd.read_csv(input_filepath)
    if verbose:
        print(f"[Info] Original Data Loaded: {df.shape[0]} patients, {df.shape[1]} features.")

    # ---------------------------------------------------------
    # 2. Basic Cleaning & Type Conversion
    # ---------------------------------------------------------
    df = df.replace('?', np.nan)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass
            
    if verbose:
        print("[Step 1] Handled '?' placeholders and converted data to numeric.")

    # ---------------------------------------------------------
    # 3. Dynamic Filtering of Corrupted Rows
    # ---------------------------------------------------------
    bad_fsi = df['First sexual intercourse'] > (df['Age'] + 1e-8)
    bad_smokes_years = df['Smokes (years)'] >= (df['Age'] + 1e-8)
    bad_hc_years = df['Hormonal Contraceptives (years)'] >= (df['Age'] + 1e-8)
    bad_iud_years = df['IUD (years)'] >= (df['Age'] + 1e-8)

    chrono_flaws_mask = bad_fsi | bad_smokes_years | bad_hc_years | bad_iud_years
    
    patients_before = len(df)
    df = df[~chrono_flaws_mask].reset_index(drop=True)
    
    if verbose:
        print(f"[Step 2] Dropped {patients_before - len(df)} chronologically impossible rows.")

    # ---------------------------------------------------------
    # 4. Handling Missing Data (Imputation Engine)
    # ---------------------------------------------------------
    cols_to_drop = ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    conditional_groups = [
        ('IUD', ['IUD (years)']), 
        ('Hormonal Contraceptives', ['Hormonal Contraceptives (years)']),
        ('Smokes', ['Smokes (years)', 'Smokes (packs/year)'])
    ]

    if imputation_method == 'median':
        if verbose:
            print("[Step 3] Executing Deterministic & Median Imputation...")
            
        for master, sub_cols in conditional_groups:
            df[master] = df[master].fillna(df[master].median())
            for sub in sub_cols:
                user_median = df[df[master] == 1][sub].median()
                df.loc[df[master] == 0, sub] = df.loc[df[master] == 0, sub].fillna(0)
                df.loc[df[master] == 1, sub] = df.loc[df[master] == 1, sub].fillna(user_median)
                
        # Global fallback for remaining columns
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

    elif imputation_method == 'knn':
        if verbose:
            print("[Step 3] Executing Deterministic Pre-fill & KNN Imputation...")
            
        for master, sub_cols in conditional_groups:
            df[master] = df[master].fillna(df[master].median())
            for sub in sub_cols:
                df.loc[df[master] == 0, sub] = df.loc[df[master] == 0, sub].fillna(0)

        binary_cols = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0.0, 1.0})]
        missing_mask = df.isnull()

        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df_imputed_scaled = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df.columns, index=df.index)
        df = pd.DataFrame(scaler.inverse_transform(df_imputed_scaled), columns=df.columns, index=df.index)

        # Domain-Aware Rounding
        for col in binary_cols:
            if missing_mask[col].any():
                df[col] = np.round(df[col])

        discrete_cols = ['Number of sexual partners', 'Num of pregnancies', 'STDs (number)', 'STDs: Number of diagnosis']
        for col in discrete_cols:
            if col in df.columns and missing_mask[col].any():
                df[col] = np.round(df[col])

    # ---------------------------------------------------------
    # 5. Post-Imputation Clinical Override Firewall
    # ---------------------------------------------------------
    if verbose:
        print("[Step 4] Enforcing Clinical Boundaries (Firewall)...")
        
    corrections = 0

    fsi_conflict = df['First sexual intercourse'] > (df['Age'] + 1e-8)
    df.loc[fsi_conflict, 'First sexual intercourse'] = df.loc[fsi_conflict, 'Age']
    corrections += fsi_conflict.sum()

    duration_cols = ['Smokes (years)', 'Hormonal Contraceptives (years)', 'IUD (years)']
    for col in duration_cols:
        if col in df.columns:
            dur_conflict = df[col] > (df['Age'] + 1e-8)
            df.loc[dur_conflict, col] = df.loc[dur_conflict, 'Age']
            corrections += dur_conflict.sum()

            bio_conflict = ((df['Age'] + 1e-8) - df[col]) < 5
            df.loc[bio_conflict, col] = df.loc[bio_conflict, 'Age'] - 5
            corrections += bio_conflict.sum()

    years_active = (df['Age'] + 1e-8) - df['First sexual intercourse']
    preg_conflict = df['Num of pregnancies'] > years_active
    df.loc[preg_conflict, 'Num of pregnancies'] = np.floor(years_active[preg_conflict])
    corrections += preg_conflict.sum()

    exposure_conflict = (df['Number of sexual partners'] == 0) & ((df['First sexual intercourse'] > 0) | (df['Num of pregnancies'] > 0))
    df.loc[exposure_conflict, 'Number of sexual partners'] = 1
    corrections += exposure_conflict.sum()

    smoke_math_conflict = (df['Smokes (packs/year)'] > 0) & (df['Smokes (years)'] == 0)
    df.loc[smoke_math_conflict, 'Smokes (years)'] = 1
    corrections += smoke_math_conflict.sum()

    all_std_related_cols = [c for c in df.columns if c.startswith('STDs') and c not in ['STDs', 'STDs (number)']]
    std_conflict = (df['STDs'] == 0) & (df[all_std_related_cols].sum(axis=1) > 0)
    df.loc[std_conflict, 'STDs'] = 1.0
    corrections += std_conflict.sum()

    condy_cols = [c for c in df.columns if 'condylomatosis' in c and c != 'STDs:condylomatosis']
    condy_conflict = (df['STDs:condylomatosis'] == 0) & (df[condy_cols].sum(axis=1) > 0)
    df.loc[condy_conflict, 'STDs:condylomatosis'] = 1.0
    corrections += condy_conflict.sum()

    if verbose:
        print(f"         -> Firewall resolved {corrections} logical contradictions.")

    # ---------------------------------------------------------
    # 6. Feature Selection (Dropping redundancy & leakage)
    # ---------------------------------------------------------
    redundant_and_leaking_cols = [
        'STDs:cervical condylomatosis', # Zero variance
        'STDs:AIDS',                    # Zero variance
        'STDs',                         # Redundant to STDs (number)
        'STDs:condylomatosis',          # Redundant master column
        'Citology',                     # Target leakage
        'Schiller',                     # Target leakage
        'Hinselmann'                    # Target leakage
    ]
    
    df = df.drop(columns=redundant_and_leaking_cols, errors='ignore')
    
    if verbose:
        print(f"[Step 5] Dropped {len(redundant_and_leaking_cols)} redundant or target-leaking features.")

    # ---------------------------------------------------------
    # 7. Save and Return
    # ---------------------------------------------------------
    if output_filepath:
        df.to_csv(output_filepath, index=False)
        if verbose:
            print(f"[Info] Clean data saved to {output_filepath}")
            
    if verbose:
        print(f"[Success] Final Clean Data Shape: {df.shape[0]} patients, {df.shape[1]} features.")
        print("--- Data Prep Complete ---\n")
        
    return df

if __name__ == "__main__":
    # If this script is run directly for testing purposes:
    input_path = "../data/risk_factors_cervical_cancer.csv"
    output_path_knn = "../data/clean_cervical_cancer_knn.csv"
    output_path_median = "../data/clean_cervical_cancer_median.csv"
    
    # Generate both versions to test downstream ML model sensitivity
    clean_df_knn = clean_and_prepare_data(input_path, output_path_knn, imputation_method='knn')
    clean_df_median = clean_and_prepare_data(input_path, output_path_median, imputation_method='median')