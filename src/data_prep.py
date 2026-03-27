import pandas as pd
import numpy as np

def clean_and_prepare_data(input_filepath, output_filepath=None, verbose=True):
    """
    Loads raw cervical cancer data, cleans it, performs complex imputation, 
    removes redundant/leaking features, and returns a ML-ready dataframe.
    """
    if verbose:
        print("--- Starting Data Prep Pipeline ---")

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
    # 3. Handling Missing Data (Imputation) and corrupted data
    # ---------------------------------------------------------
    # Drop the fundamentally corrupted rows identified in Test 10 of Step 2 (see notebook)
    df = df.drop(index=[312, 812])

    # Optional but recommended: Reset the index after dropping rows to maintain a clean sequence
    df = df.reset_index(drop=True)

    cols_to_drop = ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    std_cols = [col for col in df.columns if col.startswith('STDs')]
    for col in std_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    conditional_groups = [
        ('IUD', ['IUD (years)']), 
        ('Hormonal Contraceptives', ['Hormonal Contraceptives (years)']),
        ('Smokes', ['Smokes (years)', 'Smokes (packs/year)'])
    ]

    for master, sub_cols in conditional_groups:
        master_median = df[master].median()
        df[master] = df[master].fillna(master_median)
        
        for sub in sub_cols:
            user_median = df[df[master] == 1][sub].median()
            df.loc[df[master] == 0, sub] = df.loc[df[master] == 0, sub].fillna(0)
            df.loc[df[master] == 1, sub] = df.loc[df[master] == 1, sub].fillna(user_median)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    if verbose:
        print(f"[Step 2] Dropped {len(cols_to_drop)} highly missing columns.")
        print("[Step 3] Performed conditional and median imputation. (Total missing values: 0)")

    # ---------------------------------------------------------
    # 4. Feature Selection (Dropping redundancy & leakage)
    # ---------------------------------------------------------
    redundant_and_leaking_cols = [
        'STDs:cervical condylomatosis', # Zero variance
        'STDs:AIDS',                    # Zero variance
        # 'STDs',                         # Redundant to STDs (number)
        # 'STDs:condylomatosis',          # Redundant master column
        # 'Citology'                      # Target leakage
        'Schiller',                     # Target leakage
        'Hinselmann',                   # Target leakage
    ]
    
    df = df.drop(columns=redundant_and_leaking_cols, errors='ignore')
    
    if verbose:
        print(f"[Step 4] Dropped {len(redundant_and_leaking_cols)} redundant or target-leaking features.")

    # ---------------------------------------------------------
    # 5. Save and Return
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
    output_path = "../data/clean_cervical_cancer_data.csv"
    
    clean_df = clean_and_prepare_data(input_path, output_path)