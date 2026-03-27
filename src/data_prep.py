import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def clean_and_prepare_data(input_filepath, output_filepath=None, imputation_method='knn', verbosity=1, cols_to_drop_extra=None):
    """
    Loads raw cervical cancer data, cleans it, performs specified imputation (KNN or Median), 
    enforces rigorous clinical consistency via a Dual Firewall architecture, 
    removes redundant/leaking features, and returns an ML-ready dataframe.
    
    Parameters:
    - input_filepath (str): Path to the raw CSV data.
    - output_filepath (str): Optional path to save the cleaned data.
    - imputation_method (str): 'knn' or 'median'
    - verbosity (int): 0 (Silent), 1 (Summaries), 2 (Full Notebook-Style Audit Trail)
    - cols_to_drop_extra (list): Additional columns to drop (e.g., target leakage or redundant features).
    """
    if imputation_method not in ['knn', 'median']:
        raise ValueError("imputation_method must be either 'knn' or 'median'.")

    if verbosity >= 1:
        print("==========================================")
        print(f"--- Starting Clinical Data Prep Pipeline (Mode: {imputation_method.upper()}) ---")
        print("==========================================")

    # =========================================================
    # INTERNAL HELPER: THE CLINICAL FIREWALL
    # =========================================================
    def _run_firewall(df_work, stage_label):
        """Executes clinical boundary checks and prints Biopsy-prioritized audit logs."""
        corrections = 0

        # 1. FSI Check
        fsi_conflict = df_work['First sexual intercourse'] > (df_work['Age'] + 1e-8)
        if fsi_conflict.sum() > 0:
            if verbosity >= 2:
                print(f"-> {stage_label} OVERRIDE: Capped 'First sexual intercourse' at Age for {fsi_conflict.sum()} rows.")
                audit_df = df_work.loc[fsi_conflict, ['Age', 'First sexual intercourse', 'Biopsy']].copy()
                audit_df.rename(columns={'First sexual intercourse': f'FSI ({stage_label})'}, inplace=True)
                
            df_work.loc[fsi_conflict, 'First sexual intercourse'] = df_work.loc[fsi_conflict, 'Age']
            
            if verbosity >= 2:
                audit_df['FSI (Corrected)'] = df_work.loc[fsi_conflict, 'First sexual intercourse']
                audit_df = audit_df.sort_values(by='Biopsy', ascending=False)
                print(audit_df.head(max(5, (audit_df['Biopsy'] == 1).sum())).to_string() + "\n")
            corrections += fsi_conflict.sum()

        # 2. Duration Checks
        duration_cols = ['Smokes (years)', 'Hormonal Contraceptives (years)', 'IUD (years)']
        for col in duration_cols:
            if col in df_work.columns:
                dur_conflict = df_work[col] > (df_work['Age'] + 1e-8)
                if dur_conflict.sum() > 0:
                    if verbosity >= 2:
                        print(f"-> {stage_label} OVERRIDE: Capped '{col}' at Age for {dur_conflict.sum()} rows.")
                        audit_df = df_work.loc[dur_conflict, ['Age', col, 'Biopsy']].copy()
                        audit_df.rename(columns={col: f'{col} ({stage_label})'}, inplace=True)
                    
                    df_work.loc[dur_conflict, col] = df_work.loc[dur_conflict, 'Age']
                    
                    if verbosity >= 2:
                        audit_df[f'{col} (Corrected)'] = df_work.loc[dur_conflict, col]
                        audit_df = audit_df.sort_values(by='Biopsy', ascending=False)
                        print(audit_df.head(max(5, (audit_df['Biopsy'] == 1).sum())).to_string() + "\n")
                    corrections += dur_conflict.sum()

                bio_conflict = ((df_work['Age'] + 1e-8) - df_work[col]) < 5
                if bio_conflict.sum() > 0:
                    if verbosity >= 2:
                        print(f"-> {stage_label} OVERRIDE: Curtailed '{col}' to biological timeframe for {bio_conflict.sum()} rows.")
                        audit_df = df_work.loc[bio_conflict, ['Age', col, 'Biopsy']].copy()
                        audit_df.rename(columns={col: f'{col} ({stage_label})'}, inplace=True)
                    
                    df_work.loc[bio_conflict, col] = df_work.loc[bio_conflict, 'Age'] - 5
                    
                    if verbosity >= 2:
                        audit_df[f'{col} (Corrected)'] = df_work.loc[bio_conflict, col]
                        audit_df = audit_df.sort_values(by='Biopsy', ascending=False)
                        print(audit_df.head(max(5, (audit_df['Biopsy'] == 1).sum())).to_string() + "\n")
                    corrections += bio_conflict.sum()

        # 3. Pregnancy Checks
        years_active = (df_work['Age'] + 1e-8) - df_work['First sexual intercourse']
        preg_conflict = df_work['Num of pregnancies'] > years_active
        if preg_conflict.sum() > 0:
            if verbosity >= 2:
                print(f"-> {stage_label} OVERRIDE: Capped 'Num of pregnancies' at active years for {preg_conflict.sum()} rows.")
                audit_df = df_work.loc[preg_conflict, ['Age', 'First sexual intercourse', 'Num of pregnancies', 'Biopsy']].copy()
                audit_df.rename(columns={'Num of pregnancies': f'Pregnancies ({stage_label})'}, inplace=True)
                audit_df['Active Years Limit'] = np.floor(years_active[preg_conflict])
                
            df_work.loc[preg_conflict, 'Num of pregnancies'] = np.floor(years_active[preg_conflict])
            
            if verbosity >= 2:
                audit_df['Pregnancies (Corrected)'] = df_work.loc[preg_conflict, 'Num of pregnancies']
                audit_df = audit_df.sort_values(by='Biopsy', ascending=False)
                print(audit_df.head(max(5, (audit_df['Biopsy'] == 1).sum())).to_string() + "\n")
            corrections += preg_conflict.sum()

        # 4. Exposure Paradox
        exposure_conflict = (df_work['Number of sexual partners'] == 0) & ((df_work['First sexual intercourse'] > 0) | (df_work['Num of pregnancies'] > 0))
        if exposure_conflict.sum() > 0:
            if verbosity >= 2:
                print(f"-> {stage_label} OVERRIDE: Forced 'Number of sexual partners' to 1 for {exposure_conflict.sum()} rows (Exposure Paradox).")
                audit_df = df_work.loc[exposure_conflict, ['First sexual intercourse', 'Num of pregnancies', 'Number of sexual partners', 'Biopsy']].copy()
                audit_df.rename(columns={'Number of sexual partners': f'Partners ({stage_label})'}, inplace=True)
                
            df_work.loc[exposure_conflict, 'Number of sexual partners'] = 1
            
            if verbosity >= 2:
                audit_df['Partners (Corrected)'] = df_work.loc[exposure_conflict, 'Number of sexual partners']
                audit_df = audit_df.sort_values(by='Biopsy', ascending=False)
                print(audit_df.head(max(5, (audit_df['Biopsy'] == 1).sum())).to_string() + "\n")
            corrections += exposure_conflict.sum()

        # 5. Smoke Math
        smoke_math_conflict = (df_work['Smokes (packs/year)'] > 0) & (df_work['Smokes (years)'] == 0)
        if smoke_math_conflict.sum() > 0:
            if verbosity >= 2:
                print(f"-> {stage_label} OVERRIDE: Forced 'Smokes (years)' to 1 for {smoke_math_conflict.sum()} rows (Pack/Year Paradox).")
                audit_df = df_work.loc[smoke_math_conflict, ['Smokes (packs/year)', 'Smokes (years)', 'Biopsy']].copy()
                audit_df.rename(columns={'Smokes (years)': f'Smokes Yrs ({stage_label})'}, inplace=True)
                
            df_work.loc[smoke_math_conflict, 'Smokes (years)'] = 1
            
            if verbosity >= 2:
                audit_df['Smokes Yrs (Corrected)'] = df_work.loc[smoke_math_conflict, 'Smokes (years)']
                audit_df = audit_df.sort_values(by='Biopsy', ascending=False)
                print(audit_df.head(max(5, (audit_df['Biopsy'] == 1).sum())).to_string() + "\n")
            corrections += smoke_math_conflict.sum()

        # 6. Master STDs
        all_std_related_cols = [c for c in df_work.columns if c.startswith('STDs') and c not in ['STDs', 'STDs (number)']]
        std_conflict = (df_work['STDs'] == 0) & (df_work[all_std_related_cols].sum(axis=1) > 0)
        if std_conflict.sum() > 0:
            if verbosity >= 2:
                print(f"-> {stage_label} OVERRIDE: Forced master 'STDs' to 1.0 for {std_conflict.sum()} rows (Sub-STDs detected).")
                active_cols = [c for c in all_std_related_cols if df_work.loc[std_conflict, c].sum() > 0][:3]
                audit_df = df_work.loc[std_conflict, ['STDs'] + active_cols + ['Biopsy']].copy()
                audit_df.rename(columns={'STDs': f'STDs Master ({stage_label})'}, inplace=True)
                
            df_work.loc[std_conflict, 'STDs'] = 1.0
            
            if verbosity >= 2:
                audit_df['STDs Master (Corrected)'] = df_work.loc[std_conflict, 'STDs']
                audit_df = audit_df.sort_values(by='Biopsy', ascending=False)
                print(audit_df.head(max(5, (audit_df['Biopsy'] == 1).sum())).to_string() + "\n")
            corrections += std_conflict.sum()

        # 7. Condy Master
        condy_cols = [c for c in df_work.columns if 'condylomatosis' in c and c != 'STDs:condylomatosis']
        condy_conflict = (df_work['STDs:condylomatosis'] == 0) & (df_work[condy_cols].sum(axis=1) > 0)
        if condy_conflict.sum() > 0:
            if verbosity >= 2:
                print(f"-> {stage_label} OVERRIDE: Forced 'STDs:condylomatosis' to 1.0 for {condy_conflict.sum()} rows.")
                audit_df = df_work.loc[condy_conflict, ['STDs:condylomatosis'] + condy_cols + ['Biopsy']].copy()
                audit_df.rename(columns={'STDs:condylomatosis': f'Condy Master ({stage_label})'}, inplace=True)
                
            df_work.loc[condy_conflict, 'STDs:condylomatosis'] = 1.0
            
            if verbosity >= 2:
                audit_df['Condy Master (Corrected)'] = df_work.loc[condy_conflict, 'STDs:condylomatosis']
                audit_df = audit_df.sort_values(by='Biopsy', ascending=False)
                print(audit_df.head(max(5, (audit_df['Biopsy'] == 1).sum())).to_string() + "\n")
            corrections += condy_conflict.sum()

        return corrections

    # =========================================================
    # PIPELINE EXECUTION
    # =========================================================

    # ---------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------
    df = pd.read_csv(input_filepath)
    if verbosity >= 1:
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
            
    if verbosity >= 1:
        print("[Step 1] Handled '?' placeholders and converted data to numeric.")

    # ---------------------------------------------------------
    # 3. Smart Dropping & Ground Truth Sanitization
    # ---------------------------------------------------------
    bad_fsi = df['First sexual intercourse'] > (df['Age'] + 1e-8)
    bad_smokes_years = df['Smokes (years)'] >= (df['Age'] + 1e-8)
    bad_hc_years = df['Hormonal Contraceptives (years)'] >= (df['Age'] + 1e-8)
    bad_iud_years = df['IUD (years)'] >= (df['Age'] + 1e-8)
    impossible_hc_start = ((df['Age'] + 1e-8) - df['Hormonal Contraceptives (years)']) < 5
    impossible_iud_start = ((df['Age'] + 1e-8) - df['IUD (years)']) < 5
    
    years_active = (df['Age'] + 1e-8) - df['First sexual intercourse']
    impossible_pregnancies = df['Num of pregnancies'] > years_active

    master_corruption_mask = (
        bad_fsi | bad_smokes_years | bad_hc_years | bad_iud_years | 
        impossible_hc_start | impossible_iud_start | impossible_pregnancies
    )

    safe_to_drop_mask = master_corruption_mask & (df['Biopsy'] == 0)
    salvaged_positive_cases = master_corruption_mask & (df['Biopsy'] == 1)

    patients_before = len(df)
    df = df[~safe_to_drop_mask].reset_index(drop=True)
    
    if verbosity >= 1:
        print(f"\n[Step 2] Ground Truth Sanitization:")
        print(f"         -> Dropped {patients_before - len(df)} corrupted majority-class patients (Biopsy = 0).")
        print(f"         -> Salvaged {salvaged_positive_cases.sum()} corrupted minority-class patients (Biopsy = 1) for correction.")
        print("         -> Executing Pre-Imputation Firewall (Fixing Raw Data)...")

    # EXECUTE FIREWALL 1
    pre_corrections = _run_firewall(df, "Pre-Imputation")
    
    if verbosity >= 1:
        print(f"         -> Pre-Imputation Firewall resolved {pre_corrections} raw logical contradictions.")

    # ---------------------------------------------------------
    # 4. Handling Missing Data (Imputation Engine)
    # ---------------------------------------------------------
    cols_to_drop = ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    if verbosity >= 1:
        print(f"\n[Info] Dropped {len(cols_to_drop)} columns due to severe missingness (>90%).")
    if verbosity >= 2:
        print(f"   -> Columns removed: {cols_to_drop}")

    conditional_groups = [
        ('IUD', ['IUD (years)']), 
        ('Hormonal Contraceptives', ['Hormonal Contraceptives (years)']),
        ('Smokes', ['Smokes (years)', 'Smokes (packs/year)'])
    ]

    # --- MEDIAN IMPUTATION BRANCH ---
    if imputation_method == 'median':
        if verbosity >= 1:
            print("\n[Step 3] Executing Deterministic & Median Imputation...")
            
        for master, sub_cols in conditional_groups:
            master_missing = df[master].isnull().sum()
            master_median = df[master].median()
            df[master] = df[master].fillna(master_median)
            
            if verbosity >= 2 and master_missing > 0:
                print(f"-> '{master}': Filled {master_missing} missing values with median {master_median}.")
                
            for sub in sub_cols:
                sub_missing = df[sub].isnull().sum()
                user_median = df[df[master] == 1][sub].median()
                
                zero_fills = ((df[master] == 0) & df[sub].isnull()).sum()
                median_fills = ((df[master] == 1) & df[sub].isnull()).sum()
                
                df.loc[df[master] == 0, sub] = df.loc[df[master] == 0, sub].fillna(0)
                df.loc[df[master] == 1, sub] = df.loc[df[master] == 1, sub].fillna(user_median)
                
                if verbosity >= 2 and sub_missing > 0:
                    print(f"   - '{sub}': Filled {sub_missing} missing values (Zeros: {zero_fills}, Medians: {median_fills}).")
                
        fallback_count = 0
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                col_median = df[col].median()
                df[col] = df[col].fillna(col_median)
                if verbosity >= 2:
                    print(f"-> '{col}': Filled {missing_count} missing values with global median {col_median}.")
                fallback_count += missing_count

    # --- KNN IMPUTATION BRANCH ---
    elif imputation_method == 'knn':
        if verbosity >= 1:
            print("\n[Step 3] Executing Deterministic Pre-fill & KNN Imputation...")
            
        for master, sub_cols in conditional_groups:
            df[master] = df[master].fillna(df[master].median())
            for sub in sub_cols:
                fills = ((df[master] == 0) & df[sub].isnull()).sum()
                df.loc[df[master] == 0, sub] = df.loc[df[master] == 0, sub].fillna(0)
                if verbosity >= 2 and fills > 0:
                    print(f"-> '{sub}': Pre-filled {fills} NaNs with 0.0 because master '{master}' == 0.")

        binary_cols = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0.0, 1.0})]
        missing_mask = df.isnull()

        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df_imputed_scaled = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df.columns, index=df.index)
        df = pd.DataFrame(scaler.inverse_transform(df_imputed_scaled), columns=df.columns, index=df.index)

        if verbosity >= 2:
            print("\n   [Audit: Raw KNN Imputed Values Before Rounding]")
            knn_imputed_count = 0
            for col in df.columns:
                if missing_mask[col].any():
                    imputed_raw_vals = df.loc[missing_mask[col], col]
                    missing_count = missing_mask[col].sum()
                    print(f"   -> '{col}' ({missing_count} missing) | Min: {imputed_raw_vals.min():.4f}, Max: {imputed_raw_vals.max():.4f}, Mean: {imputed_raw_vals.mean():.4f}")
                    knn_imputed_count += missing_count
            print(f"\nTotal values dynamically guessed via KNN: {knn_imputed_count}\n")

        # Domain-Aware Rounding
        for col in binary_cols:
            if missing_mask[col].any():
                df[col] = np.round(df[col])

        discrete_cols = ['Number of sexual partners', 'Num of pregnancies', 'STDs (number)', 'STDs: Number of diagnosis']
        for col in discrete_cols:
            if col in df.columns and missing_mask[col].any():
                df[col] = np.round(df[col])

    # ---------------------------------------------------------
    # 5. Post-Imputation Compliance Firewall
    # ---------------------------------------------------------
    if verbosity >= 1:
        print("\n[Step 4] Enforcing Clinical Boundaries (Post-Imputation Firewall)...")
        
    # EXECUTE FIREWALL 2
    post_corrections = _run_firewall(df, "Post-Imputation")
    
    if verbosity >= 1:
        print(f"         -> Post-Imputation Firewall resolved {post_corrections} algorithmic artifacts.")

    # ---------------------------------------------------------
    # 6. Feature Selection (Dropping redundancy & leakage)
    # ---------------------------------------------------------
    # Default drops for zero-variance mathematical dead weight
    redundant_and_leaking_cols = [
        'STDs:cervical condylomatosis', # Zero variance
        'STDs:AIDS'                     # Zero variance
    ]
    
    # Extend the default drops with any strategic drops provided by the caller
    if cols_to_drop_extra:
        redundant_and_leaking_cols.extend(cols_to_drop_extra)
    
    df = df.drop(columns=redundant_and_leaking_cols, errors='ignore')
    
    if verbosity >= 1:
        print(f"\n[Step 5] Dropped {len(redundant_and_leaking_cols)} redundant, non-variant, or target-leaking features.")

    # ---------------------------------------------------------
    # 7. Save and Return
    # ---------------------------------------------------------
    if output_filepath:
        df.to_csv(output_filepath, index=False)
        if verbosity >= 1:
            print(f"[Info] Clean data saved to {output_filepath}")
            
    if verbosity >= 1:
        print(f"[Success] Final Clean Data Shape: {df.shape[0]} patients, {df.shape[1]} features.")
        print("--- Data Prep Complete ---\n")
        
    return df

if __name__ == "__main__":
    # If this script is run directly for testing purposes:
    input_path = "../data/risk_factors_cervical_cancer.csv"
    output_path_knn = "../data/clean_cervical_cancer_knn.csv"
    output_path_median = "../data/clean_cervical_cancer_median.csv"
    
    # Define the custom extra drops based on EDA collinearity/leakage findings
    extra_drops = [
        'STDs',                         # Redundant to STDs (number)
        'STDs:condylomatosis',          # Redundant master column
        # 'Citology',                   # Target leakage (commented out as an example of flexibility)
        'Schiller',                     # Target leakage
        'Hinselmann'                    # Target leakage
    ]
    
    # Example: Run KNN with full massive notebook-style logging, passing the extra drops
    clean_df_knn = clean_and_prepare_data(
        input_filepath=input_path, 
        output_filepath=output_path_knn, 
        imputation_method='knn', 
        verbosity=2,
        cols_to_drop_extra=extra_drops
    )
    
    # Example: Run Median with summary-style logging
    clean_df_median = clean_and_prepare_data(
        input_filepath=input_path, 
        output_filepath=output_path_median, 
        imputation_method='median', 
        verbosity=1,
        cols_to_drop_extra=extra_drops
    )