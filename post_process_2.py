import pandas as pd
import numpy as np

def evaluate_best_models(ledger_path, max_overfit_delta=0.10, min_test_auc=0.55, min_precision=0.06, target_model=None):
    """
    Rigorously evaluates the experimental ledger to extract the optimal Pre-Screen 
    and Post-Test models, penalizing severe overfitting and clinical uselessness.
    Includes a minimum precision floor and optional target_model isolation.
    """
    try:
        df = pd.read_csv(ledger_path)
    except Exception as e:
        return f"Error loading ledger: {e}"

    print("==========================================")
    print("CLINICAL MODEL EVALUATION REPORT")
    print("==========================================\n")

    # [NEW] Isolate specific models if requested
    if target_model is not None:
        if isinstance(target_model, str):
            df = df[df['Model'] == target_model].copy()
            print(f"[Audit] ISOLATING MODEL ARCHITECTURE: {target_model}\n")
        elif isinstance(target_model, list):
            df = df[df['Model'].isin(target_model)].copy()
            print(f"[Audit] ISOLATING MODEL ARCHITECTURES: {', '.join(target_model)}\n")
            
        if df.empty:
            print(f"CRITICAL FAILURE: No records found matching the specified target_model.")
            return pd.DataFrame(), pd.DataFrame()

    df['Overfit_Delta'] = df['CV_AUC'] - df['Test_AUC']

    initial_count = len(df)
    df_stable = df[df['Overfit_Delta'] <= max_overfit_delta].copy()
    
    # Apply strict thresholds: Test AUC and Minimum Precision
    df_valid = df_stable[(df_stable['Test_AUC'] >= min_test_auc) & (df_stable['Precision'] > min_precision)].copy()

    survivor_count = len(df_valid)
    print(f"[Audit] Evaluated {initial_count} configurations.")
    print(f"[Audit] {initial_count - survivor_count} configurations disqualified due to overfitting (Delta > {max_overfit_delta}), poor generalization (Test AUC < {min_test_auc}), or low precision (<= {min_precision}).\n")

    if survivor_count == 0:
        print("CRITICAL FAILURE: No models survived the mathematical firewall.")
        print("Action required: You must increase algorithmic regularization, reconsider your feature space, or lower your evaluation thresholds.")
        return pd.DataFrame(), pd.DataFrame()

    pre_screen_mask = df_valid['Experiment'].str.contains('Pre_Screen', case=False, na=False)
    post_test_mask = df_valid['Experiment'].str.contains('Post_test', case=False, na=False)

    pre_screen_df = df_valid[pre_screen_mask].copy()
    post_test_df = df_valid[post_test_mask].copy()

    # Determine columns to print safely
    base_cols = ['Experiment', 'Model', 'Threshold', 'Recall', 'Precision', 'F1_Score', 'Test_AUC', 'Overfit_Delta']
    print_cols = base_cols + ['Parameters'] if 'Parameters' in df_valid.columns else base_cols

    # Force Pandas to display the full dictionary string without truncation
    pd.set_option('display.max_colwidth', None)

    if not pre_screen_df.empty:
        # Reverted to original strategy: Recall first
        pre_screen_ranked = pre_screen_df.sort_values(
            by=['Recall', 'Precision', 'Test_AUC'], 
            ascending=[False, False, False]
        )
        print(">>> TOP 6 PRE-SCREEN MODELS <<<")
        print(f"Strategy: Maximizing Recall to cast a wide net, stabilized by F1-Score (Precision > {min_precision * 100}%).")
        print(pre_screen_ranked[print_cols].head(6).to_string(index=False))
        print("-" * 80)
    else:
        print(">>> PRE-SCREEN MODELS: None survived the statistical firewall.\n")
        pre_screen_ranked = pd.DataFrame()

    if not post_test_df.empty:
        # Reverted to original strategy: F1/Precision first
        post_test_ranked = post_test_df.sort_values(
            by=['F1_Score', 'Precision', 'Test_AUC'], 
            ascending=[False, False, False]
        )
        print("\n>>> TOP 6 POST-TEST MODELS <<<")
        print(f"Strategy: Maximizing F1 and Precision to prevent unnecessary invasive biopsies (Precision > {min_precision * 100}%).")
        print(post_test_ranked[print_cols].head(6).to_string(index=False))
        print("-" * 80)
    else:
        print("\n>>> POST-TEST MODELS: None survived the statistical firewall.\n")
        post_test_ranked = pd.DataFrame()

    # Reset pandas display options to default
    pd.reset_option('display.max_colwidth')

    return pre_screen_ranked, post_test_ranked

# Execution examples:
# 1. Evaluate everything (default behavior):
pre_best, post_best = evaluate_best_models("data/master_experiment_ledger.csv")

# 2. Evaluate ONLY Logistic Regression models:
# pre_best, post_best = evaluate_best_models("data/master_experiment_ledger.csv", target_model="Logistic Regression")

# 3. Evaluate a subset of models:
# pre_best, post_best = evaluate_best_models("data/master_experiment_ledger.csv", target_model=["Random Forest", "XGBoost"])