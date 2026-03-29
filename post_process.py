import pandas as pd
import numpy as np

def evaluate_best_models(ledger_path, max_overfit_delta=0.10, min_test_auc=0.55):
    """
    Rigorously evaluates the experimental ledger to extract the optimal Pre-Screen 
    and Post-Test models, penalizing severe overfitting and clinical uselessness.
    Now includes optimal hyperparameter extraction.
    """
    try:
        df = pd.read_csv(ledger_path)
    except Exception as e:
        return f"Error loading ledger: {e}"

    print("==========================================")
    print("CLINICAL MODEL EVALUATION REPORT")
    print("==========================================\n")

    df['Overfit_Delta'] = df['CV_AUC'] - df['Test_AUC']

    initial_count = len(df)
    df_stable = df[df['Overfit_Delta'] <= max_overfit_delta].copy()
    df_valid = df_stable[df_stable['Test_AUC'] >= min_test_auc].copy()

    survivor_count = len(df_valid)
    print(f"[Audit] Evaluated {initial_count} configurations.")
    print(f"[Audit] {initial_count - survivor_count} configurations disqualified due to overfitting (Delta > {max_overfit_delta}) or failure to generalize (Test AUC < {min_test_auc}).\n")

    if survivor_count == 0:
        print("CRITICAL FAILURE: No models survived the mathematical firewall.")
        print("Action required: You must increase algorithmic regularization (e.g., lower C in Logistic Regression) or reconsider your feature space.")
        return None, None

    pre_screen_mask = df_valid['Experiment'].str.contains('Pre_Screen', case=False, na=False)
    post_test_mask = df_valid['Experiment'].str.contains('Post_test', case=False, na=False)

    pre_screen_df = df_valid[pre_screen_mask].copy()
    post_test_df = df_valid[post_test_mask].copy()

    # Determine columns to print safely (fall back if 'Parameters' is missing from an old CSV)
    base_cols = ['Experiment', 'Model', 'Threshold', 'Recall', 'Precision', 'F1_Score', 'Test_AUC', 'Overfit_Delta']
    print_cols = base_cols + ['Parameters'] if 'Parameters' in df_valid.columns else base_cols

    # Force Pandas to display the full dictionary string without truncation
    pd.set_option('display.max_colwidth', None)

    if not pre_screen_df.empty:
        pre_screen_ranked = pre_screen_df.sort_values(
            by=['Recall', 'F1_Score', 'Test_AUC'], 
            ascending=[False, False, False]
        )
        print(">>> TOP 6 PRE-SCREEN MODELS <<<")
        print("Strategy: Maximizing Recall to cast a wide net, stabilized by F1-Score.")
        print(pre_screen_ranked[print_cols].head(6).to_string(index=False))
        print("-" * 80)
    else:
        print(">>> PRE-SCREEN MODELS: None survived the statistical firewall.\n")
        pre_screen_ranked = pd.DataFrame()

    if not post_test_df.empty:
        post_test_ranked = post_test_df.sort_values(
            by=['F1_Score', 'Precision', 'Test_AUC'], 
            ascending=[False, False, False]
        )
        print("\n>>> TOP 6 POST-TEST MODELS <<<")
        print("Strategy: Maximizing F1 and Precision to prevent unnecessary invasive biopsies.")
        print(post_test_ranked[print_cols].head(6).to_string(index=False))
        print("-" * 80)
    else:
        print("\n>>> POST-TEST MODELS: None survived the statistical firewall.\n")
        post_test_ranked = pd.DataFrame()

    # Reset pandas display options to default to prevent breaking future print statements
    pd.reset_option('display.max_colwidth')

    return pre_screen_ranked, post_test_ranked

# Execution command:
pre_best, post_best = evaluate_best_models("data/master_experiment_ledger.csv")