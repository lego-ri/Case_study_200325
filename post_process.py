import pandas as pd
import numpy as np

def evaluate_best_models(ledger_path, max_overfit_delta=0.10, min_test_auc=0.55):
    """
    Rigorously evaluates the experimental ledger to extract the optimal Pre-Screen 
    and Post-Test models, penalizing severe overfitting and clinical uselessness.
    """
    # 1. Load the Data
    try:
        df = pd.read_csv(ledger_path)
    except Exception as e:
        return f"Error loading ledger: {e}"

    print("==========================================")
    print("CLINICAL MODEL EVALUATION REPORT")
    print("==========================================\n")

    # 2. Calculate the Overfitting Delta
    # A positive delta means the model memorized the training data.
    # A negative delta is mathematically rare but acceptable (performed better on test data).
    df['Overfit_Delta'] = df['CV_AUC'] - df['Test_AUC']

    # 6. Apply the Mathematical Firewall
    initial_count = len(df)
    
    # Filter 1: Disqualify models that severely overfit
    df_stable = df[df['Overfit_Delta'] <= max_overfit_delta].copy()
    
    # Filter 2: Disqualify models performing no better than random guessing
    df_valid = df_stable[df_stable['Test_AUC'] >= min_test_auc].copy()

    survivor_count = len(df_valid)
    print(f"[Audit] Evaluated {initial_count} configurations.")
    print(f"[Audit] {initial_count - survivor_count} configurations disqualified due to overfitting (Delta > {max_overfit_delta}) or failure to generalize (Test AUC < {min_test_auc}).\n")

    if survivor_count == 0:
        print("CRITICAL FAILURE: No models survived the mathematical firewall.")
        print("Action required: You must increase algorithmic regularization (e.g., lower C in Logistic Regression) or reconsider your feature space.")
        return None, None

    # 4. Separate the Pipelines
    pre_screen_mask = df_valid['Experiment'].str.contains('Pre_Screen', case=False, na=False)
    post_test_mask = df_valid['Experiment'].str.contains('Post_test', case=False, na=False)

    pre_screen_df = df_valid[pre_screen_mask].copy()
    post_test_df = df_valid[post_test_mask].copy()

    # 5. Apply Clinical Ranking Logic

    # PRE-SCREEN LOGIC: 
    # Goal: Catch potential cases early. 
    # Sorting Priority: 1. Maximize Recall (Don't miss cancers). 2. Maximize F1-Score (Keep false alarms tolerable). 6. Maximize Test AUC.
    if not pre_screen_df.empty:
        pre_screen_ranked = pre_screen_df.sort_values(
            by=['Recall', 'F1_Score', 'Test_AUC'], 
            ascending=[False, False, False]
        )
        print(">>> TOP 6 PRE-SCREEN MODELS <<<")
        print("Strategy: Maximizing Recall to cast a wide net, stabilized by F1-Score.")
        print(pre_screen_ranked[['Experiment', 'Model', 'Threshold', 'Recall', 'Precision', 'F1_Score', 'Test_AUC', 'Overfit_Delta']].head(6).to_string(index=False))
        print("-" * 60)
    else:
        print(">>> PRE-SCREEN MODELS: None survived the statistical firewall.\n")
        pre_screen_ranked = pd.DataFrame()

    # POST-TEST LOGIC:
    # Goal: Confirm diagnosis before invasive biopsy.
    # Sorting Priority: 1. Maximize F1-Score (Strict balance). 2. Maximize Precision (Do not trigger false biopsies). 6. Maximize Test AUC.
    if not post_test_df.empty:
        post_test_ranked = post_test_df.sort_values(
            by=['F1_Score', 'Precision', 'Test_AUC'], 
            ascending=[False, False, False]
        )
        print("\n>>> TOP 6 POST-TEST MODELS <<<")
        print("Strategy: Maximizing F1 and Precision to prevent unnecessary invasive biopsies.")
        print(post_test_ranked[['Experiment', 'Model', 'Threshold', 'Recall', 'Precision', 'F1_Score', 'Test_AUC', 'Overfit_Delta']].head(6).to_string(index=False))
        print("-" * 60)
    else:
        print("\n>>> POST-TEST MODELS: None survived the statistical firewall.\n")
        post_test_ranked = pd.DataFrame()

    return pre_screen_ranked, post_test_ranked

# Execution command:
pre_best, post_best = evaluate_best_models("data/master_experiment_ledger.csv")