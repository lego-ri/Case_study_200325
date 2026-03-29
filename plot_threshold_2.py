import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_champion_model_metrics(ledger_path, target_experiment, target_model):
    """
    Plots the Recall, Precision, and F1-Score for a SINGLE model 
    to demonstrate the clinical threshold trade-off.
    """
    df = pd.read_csv(ledger_path)
    
    # 1. Strict filtering for the specific scenario
    champion_df = df[(df['Experiment'] == target_experiment) & 
                     (df['Model'] == target_model)].copy()
    
    if champion_df.empty:
        print(f"Error: No data found for {target_model} in {target_experiment}.")
        return

    # 2. Prepare data for plotting
    plot_df = champion_df.melt(id_vars=['Threshold'], 
                               value_vars=['Recall', 'Precision', 'F1_Score'],
                               var_name='Metric', value_name='Score')

    # 3. Create the Visual
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Draw lines with distinct markers for accessibility
    ax = sns.lineplot(data=plot_df, x='Threshold', y='Score', hue='Metric', 
                      style='Metric', markers=True, markersize=10, linewidth=3)

    # 4. Inject Clinical Context Markers
    # plt.axvline(0.15, color='#e67e22', linestyle='--', linewidth=2, label='0.15 Triage Point')
    # plt.axvline(0.50, color='#2c3e50', linestyle='--', linewidth=2, label='0.50 Diagnostic Point')

    # 5. Presentation-Ready Styling
    # plt.title(f"Clinical Metric Sensitivity: {target_model}\n({target_experiment})", 
    #           fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Decision Threshold", fontsize=12, fontweight='bold')
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12, fontweight='bold')
    plt.ylim(-0.05, 1.05)
    plt.xlim(champion_df['Threshold'].min(), champion_df['Threshold'].max())
    
    # Place legend outside for clarity
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()

# --- Example Execution for your Slide 5 ---
# plot_champion_model_metrics("data/master_experiment_ledger.csv", 
#                             target_experiment="KNN_Post_test_standard_NoSMOTE", 
#                             target_model="XGBoost")
plot_champion_model_metrics("data/master_experiment_ledger.csv", 
                            target_experiment="MEDIAN_Pre_Screen_standard_NoSMOTE", 
                            target_model="Logistic Regression")
