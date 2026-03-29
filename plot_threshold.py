import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_ledger_recall_plot(ledger_path, target_experiment=False):
    """
    Generates a rigorous Threshold vs. Metrics plot from the saved ledger.
    Helps visualize the 'Trade-off' point for Clinical Triage.
    """
    df = pd.read_csv(ledger_path)
    
    # Filter for a specific experiment if requested (e.g., 'Pre_Screen_KNN_Ablation')
    if target_experiment:
        df = df[df['Experiment'] == target_experiment]
    
    # Melt the dataframe to make it 'tidy' for Seaborn (Long-format)
    # We want to plot Recall and Precision on the same Y-axis
    plot_df = df.melt(id_vars=['Model', 'Threshold', 'Experiment'], 
                      value_vars=['Recall', 'Precision', 'F1_Score'],
                      var_name='Metric', value_name='Score')

    # Set the visual style to match a professional presentation
    sns.set_theme(style="whitegrid")
    g = sns.FacetGrid(plot_df, col="Model", hue="Metric", height=5, aspect=1.2, col_wrap=3)
    
    # Map the lineplot
    g.map(sns.lineplot, "Threshold", "Score", marker="o", linewidth=3)
    
    # Add a vertical line at the 0.15 Triage mark to match your Slide 5 narrative
    for ax in g.axes.flat:
        ax.axvline(0.15, color='orange', linestyle='--', alpha=0.7, label='Triage (0.15)')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Decision Threshold", fontweight='bold')
        ax.set_ylabel("Metric Score", fontweight='bold')

    g.add_legend(title="Clinical Metrics")
    g.set_titles("{col_name}", fontweight='bold', size=14)
    
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f"Threshold Sensitivity Analysis: {target_experiment if target_experiment else 'All Experiments'}", 
                   fontsize=18, fontweight='bold')
    
    plt.show()

# Execution:
generate_ledger_recall_plot("data/master_experiment_ledger.csv")