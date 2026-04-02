import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_ledger_recall_plot(ledger_path, target_experiment=None, target_model=None):
    """
    Generates the Threshold vs. Metrics plot for a SPECIFIC model and experiment.
    Aggressively compressed height for a wide, banner-style presentation footprint.
    """
    df = pd.read_csv(ledger_path)
    
    # 1. Rigorous Filtering
    if target_experiment:
        df = df[df['Experiment'] == target_experiment]
    
    if target_model:
        df = df[df['Model'] == target_model]

    if df.empty:
        print(f"Error: No data found for Model: {target_model} in Experiment: {target_experiment}")
        return

    # 2. Melt for Seaborn (Long-format)
    plot_df = df.melt(id_vars=['Model', 'Threshold', 'Experiment'], 
                      value_vars=['Recall', 'Precision', 'F1_Score'],
                      var_name='Metric', value_name='Score')

    # 3. Visual Styling
    sns.set_theme(style="whitegrid")
    
    # ADJUSTED HERE: height=3.5 and aspect=2.0 strictly flattens the plot
    g = sns.FacetGrid(plot_df, col="Model", hue="Metric", height=3.5, aspect=4.0)
    
    # 4. Map the lineplot
    g.map(sns.lineplot, "Threshold", "Score", marker="o", linewidth=3)
    
    # 5. Injection of Clinical Reference Lines
    for ax in g.axes.flat:
        ax.axvline(0.15, color='orange', linestyle='--', alpha=0.7)#, label='Triage (0.15)')
        ax.axvline(0.50, color='black', linestyle='--', alpha=0.5)#, label='Balance (0.50)')
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Decision Threshold", fontweight='bold')
        ax.set_ylabel("Metric Score", fontweight='bold')
        
        # Strip the default Seaborn subplot title
        ax.set_title("")

    # 6. Precision Legend Placement
    # Tight bounding box keeps it directly next to the compressed plot
    plt.legend(title=target_model if target_model else "Model Metrics", 
               bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
               title_fontproperties={'weight': 'bold', 'size': 14})
    
    # Forces the bounding box to encapsulate the legend without cutting it off
    plt.tight_layout()
    plt.show()

# --- Execution ---
generate_ledger_recall_plot(
    ledger_path="data/master_experiment_ledger.csv", 
    # target_experiment="MEDIAN_Pre_Screen_standard_NoSMOTE", 
    # target_model="Logistic Regression"
    # target_model="Random Forest"
    target_model="XGBoost"

)