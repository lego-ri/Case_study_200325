from src.data_prep import load_and_clean_data
# from src.model_training import train_model (You will build this later)

def main():
    print("Starting Cervical Cancer ML Pipeline...")
    
    # 1. Data Preparation
    file_path = "data/risk_factors_cervical_cancer.csv"
    print("Loading and cleaning data...")
    clean_df = load_and_clean_data(file_path)
    
    # Quick sanity check for the console
    print(f"Data ready! Shape: {clean_df.shape}. Missing values: {clean_df.isnull().sum().sum()}")
    
    # 2. Model Training (Placeholder for your next step)
    # print("Training model...")
    # model, metrics = train_model(clean_df, target_col='Biopsy')

if __name__ == "__main__":
    main()