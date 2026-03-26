import pandas as pd
from src.data_prep import clean_and_prepare_data

# 1. Prepare Data
input_path = "data/risk_factors_cervical_cancer.csv"
output_path = "data/risk_factors_cervical_cancer_cleaned.csv"
clean_df = clean_and_prepare_data(input_path, output_path)

# Now clean_df is perfectly ready for Phase 4!