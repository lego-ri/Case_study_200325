import pandas as pd
import numpy as np

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Cervical Cancer dataset, handles missing values, 
    and returns a clean, numeric pandas DataFrame ready for modeling.
    """
    # 1. Load the data
    df = pd.read_csv(file_path)

    # 2. Standardize missing values
    df = df.replace('?', np.nan)

    # 3. Convert all columns to numeric
    # 'coerce' forces unconvertible text to NaN, which handles our '?' replacements perfectly
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Drop heavily missing columns (>50% missing)
    columns_to_drop = ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # 5. Impute remaining missing values with the median
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    return df