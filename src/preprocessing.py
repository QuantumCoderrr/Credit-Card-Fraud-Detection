import pandas as pd
import zipfile

def load_data(file_path: str):
    with zipfile.ZipFile(file_path) as z:
        with z.open('creditcard.csv') as f:
            df = pd.read_csv(f)
    return df

def preprocess_data(df):
    # Basic EDA and data preprocessing
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())

    # Check for missing values
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())

    # Check class distribution
    print("\nClass distribution:")
    print(df['Class'].value_counts())

    # Sampling 10% of the dataset for faster training
    print("\nSampling 10% of the dataset for faster training...")
    df_sampled = df.sample(frac=0.1, random_state=42)

    return df_sampled
