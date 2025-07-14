import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Extract
def extract_data(file_path):
    print("Extracting data...")
    return pd.read_csv(file_path)

        # Step 2: Transform
def transform_data(df):
    print("Transforming data...")
                
# Handle missing values
    df = df.dropna()

 # Encode categorical columns (if any)
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols)

# Scale numerical features
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
# Step 3: Load
def load_data(df, output_path):
    print("Loading data...")
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

# Main pipeline
def run_pipeline(input_path, output_path):
    df = extract_data(input_path)
    df_transformed = transform_data(df)
    load_data(df_transformed, output_path)

if __name__ == "__main__":
 # Sample file paths
    input_file = "input_data.csv"
    output_file = "cleaned_data.csv"
    run_pipeline(input_file, output_file)