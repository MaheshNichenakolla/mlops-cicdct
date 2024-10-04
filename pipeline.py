import os
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import boto3
from datetime import datetime

# Constants
GITHUB_REPO_URL = 'https://github.com/MaheshNichenakolla/mlops-cicdct.git'  # Replace with your GitHub repo URL
LOCAL_REPO_PATH = '/home/ubuntu/pipeline'  # Replace with your local repo path
S3_BUCKET = 'mlpipelinetest'  # Replace with your S3 bucket name
S3_MODEL_PATH = 'models/'  # Path in the S3 bucket to store models

def fetch_data_from_github():
    """Fetch the latest data from the GitHub repository."""
    if not os.path.exists(LOCAL_REPO_PATH):
        subprocess.run(['git', 'clone', GITHUB_REPO_URL, LOCAL_REPO_PATH], check=True)
    else:
        subprocess.run(['git', '-C', LOCAL_REPO_PATH, 'pull'], check=True)

def load_data():
    """Load the CGPA dataset."""
    data_path = os.path.join(LOCAL_REPO_PATH, 'placement.csv')  # Adjust if your data file has a different name
    data = pd.read_csv(data_path)
    return data

def validate_data(data):
    """Perform basic data validation."""
    if data.isnull().values.any():
        raise ValueError("Data contains missing values!")
    return True

def train_model(data):
    """Train the linear regression model."""
    X = data[['cgpa']]  # Replace with actual feature names
    y = data['package']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Model trained with Mean Squared Error: {mse}')

    return model

def save_model_to_s3(model):
    """Save the trained model to S3."""
    model_name = f"model_{datetime.now().strftime('%d_%m')}.joblib"
    joblib.dump(model, model_name)

    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(model_name, S3_BUCKET, f"{S3_MODEL_PATH}{model_name}")
    print(f'Model saved to S3 as {model_name}')

def main():
    """Main function to run the ML pipeline."""
    # Fetch the latest data from GitHub
    fetch_data_from_github()
    
    # Load and validate data
    data = load_data()
    if validate_data(data):
        # Train model
        model = train_model(data)
        # Save the trained model to S3
        save_model_to_s3(model)

if __name__ == "__main__":
    main()
