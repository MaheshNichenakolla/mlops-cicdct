import os
import subprocess
import time
import signal
import boto3
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Global variable to keep track of the running process
inference_process = None

# Constants
S3_BUCKET = 'your-s3-bucket'  # Replace with your S3 bucket name
S3_MODEL_PATH = 'models/'  # Path in the S3 bucket to fetch models

def get_latest_model_name():
    """Get the latest model file name from S3."""
    s3_client = boto3.client('s3')

    # List objects in the specified S3 bucket and path
    response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_MODEL_PATH)
    model_files = []

    # Collect model file names
    if 'Contents' in response:
        for obj in response['Contents']:
            model_files.append(obj['Key'])

    # Filter and sort the model files based on date in the filename
    model_files = [f for f in model_files if f.endswith('.joblib')]
    model_files.sort(key=lambda x: datetime.strptime(x.split('/')[-1].split('_')[1:3], '%d_%m'))

    # Return the latest model file name
    return model_files[-1] if model_files else None

def download_model_from_s3(model_name):
    """Download the model from S3."""
    s3_client = boto3.client('s3')
    local_model_path = f"./{model_name}"
    s3_client.download_file(S3_BUCKET, model_name, local_model_path)
    return local_model_path

def load_model(model_name):
    """Load the model from the specified path."""
    return joblib.load(model_name)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict using the trained model."""
    input_data = request.json
    df = pd.DataFrame([input_data])

    # Get the latest model name
    latest_model_name = get_latest_model_name()
    if not latest_model_name:
        return jsonify({'error': 'No model found in S3'}), 404

    # Download and load the latest model
    local_model_path = download_model_from_s3(latest_model_name)
    model = load_model(local_model_path)

    # Make predictions
    prediction = model.predict(df)

    return jsonify({'prediction': prediction[0]})

def start_inference_service():
    """Start the Flask inference service."""
    global inference_process
    if inference_process is not None:
        os.kill(inference_process.pid, signal.SIGTERM)  # Kill the old process
    
    inference_process = subprocess.Popen(['python3', 'inference.py'])

def trigger_redeploy():
    """Trigger the redeployment of the inference service."""
    start_inference_service()

if __name__ == '__main__':
    while True:
        trigger_redeploy()
        time.sleep(60)  # Check every minute; adjust as necessary
