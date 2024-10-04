from flask import Flask, request
import os
import subprocess

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def github_webhook():
    # Redeploy the ML pipeline
    subprocess.run(['python3', 'pipeline.py'], check=True)
    return '', 200

@app.route('/s3trigger', methods=['POST'])
def s3_trigger():
    # Redeploy the ML pipeline
    subprocess.run(['python3', 'pipeline.py'], check=True)
    return '', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
