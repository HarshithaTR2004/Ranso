from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import pickle
from keras.models import load_model
from model_training import predict_file  # Import necessary function

app = Flask(__name__)  # Corrected to use __name__

# Paths for the model and scaler files
MODEL_SAVE_PATH = r"C:\Users\harsh\ransomware\backend\static\model"
RNN_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, 'rnn_model.h5')  # Updated to RNN model
SCALER_PATH = os.path.join(MODEL_SAVE_PATH, 'scaler.pkl')

# Load the RNN model and scaler when the app starts
try:
    rnn_model = load_model(RNN_MODEL_PATH)
except FileNotFoundError:
    rnn_model = None
    print("RNN model file not found. Please train the model first.")

try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = None
    print("Scaler file not found. Please train the model first.")

# List of important features to be used in predictions
important_features = [
    'pslist.avg_handlers', 'dlllist.ndlls',
    'dlllist.avg_dlls_per_proc', 'handles.nhandles',
    'handles.avg_handles_per_proc', 'handles.nevent', 'handles.nkey',
    'handles.nthread', 'handles.nsemaphore', 'handles.nsection',
    'handles.nmutant', 'svcscan.nservices', 'svcscan.kernel_drivers',
    'svcscan.shared_process_services'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read JSON file into a dictionary
        file_data = pd.read_json(file, typ='series').to_dict()

        # Convert the dictionary values to lists (to mimic rows in a DataFrame)
        for key in file_data.keys():
            file_data[key] = [file_data[key]]  # Wrap scalar values in lists

        # Convert the updated dictionary into a DataFrame
        file_data_df = pd.DataFrame(file_data)

        # Check for missing important features
        missing_features = [feature for feature in important_features if feature not in file_data_df.columns]
        if missing_features:
            return jsonify({'error': f'Missing important features: {", ".join(missing_features)}'})

        # Perform prediction using the loaded RNN model
        predictions = predict_file(file_data_df, rnn_model, scaler)

        # Convert predictions to readable labels
        predicted_labels = ['benign' if pred == 0 else 'ransomware' for pred in predictions.flatten()]

        return jsonify({'predictions': predicted_labels})

    except ValueError as e:
        return jsonify({'error': f'ValueError: {str(e)}'})
    except KeyError as e:
        return jsonify({'error': f'KeyError: {str(e)}'})
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'})

if __name__ == '__main__':  # Corrected to __name__ and __main__
    app.run(debug=True)
