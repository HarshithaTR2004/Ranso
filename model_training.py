import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

def train_model():
    # Load the dataset
    data = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\HARSHI\HARSHI\SEM 5\SECURE CODING\webscanner\train_data1.csv")

    # Selecting all relevant features based on dynamic analysis
    important_features = [
        'pslist.nproc', 'pslist.nppid', 'pslist.avg_threads', 'pslist.nprocs64bit', 'pslist.avg_handlers', 
        'dlllist.ndlls', 'dlllist.avg_dlls_per_proc', 'handles.nhandles', 'handles.avg_handles_per_proc', 
        'handles.nport', 'handles.nfile', 'handles.nevent', 'handles.ndesktop', 'handles.nkey', 
        'handles.nthread', 'handles.ndirectory', 'handles.nsemaphore', 'handles.ntimer', 'handles.nsection', 
        'handles.nmutant', 'ldrmodules.not_in_load', 'ldrmodules.not_in_init', 'ldrmodules.not_in_mem', 
        'ldrmodules.not_in_load_avg', 'ldrmodules.not_in_init_avg', 'malfind.ninjections', 
        'malfind.commitCharge', 'malfind.protection', 'malfind.uniqueInjections', 'psxview.not_in_pslist', 
        'psxview.not_in_eprocess_pool', 'psxview.not_in_ethread_pool', 'psxview.not_in_pspcid_list', 
        'psxview.not_in_csrss_handles', 'psxview.not_in_session', 'psxview.not_in_deskthrd', 
        'modules.nmodules', 'svcscan.nservices', 'svcscan.kernel_drivers', 'svcscan.fs_drivers', 
        'svcscan.process_services', 'svcscan.shared_process_services', 'svcscan.interactive_process_services', 
        'svcscan.nactive', 'callbacks.ncallbacks', 'callbacks.nanonymous', 'callbacks.ngeneric'
    ]

    # Check if 'Class' exists in the DataFrame
    if 'Class' not in data.columns:
        print("Error: 'Class' column not found in the dataset.")
        return []

    # Extract features and target
    X = data[important_features]
    y = data['Class']

    # Encode the target variable to numeric
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshaping for RNN (adding a time-step dimension)
    X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    # Use Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(X_scaled.shape[0])
    train_preds = np.zeros(X_scaled.shape[0])  # For tracking training set predictions

    # Train the model using cross-validation
    for train_index, valid_index in skf.split(X_scaled, y_encoded):
        X_train, X_valid = X_scaled[train_index], X_scaled[valid_index]
        y_train, y_valid = y_encoded[train_index], y_encoded[valid_index]

        # Build the RNN model
        model = Sequential()
        model.add(SimpleRNN(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(SimpleRNN(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(len(label_encoder.classes_), activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Early stopping and learning rate reduction to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

        # Train the model
        model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, batch_size=32,
                  callbacks=[early_stopping, reduce_lr])

        # Make predictions on the validation set
        oof_preds[valid_index] = model.predict(X_valid).argmax(axis=1)

        # Store training predictions for this fold
        train_preds[train_index] = model.predict(X_train).argmax(axis=1)

    # Evaluate the model
    accuracy = accuracy_score(y_encoded, oof_preds)
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")

    # Print confusion matrix and classification report for validation set
    conf_matrix_val = confusion_matrix(y_encoded, oof_preds)
    print("Validation Set Confusion Matrix:")
    print(conf_matrix_val)

    # Fix for classification report by converting labels to strings
    class_labels_str = [str(cls) for cls in label_encoder.classes_]
    
    class_report_val = classification_report(y_encoded, oof_preds, target_names=class_labels_str)
    print("Validation Set Classification Report:")
    print(class_report_val)

    # Print confusion matrix and classification report for training set
    conf_matrix_train = confusion_matrix(y_encoded, train_preds)
    print("Training Set Confusion Matrix:")
    print(conf_matrix_train)

    class_report_train = classification_report(y_encoded, train_preds, target_names=class_labels_str)
    print("Training Set Classification Report:")
    print(class_report_train)

    # Save the model using Keras
    MODEL_SAVE_PATH = r"C:\Users\harsh\ransomware\backend\static\model"
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    model_save_path = os.path.join(MODEL_SAVE_PATH, 'rnn_model.h5')
    model.save(model_save_path)
    print("RNN model saved to", model_save_path)

    # Save the scaler
    with open(os.path.join(MODEL_SAVE_PATH, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to", os.path.join(MODEL_SAVE_PATH, 'scaler.pkl'))

def predict_file(input_data, model, scaler):
    # Select only the important features
    important_features = [
        'pslist.nproc', 'pslist.nppid', 'pslist.avg_threads', 'pslist.nprocs64bit', 'pslist.avg_handlers', 
        'dlllist.ndlls', 'dlllist.avg_dlls_per_proc', 'handles.nhandles', 'handles.avg_handles_per_proc', 
        'handles.nport', 'handles.nfile', 'handles.nevent', 'handles.ndesktop', 'handles.nkey', 
        'handles.nthread', 'handles.ndirectory', 'handles.nsemaphore', 'handles.ntimer', 'handles.nsection', 
        'handles.nmutant', 'ldrmodules.not_in_load', 'ldrmodules.not_in_init', 'ldrmodules.not_in_mem', 
        'ldrmodules.not_in_load_avg', 'ldrmodules.not_in_init_avg', 'malfind.ninjections', 
        'malfind.commitCharge', 'malfind.protection', 'malfind.uniqueInjections', 'psxview.not_in_pslist', 
        'psxview.not_in_eprocess_pool', 'psxview.not_in_ethread_pool', 'psxview.not_in_pspcid_list', 
        'psxview.not_in_csrss_handles', 'psxview.not_in_session', 'psxview.not_in_deskthrd', 
        'modules.nmodules', 'svcscan.nservices', 'svcscan.kernel_drivers', 'svcscan.fs_drivers', 
        'svcscan.process_services', 'svcscan.shared_process_services', 'svcscan.interactive_process_services', 
        'svcscan.nactive', 'callbacks.ncallbacks', 'callbacks.nanonymous', 'callbacks.ngeneric'
    ]
    
    # Ensure the input data contains the necessary features
    input_data = input_data[important_features]

    # Scale the features
    scaled_data = scaler.transform(input_data)

    # Reshape for RNN input (adding the time-step dimension)
    scaled_data = scaled_data.reshape(scaled_data.shape[0], 1, scaled_data.shape[1])

    # Make predictions
    predictions = model.predict(scaled_data)
    return predictions.argmax(axis=1)  # Return the class with the highest probability

if __name__ == "__main__":
    train_model()
