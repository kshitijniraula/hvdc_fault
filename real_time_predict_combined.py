import numpy as np
import joblib
import json
import sys
import os
import time
import warnings

def real_time_prediction():
    # Suppress UserWarning from scikit-learn
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load models, scaler, and metadata on every call
    start_load_time = time.perf_counter()
    try:
        model_catboost = joblib.load('catboost_fault_model.joblib')
        model_xgboost = joblib.load('xgboost_fault_model.joblib')
        scaler = joblib.load('feature_scaler.joblib')
        with open('metadata_features_labels.json', 'r') as f:
            metadata = json.load(f)
            label_to_numeric_map = metadata['label_to_numeric']
    except FileNotFoundError:
        print(json.dumps({"status": "error", "message": "Required model or metadata files not found."}))
        return
    load_time_ms = (time.perf_counter() - start_load_time) * 1000

    # Get feature vector and true label from command-line arguments
    if len(sys.argv) < 3:
        print(json.dumps({"status": "error", "message": "Usage: python script.py <feature_vector_str> <true_label_str_raw>"}))
        return

    feature_vector_str = sys.argv[1]
    true_label_str_raw = sys.argv[2]
    
    # Process the input data
    start_pred_time = time.perf_counter()
    feature_vector = np.array([float(val) for val in feature_vector_str.split(',')]).reshape(1, -1)
    
    # Scale the features
    scaled_features = scaler.transform(feature_vector)
    
    # Map the true label for comparison
    if true_label_str_raw == 'f_1p':
        true_label_str = 'Pole1_Pos_Fault'
    elif true_label_str_raw == 'f_1n':
        true_label_str = 'Pole1_Neg_Fault'
    elif true_label_str_raw == 'f_2p':
        true_label_str = 'Pole2_Pos_Fault'
    elif true_label_str_raw == 'f_2n':
        true_label_str = 'Pole2_Neg_Fault'
    elif true_label_str_raw == 'f_3p':
        true_label_str = 'Pole3_Pos_Fault'
    elif true_label_str_raw == 'f_3n':
        true_label_str = 'Pole3_Neg_Fault'
    elif true_label_str_raw == 'f_pp':
        true_label_str = 'Pole_to_Pole_Fault'
    else:
        true_label_str = true_label_str_raw

    true_label_numeric = label_to_numeric_map.get(true_label_str, -1)
    numeric_to_label_map = {v: k for k, v in label_to_numeric_map.items()}

    # Perform predictions
    catboost_pred = model_catboost.predict(scaled_features)[0][0]
    xgboost_pred = model_xgboost.predict(scaled_features)[0]
    
    prediction_time_ms = (time.perf_counter() - start_pred_time) * 1000

    # Create the response dictionary
    response = {
        "status": "ok",
        "load_time_ms": load_time_ms,
        "prediction_time_ms": prediction_time_ms,
        "catboost_pred": numeric_to_label_map.get(catboost_pred, 'Unknown'),
        "xgboost_pred": numeric_to_label_map.get(xgboost_pred, 'Unknown'),
        "true_label": true_label_str,
        # Convert booleans to integers to be JSON serializable
        "is_catboost_correct": int(catboost_pred == true_label_numeric),
        "is_xgboost_correct": int(xgboost_pred == true_label_numeric),
    }
    print(json.dumps(response))

if __name__ == '__main__':
    real_time_prediction()