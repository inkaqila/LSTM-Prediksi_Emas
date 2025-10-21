import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# --- Load model & scaler ---
try:
    model = load_model('model.h5')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
    raise

# --- Load and preprocess data ---
try:
    df = pd.read_csv('XAU_1d_data.csv', sep=';')
    if 'Date' not in df.columns:
        raise ValueError("Kolom 'Date' tidak ditemukan. Kolom yang tersedia: " + str(df.columns.tolist()))
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    if 'Close' not in df.columns:
        raise ValueError("Kolom 'Close' tidak ditemukan. Kolom yang tersedia: " + str(df.columns.tolist()))

    close_data = df[['Close']].values  # shape (N,1)
    scaled_data = scaler.transform(close_data)  # shape (N,1)
    timesteps = 60
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate')
def evaluate():
    try:
        total_len = len(scaled_data)
        if total_len < timesteps + 10:
            return jsonify({'error': 'Dataset too small for evaluation'}), 400

        split = int(0.8 * total_len)
        if split + timesteps > total_len:
            split = total_len - timesteps - 1
            if split < timesteps:
                return jsonify({'error': 'Dataset too small after split adjustment'}), 400

        # --- Prepare test sequences ---
        # We want the test part to start from index `split`
        X_test_base = scaled_data[split - timesteps:]  # start earlier so we can make first window
        X_test_seq = []
        for i in range(timesteps, len(X_test_base)):
            X_test_seq.append(X_test_base[i - timesteps:i])  # each item shape (timesteps, 1)
        X_test_seq = np.array(X_test_seq)  # shape (num_samples, timesteps, 1)

        if X_test_seq.size == 0:
            return jsonify({'error': 'Not enough data to create test sequences'}), 400

        # --- Predictions on test set ---
        y_pred_scaled = model.predict(X_test_seq, verbose=0)  # shape (num_samples, 1)
        # ensure shape (num_samples,1) before inverse
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_pred = scaler.inverse_transform(y_pred_scaled)  # shape (num_samples, 1)

        # Actual y values corresponding to predictions start at index `split`
        y_test_actual = close_data[split: split + len(y_pred)]  # shape (num_samples,1)

        # --- Metrics ---
        rmse = float(np.sqrt(mean_squared_error(y_test_actual, y_pred)))
        mae = float(mean_absolute_error(y_test_actual, y_pred))

        # --- Prepare plotting data ---
        # For the main "last 100" actual chart
        last_100 = df['Close'][-100:]
        dates_last100 = [str(d)[:10] for d in last_100.index]
        actual_last100 = last_100.tolist()

        # Dates that correspond to y_pred (start at index `split`)
        dates_pred = [str(d)[:10] for d in df.index[split: split + len(y_pred)]]
        predicted_list = y_pred.flatten().tolist()

        # --- 10-day ahead forecast (recursive) ---
        # start from last timesteps of scaled_data
        last_sequence = scaled_data[-timesteps:].reshape(1, timesteps, 1)  # shape (1,timesteps,1)
        current_seq = last_sequence.copy()
        future_preds = []

        for _ in range(10):
            pred_scaled = model.predict(current_seq, verbose=0)  # shape (1,1)
            # ensure shape (1,1)
            if pred_scaled.ndim == 1:
                pred_scaled = pred_scaled.reshape(1, 1)
            future_preds.append(pred_scaled[0, 0])
            # append new pred to current_seq correctly as 3D array
            # remove first timestep, concat new pred (1,1,1) on axis=1
            new_val = pred_scaled.reshape(1, 1, 1)  # shape (1,1,1)
            current_seq = np.concatenate((current_seq[:, 1:, :], new_val), axis=1)  # keep shape (1,timesteps,1)

        future_preds = np.array(future_preds).reshape(-1, 1)
        future_prices = scaler.inverse_transform(future_preds).flatten().tolist()

        last_date = df.index[-1]
        forecast_dates = [(last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in range(10)]

        return jsonify({
            'rmse': rmse,
            'mae': mae,
            'dates_actual': dates_last100,
            'actual': actual_last100,
            'dates_pred': dates_pred,
            'predicted': predicted_list,
            'forecast_dates': forecast_dates,
            'forecast_prices': future_prices
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_one_day')
def predict_one_day():
    try:
        last_sequence = scaled_data[-timesteps:].reshape(1, timesteps, 1)
        pred_scaled = model.predict(last_sequence, verbose=0)
        if pred_scaled.ndim == 1:
            pred_scaled = pred_scaled.reshape(1, 1)
        pred_actual = scaler.inverse_transform(pred_scaled)[0, 0]

        today = datetime.now()
        next_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')

        return jsonify({
            'date': next_date,
            'price': float(pred_actual),
            'message': f'Prediksi harga emas pada {next_date} adalah {pred_actual:.2f} USD'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_metrics')
def save_metrics():
    try:
        total_len = len(scaled_data)
        if total_len < timesteps + 10:
            return "❌ Dataset too small for metrics", 400

        split = int(0.8 * total_len)
        if split + timesteps > total_len:
            split = total_len - timesteps - 1

        X_test_base = scaled_data[split - timesteps:]
        X_test_seq = []
        for i in range(timesteps, len(X_test_base)):
            X_test_seq.append(X_test_base[i - timesteps:i])
        X_test_seq = np.array(X_test_seq)

        y_pred_scaled = model.predict(X_test_seq, verbose=0)
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_actual = close_data[split: split + len(y_pred)]

        rmse = float(np.sqrt(mean_squared_error(y_test_actual, y_pred)))
        mae = float(mean_absolute_error(y_test_actual, y_pred))

        os.makedirs('static', exist_ok=True)
        plt.figure(figsize=(8, 4))
        bars = plt.bar(['RMSE', 'MAE'], [rmse, mae])
        plt.title('Metrik Evaluasi Model LSTM')
        plt.ylabel('Nilai Error')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('static/rmse_plot.png')
        plt.close()

        return "✅ Grafik metrik berhasil disimpan di static/rmse_plot.png"
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

