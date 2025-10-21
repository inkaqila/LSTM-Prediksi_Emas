import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# 1. Baca dataset
df = pd.read_csv('XAU_1d_data.csv', sep=';')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 2. Ambil hanya kolom 'Close'
data = df[['Close']].values  # Shape: (N, 1)

# 3. Normalisasi data ke rentang [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)  # Shape: (N, 1)

# 4. Buat windowing: 60 hari -> prediksi hari ke-61
timesteps = 60
X, y = [], []

for i in range(timesteps, len(scaled_data)):
    X.append(scaled_data[i - timesteps:i, 0])  # 60 nilai sebelumnya
    y.append(scaled_data[i, 0])                # nilai ke-61

X = np.array(X)  # Shape: (samples, 60)
y = np.array(y)  # Shape: (samples,)

# 5. Reshape X ke format 3D untuk LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Shape: (samples, 60, 1)

# 6. Bagi data: 80% latih, 20% uji
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 7. Bangun model LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(timesteps, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 8. Latih model
print("🧠 Melatih model LSTM... (tunggu 2–5 menit)")
model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=1)

# 9. Simpan model dan scaler
model.save('model.h5')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model dan scaler berhasil disimpan!")