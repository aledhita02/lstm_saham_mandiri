import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Judul halaman
st.title("📊 Evaluasi Model Prediksi Harga Saham")

# Deskripsi halaman
st.write("""
Halaman ini menampilkan evaluasi performa model prediksi harga saham. 
Pilih model yang ingin dievaluasi (LSTM atau GRU), dan hasil evaluasi akan ditampilkan dalam bentuk grafik dan tabel.
""")

# Pilihan model
model_option = st.selectbox("Pilih Model untuk Evaluasi", ["LSTM", "GRU"])

# Muat dataset


@st.cache
def load_data():
    # Muat dataset
    data = pd.read_csv("dataset_mandiri_20tahun.csv")

    # Preprocessing data
    # Hapus karakter non-numerik (seperti koma) dan konversi ke float
    numeric_columns = ['Close', 'Open', 'High', 'Low', 'Volume']
    for col in numeric_columns:
        data[col] = data[col].replace({',': ''}, regex=True)  # Hapus koma
        data[col] = pd.to_numeric(
            data[col], errors='coerce')  # Konversi ke numeric

    # Drop baris dengan missing values
    data = data.dropna(subset=numeric_columns)

    return data


data = load_data()

# Tampilkan dataset
st.header("📋 Dataset Harga Saham")
st.write("Berikut adalah 5 baris pertama dari dataset:")
st.dataframe(data.head())

# Pilih kolom untuk prediksi
column = st.selectbox("Pilih Kolom untuk Prediksi", [
                      'Close', 'Open', 'High', 'Low', 'Volume'])

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[column].values.reshape(-1, 1))

# Bagi data menjadi data latih dan data uji
train_size = int(len(scaled_data) * 0.8)  # 80% data latih, 20% data uji
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Fungsi untuk membuat sequence data


def create_sequences(data, sequence_length=30):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# Siapkan data uji
sequence_length = 30
X_test, y_test = create_sequences(test_data, sequence_length)
# Reshape untuk LSTM/GRU
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Muat model yang dipilih
if model_option == "LSTM":
    model = load_model("lstm_model.h5")  # Muat model LSTM
    st.write("Model LSTM dipilih untuk evaluasi.")
elif model_option == "GRU":
    model = load_model("gru_model.h5")   # Muat model GRU
    st.write("Model GRU dipilih untuk evaluasi.")

# Lakukan prediksi menggunakan model yang dipilih
y_pred = model.predict(X_test).flatten()  # Prediksi model

# Denormalisasi data
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Hitung metrik evaluasi
mae = mean_absolute_error(y_test_actual, y_pred_actual)
mse = mean_squared_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred_actual)

# Tampilkan grafik untuk MAE dan MSE
st.header("📈 Grafik Evaluasi")

# Plot MAE dan MSE
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Grafik MAE
ax[0].bar(['MAE'], [mae], color='blue')
ax[0].set_title('Mean Absolute Error (MAE)')
ax[0].set_ylim(0, max(mae * 2, 1))  # Atur batas y-axis

# Grafik MSE
ax[1].bar(['MSE'], [mse], color='orange')
ax[1].set_title('Mean Squared Error (MSE)')
ax[1].set_ylim(0, max(mse * 2, 1))  # Atur batas y-axis

# Tampilkan grafik di Streamlit
st.pyplot(fig)

# Tampilkan tabel untuk MAE, MSE, RMSE, dan R2 score
st.header("📋 Tabel Evaluasi")

# Buat DataFrame untuk metrik evaluasi
evaluation_metrics = {
    "Metric": ["MAE", "MSE", "RMSE", "R2 Score"],
    "Value": [mae, mse, rmse, r2]
}

evaluation_df = pd.DataFrame(evaluation_metrics)

# Tampilkan tabel
st.table(evaluation_df)

# Penjelasan tambahan
st.write("""
### Penjelasan Metrik:
- **MAE (Mean Absolute Error)**: Rata-rata selisih absolut antara nilai sebenarnya dan prediksi.
- **MSE (Mean Squared Error)**: Rata-rata kuadrat selisih antara nilai sebenarnya dan prediksi.
- **RMSE (Root Mean Squared Error)**: Akar kuadrat dari MSE, memberikan interpretasi yang lebih baik dalam skala data.
- **R2 Score (Koefisien Determinasi)**: Mengukur seberapa baik model menjelaskan variasi data. Nilai berkisar antara 0 hingga 1, di mana 1 menunjukkan prediksi sempurna.
""")
