import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Judul halaman
st.title("📊 Dataset Harga Saham PT Mandiri Tbk (Bank Mandiri)")

# Deskripsi dataset
st.header("📝 Tentang Dataset")
st.write("""
Dataset ini berisi data historis harga saham PT Mandiri Tbk (Bank Mandiri) selama **20 tahun**. 
Data yang disediakan mencakup informasi seperti **Harga Penutupan (Close)**, **Harga Pembukaan (Open)**, 
**Harga Tertinggi (High)**, **Harga Terendah (Low)**, dan **Volume Perdagangan**.
""")
st.write("""
Dataset ini digunakan untuk melatih model prediksi harga saham menggunakan algoritma **LSTM** dan **GRU**. 
Sebelum digunakan, data dinormalisasi dan dipersiapkan untuk model time-series.
""")

# Load dataset


@st.cache
def load_data():
    # Ganti dengan path file dataset Anda
    data = pd.read_csv("dataset_mandiri_20tahun.csv")

    # Preprocess data: Remove commas and convert to float
    numeric_columns = ['Close']
    for col in numeric_columns:
        data[col] = data[col].replace({',': ''}, regex=True)  # Remove commas
        data[col] = pd.to_numeric(
            data[col], errors='coerce')  # Convert to numeric

    # Drop rows with NaN values in numeric columns
    data = data.dropna(subset=numeric_columns)

    # Sort data by date (assuming the date column is named 'Date')
    data['Date'] = pd.to_datetime(data['Date'])
    # Pastikan data diurutkan berdasarkan tanggal
    data = data.sort_values(by='Date')

    return data


data = load_data()

# Tampilkan data sebelum normalisasi
st.header("📋 Data Sebelum Normalisasi")
st.write("Berikut adalah 5 baris pertama dari dataset sebelum normalisasi:")
st.dataframe(data.head())

# Function to prepare data for LSTM/GRU


def prepare_data(df, column='Close', sequence_length=30):
    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[column].values.reshape(-1, 1))

    # Prepare the sequences for LSTM/GRU
    X, y = [], []
    for i in range(sequence_length, len(df)):
        X.append(df_scaled[i-sequence_length:i, 0])
        y.append(df_scaled[i, 0])

    X, y = np.array(X), np.array(y)

    # Reshape for LSTM/GRU model input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


# Prepare data for LSTM/GRU
st.header("📈 Persiapan Data untuk LSTM/GRU")
st.write("""
Data akan dipersiapkan untuk model LSTM/GRU dengan langkah-langkah berikut:
1. **Normalisasi**: Mengubah nilai data ke rentang 0 hingga 1.
2. **Pembuatan Sequence**: Membuat sequence data untuk input model time-series.
""")

# Pilih kolom yang akan digunakan
column = st.selectbox("Pilih Kolom untuk Prediksi", [
                      'Close'])
sequence_length = st.slider(
    "Panjang Sequence", min_value=10, max_value=100, value=30)

# Prepare data
X, y, scaler = prepare_data(
    data, column=column, sequence_length=sequence_length)

# Tampilkan hasil persiapan data
st.subheader("Hasil Persiapan Data")
st.write(f"**Input Data (X)**: {X.shape}")
st.write(f"**Target Data (y)**: {y.shape}")

# Visualisasi data sebelum dan setelah normalisasi
st.header("📊 Visualisasi Data Sebelum dan Sesudah Normalisasi")
st.write("Berikut adalah perbandingan data sebelum dan setelah normalisasi untuk kolom **{}**:".format(column))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sebelum Normalisasi")
    st.line_chart(data[column])

with col2:
    st.subheader("Setelah Normalisasi")
    st.line_chart(scaler.transform(data[column].values.reshape(-1, 1)))

# Penjelasan tambahan
st.write("""
Dari visualisasi di atas, dapat dilihat bahwa pola data tetap sama setelah normalisasi, 
namun nilai-nilainya diubah ke dalam rentang 0 hingga 1.
""")

# Tautan ke sumber data (opsional)
st.markdown("[📥 Download Dataset Contoh](https://www.example.com)")
