import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import io

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
    data = pd.read_csv("dataset/dataset_mandiri_20tahun.csv")

    # Preprocess data: Remove commas and convert to float
    numeric_columns = ['Close']
    for col in numeric_columns:
        data[col] = data[col].replace({',': ''}, regex=True)  # Remove commas
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric

    # Drop rows with NaN values in numeric columns
    data = data.dropna(subset=numeric_columns)

    # Sort data by date (assuming the date column is named 'Date')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date')  # Pastikan data diurutkan berdasarkan tanggal

    return data

data = load_data()

# Fungsi untuk membuat template dataset
def create_template_dataset():
    # Membuat dataframe contoh
    template_data = {
        'Date': pd.date_range(start='2023-01-01', periods=5),
        'Open': [100.0, 101.5, 102.3, 103.1, 104.2],
        'High': [101.2, 102.8, 103.5, 104.0, 105.0],
        'Low': [99.5, 100.8, 101.5, 102.5, 103.5],
        'Close': [100.5, 101.8, 102.5, 103.5, 104.5],
        'Volume': [1000000, 1200000, 950000, 1100000, 1300000]
    }
    return pd.DataFrame(template_data)

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
column = st.selectbox("Pilih Kolom untuk Prediksi", ['Close'])
sequence_length = st.slider("Panjang Sequence", min_value=10, max_value=100, value=30)

# Prepare data
X, y, scaler = prepare_data(data, column=column, sequence_length=sequence_length)

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

# Download template dataset
st.header("📥 Download Template Dataset")
st.write("Anda dapat mengunduh template dataset untuk format yang sesuai:")

# Membuat template dataset
template_df = create_template_dataset()

# Mengubah dataframe ke format CSV untuk download
csv = template_df.to_csv(index=False).encode('utf-8')

# Tombol download
st.download_button(
    label="Download Template Dataset (CSV)",
    data=csv,
    file_name="stock_price_template.csv",
    mime="text/csv",
    help="Klik untuk mengunduh template dataset dalam format CSV"
)

# Tampilkan preview template
st.write("Preview Template Dataset:")
st.dataframe(template_df)