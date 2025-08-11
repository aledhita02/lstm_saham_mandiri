import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import timedelta
import plotly.express as px

# Fungsi untuk mempersiapkan data


def prepare_data(df, column='Close', sequence_length=30):
    # Remove commas and convert the column to float
    df[column] = df[column].replace({',': ''}, regex=True)  # Remove commas
    df[column] = pd.to_numeric(df[column], errors='coerce')  # Convert numeric

    # Drop rows with NaN values that might have been introduced
    df = df.dropna(subset=[column])

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

# Fungsi untuk memprediksi harga penutupan


def predict_next_days(model, last_sequence, days=7):
    predict_next = []
    for _ in range(days):
        next_pred = model.predict(last_sequence)
        predict_next.append(next_pred[0][0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1] = next_pred
    return np.array(predict_next)

# Fungsi untuk plotting data aktual dari dataset


def plot_actual(df, actual):
    # Buat DataFrame untuk plot
    plot_df = pd.DataFrame({
        'Date': df['Date'],
        'Close Price Actual': actual  # Beri nama kolom yang deskriptif
    })

    # Buat plot interaktif dengan Plotly
    fig = px.line(plot_df, x='Date', y='Close Price Actual',
                  labels={'Close Price Actual': 'Close Price', 'Date': 'Date'},
                  line_shape='linear')  # Garis lurus

    # Atur warna garis dan tambahkan label
    fig.update_traces(line=dict(color='blue', width=2),
                      name='Close Price Actual')  # Warna biru dan label

    # Tambahkan grid
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    # Tampilkan plot di Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Fungsi untuk plotting hasil prediksi vs data aktual terakhir


def plot_actual_vs_predicted(df, actual, predicted, prediction_dates, last_n_days=30):
    plt.figure(figsize=(15, 7))

    # Ambil data aktual terakhir (misalnya, 30 hari terakhir)
    actual_last_n = actual[-last_n_days:]
    dates_last_n = df['Date'].iloc[-last_n_days:]

    # Pastikan dates_last_n tidak kosong
    if len(dates_last_n) == 0:
        st.error("Tidak ada data aktual yang tersedia untuk ditampilkan.")
        return

    # Gabungkan data aktual terakhir dengan data prediksi
    combined_dates = np.concatenate([dates_last_n, prediction_dates])
    combined_values = np.concatenate([actual_last_n, predicted])

    # Plot data aktual dan prediksi sebagai satu garis yang terhubung
    plt.plot(combined_dates, combined_values,
             label='Data Aktual Harga Close', color='blue')

    # Tentukan indeks di mana prediksi dimulai
    split_index = len(dates_last_n)

    # Plot bagian prediksi dengan warna yang berbeda (misalnya, oranye)
    plt.plot(combined_dates[split_index-1:], combined_values[split_index-1:],
             color='orange', label='Predicted Close Price')

    # Garis pemisah antara data aktual dan prediksi
    plt.axvline(x=dates_last_n.iloc[-1], color='red',
                linestyle='--', label='Start of Prediction')

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.grid(True)
    plt.title(
        f'Actual Data (Last {last_n_days} Days) & Predict Future Close')
    plt.legend()
    st.pyplot(plt)


# Streamlit UI
st.title("Prediksi Harga Close Saham PT. Mandiri (Persero) Tbk (BMRI.JK)")

model_option = st.selectbox("Pilih Metode", ['LSTM', 'GRU'])
prediction_days = st.selectbox(
    "Pilih Jumlah Hari untuk Prediksi", [1, 3, 7, 10, 15, 20, 30])

# Upload file di main page
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")
if uploaded_file:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')  # Pastikan data diurutkan berdasarkan tanggal
    # Plot data aktual dari dataset
    st.subheader("Grafik Data Aktual Harga Close")
    actual_values = df['Close'].values  # Data aktual
    plot_actual(df, actual_values)

    st.write(df.head())  # Menampilkan preview data

    # Menyiapkan data untuk prediksi
    X, y, scaler = prepare_data(df, column='Close')

    # Memuat model
    if model_option == 'LSTM':
        model = load_model('lstm_model.h5')
    else:
        model = load_model('gru_model.h5')

    # Prediksi harga penutupan untuk beberapa hari ke depan
    # Ambil data terakhir dari test set
    last_sequence = X[-1].reshape(1, -1, 1)
    predict_values_scaled = predict_next_days(
        model, last_sequence, days=prediction_days)

    # Kembalikan ke skala asli
    predict_values_original = scaler.inverse_transform(
        predict_values_scaled.reshape(-1, 1))

    # Tanggal untuk prediksi
    last_date = df['Date'].iloc[-1]
    prediction_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=prediction_days, freq='D')

    # Menampilkan hasil prediksi dalam bentuk tabel
    predictions_df = pd.DataFrame({
        'Tanggal Prediksi': prediction_dates.strftime('%Y-%m-%d'),
        'Prediksi Harga Penutupan': predict_values_original.flatten()
    })
    st.subheader(
        f"Prediksi Harga Penutupan untuk {prediction_days} Hari ke Depan:")

    # Plot hasil prediksi vs data aktual terakhir
    actual_values = df['Close'].values  # Data aktual
    predicted_values = predict_values_original.flatten()  # Data prediksi
    plot_actual_vs_predicted(
        df, actual_values, predicted_values, prediction_dates, last_n_days=30)

    st.write(predictions_df)  # Display predictions as a table
