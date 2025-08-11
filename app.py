import streamlit as st

# Judul halaman
st.title("Welcome!")

# Gambar header di tengah menggunakan HTML dan CSS
st.markdown(
    """
    <div style="display: flex; justify-content: center; margin-bottom: 50px;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ad/Bank_Mandiri_logo_2016.svg/2560px-Bank_Mandiri_logo_2016.svg.png" width="300">
    </div>
    """,
    unsafe_allow_html=True
)

# Deskripsi aplikasi
st.write("""
Selamat datang di halaman **Apps** aplikasi prediksi harga saham PT Mandiri Tbk (Bank Mandiri). 
Aplikasi ini dirancang untuk membantu investor dan analis keuangan dalam memprediksi pergerakan harga saham 
menggunakan model kecerdasan buatan (AI) berbasis **LSTM (Long Short-Term Memory)** dan **GRU (Gated Recurrent Unit)**.
""")

# Informasi tentang PT Mandiri Tbk
st.header("📊 Tentang PT Mandiri Tbk (Bank Mandiri)")
st.write("""
**PT Bank Mandiri Tbk** adalah salah satu bank terbesar di Indonesia yang menyediakan berbagai layanan perbankan, 
mulai dari perbankan ritel, korporasi, hingga syariah. Saham Bank Mandiri (kode saham: **BMRI**) 
tercatat di Bursa Efek Indonesia (BEI) dan menjadi salah satu saham blue-chip yang banyak diminati oleh investor.
""")

# Manfaat aplikasi
st.header("🛠️ Manfaat Aplikasi")
st.write("""
Aplikasi ini dapat membantu pengguna dalam:
- Memprediksi pergerakan harga saham BMRI secara akurat.
- Memberikan insight tentang tren harga saham menggunakan model AI.
- Membantu pengambilan keputusan investasi yang lebih terinformasi.
""")

# Teknologi yang digunakan
st.header("🤖 Teknologi yang Digunakan")
st.write("""
Aplikasi ini dibangun dengan:
- **Streamlit**: Framework untuk membuat antarmuka pengguna yang interaktif.
- **TensorFlow/Keras**: Library untuk membangun dan melatih model LSTM dan GRU.
- **Data Historis Saham**: Data harga saham BMRI dari sumber terpercaya.
""")

# Gambar tambahan di tengah
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://www.idx.co.id/media/4437/bank-mandiri.jpg" width="500">
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("Gedung PT Bank Mandiri Tbk")

# Penutup
st.write("""
Terima kasih telah menggunakan aplikasi ini. Semoga aplikasi ini dapat memberikan manfaat 
dan membantu Anda dalam melakukan analisis saham PT Mandiri Tbk.
""")

# Tautan eksternal (opsional)
st.markdown(
    "[📈 Lihat Data Saham BMRI di Bursa Efek Indonesia](https://www.idx.co.id)"
)