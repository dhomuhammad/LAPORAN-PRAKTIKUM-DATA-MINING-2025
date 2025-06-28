import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Memuat dataset
df = pd.read_csv("1heart_2020_cleaned.csv")

# Mengkodekan kolom Smoking (Yes/No) menjadi numerik (1/0)
label_encoder = LabelEncoder()
df["Smoking"] = label_encoder.fit_transform(df["Smoking"])  # Yes=1, No=0

# Memilih fitur dan target untuk klasifikasi
fitur = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]
target = "Smoking"

# Menyiapkan data untuk model
X = df[fitur]
y = df[target]

# Melatih model regresi logistik
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Judul aplikasi Streamlit
st.title("Prediksi Status Merokok Berdasarkan Kesehatan")

# Kolom input untuk pengguna
st.header("Masukkan Data Kesehatan")
bmi = st.number_input("BMI (Indeks Massa Tubuh)", min_value=10.0, max_value=60.0, value=25.0)
kesehatan_fisik = st.number_input("Jumlah Hari Kesehatan Fisik Buruk (0-30)", min_value=0.0, max_value=30.0, value=0.0)
kesehatan_mental = st.number_input("Jumlah Hari Kesehatan Mental Buruk (0-30)", min_value=0.0, max_value=30.0, value=0.0)
waktu_tidur = st.number_input("Jam Tidur per Hari", min_value=0.0, max_value=24.0, value=7.0)

# Tombol prediksi
if st.button("Prediksi Status Merokok"):
    # Melakukan prediksi
    data_input = np.array([[bmi, kesehatan_fisik, kesehatan_mental, waktu_tidur]])
    prediksi = model.predict(data_input)[0]
    probabilitas = model.predict_proba(data_input)[0]
    
    # Mengubah prediksi numerik kembali ke label
    status = "Perokok" if prediksi == 1 else "Bukan Perokok"
    probabilitas_perokok = round(probabilitas[1] * 100, 2)
    
    # Menampilkan hasil
    st.success(f"Prediksi: {status}")
    st.write(f"Probabilitas sebagai Perokok: {probabilitas_perokok}%")