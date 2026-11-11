# Import library Streamlit untuk membuat antarmuka web
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- Simulasi Data & Model agar Aplikasi Mandiri ---
# Karena ini dijalankan di server terpisah, kita definisikan ulang data latih sederhana agar model & vectorizer aktif
data = {
    'teks_lowongan': [
        "Dibutuhkan segera IT Support. Syarat: lulusan SMK TKJ, paham routing Mikrotik, penempatan Jakarta Timur.",
        "KERJA MUDAH GAJI JUTAAN! Cukup balas chat dari rumah. Transfer biaya pendaftaran member 250rb ke rekening admin.",
        "Lowongan Network Engineer. Menguasai Cisco Packet Tracer, instalasi LAN, dan konfigurasi server.",
        "Dicari kandidat pegawai BUMN tanpa tes! Kuota terbatas. Bayar biaya akomodasi tiket pesawat 1 juta, nanti diganti.",
        "Dibutuhkan staf magang teknisi komputer, terbiasa merakit PC dan instalasi Windows. Kirim CV ke email resmi.",
        "Kerja sampingan ketik captcha dibayar dollar. Syarat bayar deposit awal $50."
    ],
    'label': [0, 1, 0, 1, 0, 1]
}
df_temp = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(df_temp['teks_lowongan'])
model = MultinomialNB()
model.fit(X_train_vec, df_temp['label'])
# --------------------------------------------------

st.set_page_config(page_title="Pendeteksi Loker Palsu", page_icon="🔎", layout="centered")

st.title("Pendeteksi Loker Palsu (Anti-Scam) 🔎")
st.write("Sistem Machine Learning untuk mendeteksi indikasi penipuan pada lowongan pekerjaan beserta alasan analisisnya.")

# Kolom input teks lowongan
teks_input = st.text_area("Teks Lowongan", placeholder="Paste teks lowongan di sini... (misal: syarat kerja, cara daftar, dll)", height=150)

# Kolom unggah screenshot / dokumen
file_input = st.file_uploader("Unggah Screenshot / File Lowongan (Opsional)", type=["png", "jpg", "jpeg", "pdf"])

if st.button("Mulai Analisis AI"):
    teks_untuk_dianalisis = teks_input.strip()
    catatan_tambahan = ""

    if file_input is not None:
        catatan_tambahan = "\n\n*[Sistem mendeteksi adanya lampiran file/gambar. Analisis saat ini berfokus pada teks yang dimasukkan.]*"

    if not teks_untuk_dianalisis:
        if file_input is not None:
            st.info("File berhasil diunggah! Mohon salin/ketik teks penting yang ada di dalam gambar/dokumen tersebut ke kolom teks untuk dianalisis oleh AI.")
        else:
            st.warning("Masukkan teks lowongan terlebih dahulu.")
    else:
        # Ekstraksi fitur teks menggunakan vectorizer
        teks_vec = vectorizer.transform([teks_untuk_dianalisis])

        # AI memberikan prediksi dan probabilitas
        hasil_prediksi = model.predict(teks_vec)[0]
        probabilitas = model.predict_proba(teks_vec)[0]

        prob_asli = probabilitas[0] * 100
        prob_palsu = probabilitas[1] * 100

        st.subheader("Hasil Analisis AI & Alasan Keyakinan")

        if hasil_prediksi == 1:
            st.error(f"⚠️ HATI-HATI! Loker ini terindikasi PALSU / PENIPUAN.")
            st.write(f"**Probabilitas Palsu:** {prob_palsu:.2f}% (Tinggi)")
            st.write(f"**Probabilitas Asli:** {prob_asli:.2f}%")
            st.write("**Alasan Analisis:** Sistem mendeteksi kecenderungan pola kalimat yang sering digunakan oleh penipu, seperti meminta biaya pendaftaran, janji gaji tidak realistis, atau prosedur rekrutmen tidak resmi.")
        else:
            st.success(f"✅ AMAN. Loker ini terindikasi ASLI.")
            st.write(f"**Probabilitas Asli:** {prob_asli:.2f}% (Tinggi)")
            st.write(f"**Probabilitas Palsu:** {prob_palsu:.2f}%")
            st.write("**Alasan Analisis:** Sistem menilai struktur teks ini aman dan serupa dengan karakteristik penulisan lowongan kerja formal/profesional.")

        if catatan_tambahan:
            st.markdown(catatan_tambahan)
