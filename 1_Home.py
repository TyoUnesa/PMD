import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="Home", layout="wide")

st.title("Sistem Prediksi Kemenangan Pertandingan Mobile Legends Berbasis Data Turnamen")
st.image("S4_Projek_PMD2/logo ML (1).png")
st.markdown('<hr style="border: 2px solid orange;">', unsafe_allow_html=True)
st.markdown("""

Mobile Legends: Bang Bang (MLBB) merupakan salah satu e-sports terbesar dengan ekosistem kompetitif yang matang. Dalam dunia profesional, strategi dimulai bahkan sebelum pertandingan dimulai—yakni pada fase draft pick dan ban. Oleh karena itu, data pertandingan profesional menjadi sangat berharga untuk dianalisis.

Sistem ini bertujuan untuk membangun model prediksi kemenangan tim MLBB berbasis machine learning dengan algoritma Backpropagation, data yang digunakan adalah data pertandingan aktual dari berbagai liga dan turnamen internasional seperti MPL Indonesia Season 13, MPL Philippines Season 13, Games of the Future 2024, dan MDL Season 10.

*• Dengan mengolah data historis dari ratusan pertandingan, proyek ini bertujuan untuk*:
- Mengembangkan model klasifikasi yang mampu memprediksi tim pemenang berdasarkan fitur-fitur penting dalam hasil pertandingan.
- Mengidentifikasi pola strategis dan komposisi hero yang sering berkorelasi dengan kemenangan.
- Memberikan insight berbasis data kepada pelatih, analis, dan pemain profesional untuk mendukung pengambilan keputusan.

*• Beberapa fitur kunci yang diambil dari dataset meliputi*:
- Jumlah kill, death, dan assist tim
- Perolehan gold rata-rata per tim
- Durasi pertandingan

*• Model yang dihasilkan akan memiliki manfaat luas, seperti*:
- Analisis pertandingan profesional (post-match review)
- Pendamping analisa dalam sesi latihan scrim
- Prediksi hasil pertandingan untuk konten media, analisis turnamen, hingga pasar prediksi e-sports
- Analisa hasil pertandingan serta hal-hal yang dapat mempengaruhi kemenangan suatu tim
""")


