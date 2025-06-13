import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd



df_cleaned = pd.read_csv("df_cleaned.csv")
grouped = pd.read_csv("grouped.csv")  
grouped_pairs = pd.read_csv("grouped_pairs.csv")


st.title("Visualisasi Data")
st.markdown("<br><br><br>", unsafe_allow_html=True)


st.subheader("Kumpulan Data Tiap Player")
st.dataframe(df_cleaned)
st.markdown("<br><br>", unsafe_allow_html=True)


st.subheader("Kumpulan Data Tiap Tim ")
st.dataframe(grouped)
st.markdown("<br><br>", unsafe_allow_html=True)


st.subheader("Kumpulan Data Tiap Match")
st.dataframe(grouped_pairs)
st.markdown("<br><br><br>", unsafe_allow_html=True)



st.subheader("Distribusi Kemenangan (Win vs Lose)")
fig1, ax1 = plt.subplots()
sns.countplot(x='Win', data=grouped, ax=ax1)
ax1.set_title("Distribusi Kemenangan (Win vs Lose)")
ax1.set_xlabel("Label (0 = Lose, 1 = Win)")
ax1.set_ylabel("Jumlah")
st.pyplot(fig1)
st.markdown('<hr style="border: 2px solid orange;">', unsafe_allow_html=True)
st.markdown(""" Visualisasi tersebut menunjukkan distribusi kemenangan (Win) dan kekalahan (Lose) dalam bentuk diagram batang. 
            Label 0 mewakili kekalahan dan 1 mewakili kemenangan. Dari grafik, tampak bahwa jumlah kemenangan dan kekalahan yang seimbang, yaitu 406 data.   """)
st.markdown("<br><br><br>", unsafe_allow_html=True)


st.subheader("Distribusi Kemenangan Tim1 dari 406 Match")
fig3, ax3 = plt.subplots()
sns.countplot(x='Win_1', data=grouped_pairs, ax=ax3)
ax3.set_title("Distribusi Kemenangan Tim1 dari 406 Match")
ax3.set_xlabel("Label (0 = Lose, 1 = Win)")
ax3.set_ylabel("Jumlah")
st.pyplot(fig3)
st.markdown('<hr style="border: 2px solid orange;">', unsafe_allow_html=True)
st.markdown("""Visualisasi tersebut menunjukkan distribusi hasil pertandingan untuk Tim1 dari total 406 pertandingan, di mana label 0 menunjukkan kekalahan dan label 1 menunjukkan kemenangan. Dari grafik dapat dilihat bahwa jumlah kemenangan Tim1 (label 1) lebih tinggi dibandingkan jumlah kekalahan (label 0). 
            Hal ini menunjukkan bahwa secara keseluruhan, Tim1 lebih sering menang daripada kalah dalam dataset tersebut. Meskipun tidak sepenuhnya seimbang, distribusi ini masih tergolong wajar dan bisa memberikan informasi penting dalam analisis performa.""")
st.markdown("<br><br><br>", unsafe_allow_html=True)


st.subheader("Distribusi Fitur Numerik")
features = ['K_1', 'D_1', 'A_1', 'Gold_1', 'Level_1', 'Time_seconds_1', 'K_2', 'D_2', 'A_2', 'Gold_2', 'Level_2']
fig2, ax2 = plt.subplots(figsize=(12, 10))
grouped_pairs[features].hist(bins=15, figsize=(12, 10), color='skyblue', edgecolor='black')
plt.suptitle("Distribusi Fitur Numerik")
st.pyplot(plt.gcf())  
st.markdown("""
Visualisasi di atas menampilkan histogram distribusi dari fitur-fitur numerik yang berkaitan dengan dua tim (Tim1 dan Tim2) dalam dataset pertandingan.

Secara umum, kita dapat melihat bahwa:

- **Fitur K (Kill), D (Death), dan A (Assist)** untuk kedua tim menunjukkan distribusi yang condong ke kiri, artinya sebagian besar pemain mencetak jumlah yang relatif kecil dari KDA, dengan sedikit pemain yang memiliki skor tinggi.
- **Gold** dan **Time_seconds** juga menunjukkan distribusi miring ke kanan, yang mengindikasikan bahwa sebagian besar pemain mengumpulkan gold atau bermain dalam durasi waktu yang moderat, dengan hanya sebagian kecil yang memiliki nilai ekstrim tinggi.
- **Level** cenderung berada dalam kisaran yang sempit dan banyak berkumpul di angka maksimal (sekitar level 15), menunjukkan bahwa mayoritas pemain mencapai level maksimum dalam pertandingan.
- **Distribusi antar Tim1 dan Tim2** secara visual terlihat cukup mirip, yang menunjukkan tidak ada perbedaan signifikan dalam pola performa dasar antar kedua tim.
""")
st.markdown("<br><br><br>", unsafe_allow_html=True)
