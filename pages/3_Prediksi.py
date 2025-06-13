import streamlit as st
import numpy as np
import pandas as pd


# fungsi backpro
dataset = pd.read_csv("grouped_pairs.csv")

X = dataset.drop(columns=['Win_1'])
y = dataset['Win_1']

def standarisasi_zscore(data):
    rata_rata = np.mean(data, axis=0)
    standar_deviasi = np.std(data, axis=0)
    if standar_deviasi.all() == 0:
        return np.zeros_like(data)
    return (data - rata_rata) / standar_deviasi

def standarisasi_zscore_prediksi(data, rata_rata_latih, std_dev_latih):
    return (data - rata_rata_latih) / std_dev_latih

X_normalized = standarisasi_zscore(X)

rata_rata_latih = np.mean(X, axis=0)
std_dev_latih = np.std(X, axis=0)

split_index = int(0.8 * len(X))
X_train = X_normalized[:split_index]
X_test = X_normalized[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

def inisialisasi_bobot(ukuran_layer, seed=42):
    np.random.seed(seed)
    bobot = []
    bias = []
    for i in range(len(ukuran_layer) - 1):
        matriks_bobot = np.random.uniform(-1, 1, (ukuran_layer[i + 1], ukuran_layer[i]))
        vektor_bias = np.ones((ukuran_layer[i + 1], 1))
        bobot.append(matriks_bobot)
        bias.append(vektor_bias)
    return bobot, bias

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def turunan_sigmoid(x):
    return x * (1 - x)

def propagasi_maju(input_data, bobot, bias):
    znet_hidden = np.dot(bobot[0], input_data) + bias[0]
    z_hidden = sigmoid(znet_hidden)
    znet_output = np.dot(bobot[1], z_hidden) + bias[1]
    y_output = sigmoid(znet_output)
    return znet_hidden, z_hidden, znet_output, y_output

def propagasi_mundur(input_data, target, bobot, bias, z_hidden, y_output, learning_rate):
    error = target - y_output
    sk = error * turunan_sigmoid(y_output)
    sj = np.dot(bobot[1].T, sk) * turunan_sigmoid(z_hidden)
    bobot[1] += learning_rate * np.dot(sk, z_hidden.T)
    bias[1] += learning_rate * sk
    bobot[0] += learning_rate * np.dot(sj, input_data.T)
    bias[0] += learning_rate * sj

# Inisialisasi
ukuran_layer = [X_train.shape[1], 2, 1]
bobot, bias = inisialisasi_bobot(ukuran_layer)
learning_rate = 0.1
epochs = 50

# Training
for epoch in range(epochs):
    for i in range(len(X_train)):
        input_data = X_train[i].reshape(-1, 1)
        target = y_train[i].reshape(-1, 1)
            
        znet_hidden, z_hidden, znet_output, y_output = propagasi_maju(input_data, bobot, bias)
        propagasi_mundur(input_data, target, bobot, bias, z_hidden, y_output, learning_rate)



#tata letak
st.set_page_config(layout="wide")
st.title("Prediksi Kemenangan Tim Mobile Legends")

def input_hero(team_name, hero_num, default_values=None):
    """Fungsi ini sekarang mengembalikan dictionary sederhana dengan opsi default."""
    st.markdown(f"**Hero {hero_num}**")
    col1, col2, col3, col4, col5 = st.columns(5)

    level_default = default_values.get('Level', 1) if default_values else 1
    k_default = default_values.get('K', 0) if default_values else 0
    d_default = default_values.get('D', 0) if default_values else 0
    a_default = default_values.get('A', 0) if default_values else 0
    gold_default = default_values.get('Gold', 1000) if default_values else 1000

    hero_data = {
        'Level': col1.number_input("Level", 1, 15, level_default, key=f"{team_name}_level_{hero_num}"),
        'K': col2.number_input("Kills", 0, None, k_default, key=f"{team_name}_K_{hero_num}"),
        'D': col3.number_input("Deaths", 0, None, d_default, key=f"{team_name}_D_{hero_num}"),
        'A': col4.number_input("Assists", 0, None, a_default, key=f"{team_name}_A_{hero_num}"),
        'Gold': col5.number_input("Gold", 1000, 30000, gold_default, key=f"{team_name}_Gold_{hero_num}")
    }
    return hero_data

default_tim_a = [
    {'Level': 14, 'K': 5, 'D': 1, 'A': 18, 'Gold': 8702},
    {'Level': 14, 'K': 8, 'D': 2, 'A': 19, 'Gold': 10414},
    {'Level': 15, 'K': 8, 'D': 1, 'A': 10, 'Gold': 11268},
    {'Level': 14, 'K': 12, 'D': 2, 'A': 14, 'Gold': 9346},
    {'Level': 15, 'K': 7, 'D': 3, 'A': 11, 'Gold': 10216}
]

default_tim_b = [
    {'Level': 13, 'K': 0, 'D': 5, 'A': 5, 'Gold': 8747},
    {'Level': 12, 'K': 1, 'D': 10, 'A': 4, 'Gold': 7204},
    {'Level': 11, 'K': 2, 'D': 8, 'A': 4, 'Gold': 6805},
    {'Level': 14, 'K': 4, 'D': 7, 'A': 2, 'Gold': 8904},
    {'Level': 11, 'K': 2, 'D': 10, 'A': 3, 'Gold': 6633}
]


col_title_a, col_waktu, col_title_b = st.columns([2.2, 1, 2.2])
with col_title_a:
    st.subheader("🔵 Tim A")
with col_waktu:
    st.subheader("Waktu Pertandingan")
    col_menit, col_detik = st.columns(2)
    menit = col_menit.number_input("Menit", 0, None, 15, key="waktu_menit")
    detik = col_detik.number_input("Detik", 0, 59, 0, key="waktu_detik")
with col_title_b:
    st.subheader("🔴 Tim B")

st.divider()

tim_a_data, tim_b_data = [], []
for i in range(1, 6):
    col_a, col_b = st.columns(2)
    with col_a:
        tim_a_data.append(input_hero("1", i, default_tim_a[i-1]))
    with col_b:
        tim_b_data.append(input_hero("2", i, default_tim_b[i-1]))

st.divider()

if st.button("Prediksi Pemenang", use_container_width=True):
    
    df_a = pd.DataFrame(tim_a_data)
    df_b = pd.DataFrame(tim_b_data)
    avg_a = df_a.mean()
    avg_b = df_b.mean()
    waktu_total_detik = (menit * 60) + detik
    
    data_prediksi_mentah = pd.DataFrame([{
        'K_1': avg_a['K'],
        'D_1': avg_a['D'],
        'A_1': avg_a['A'],
        'Gold_1': avg_a['Gold'],
        'Level_1': avg_a['Level'],
        'Time_seconds_1': waktu_total_detik,  
        'K_2': avg_b['K'],
        'D_2': avg_b['D'],
        'A_2': avg_b['A'],
        'Gold_2': avg_b['Gold'],
        'Level_2': avg_b['Level']
    }])
    
    st.write("Rata - Rata : ")
    st.dataframe(data_prediksi_mentah)

    data_prediksi_normalized = standarisasi_zscore_prediksi(data_prediksi_mentah, rata_rata_latih, std_dev_latih)
    data_prediksi_final = data_prediksi_normalized.to_numpy().flatten()
    
 
    _, _, _, hasil_prob = propagasi_maju(data_prediksi_final, bobot, bias)
    probabilitas_menang_A = hasil_prob[0][0]
    
 
    prediksi_label = "Tim A Menang" if probabilitas_menang_A >= 0.5 else "Tim B Menang"
    
    st.balloons()
    st.success("Prediksi berhasil dibuat!")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.metric(label="**Hasil Prediksi**", value=prediksi_label)
    if probabilitas_menang_A >= 0.5:
        with col_res2:
            st.metric(label="**Probabilitas Kemenangan Tim A**", value=f"{probabilitas_menang_A:.2%}")
    else:
        with col_res2:
            st.metric(label="**Probabilitas Kemenangan Tim B**", value=f"{1 - probabilitas_menang_A:.2%}")







st.header("📊 Input Rata-rata Tim")

col_avg_a, col_avg_b = st.columns(2)

with col_avg_a:
    st.subheader("🔵 Rata-rata Tim A")
    avg_K_A = st.number_input("Rata-rata Kills Tim A", 0.0, None, 6.0, step=0.1, key="avg_K_A")
    avg_D_A = st.number_input("Rata-rata Deaths Tim A", 0.0, None, 3.0, step=0.1, key="avg_D_A")
    avg_A_A = st.number_input("Rata-rata Assists Tim A", 0.0, None, 12.0, step=0.1, key="avg_A_A")
    avg_Gold_A = st.number_input("Rata-rata Gold Tim A", 1000.0, 30000.0, 9500.0, step=100.0, key="avg_Gold_A")
    avg_Level_A = st.number_input("Rata-rata Level Tim A", 1.0, 15.0, 14.0, step=0.1, key="avg_Level_A")

with col_avg_b:
    st.subheader("🔴 Rata-rata Tim B")
    avg_K_B = st.number_input("Rata-rata Kills Tim B", 0.0, None, 2.0, step=0.1, key="avg_K_B")
    avg_D_B = st.number_input("Rata-rata Deaths Tim B", 0.0, None, 7.0, step=0.1, key="avg_D_B")
    avg_A_B = st.number_input("Rata-rata Assists Tim B", 0.0, None, 5.0, step=0.1, key="avg_A_B")
    avg_Gold_B = st.number_input("Rata-rata Gold Tim B", 1000.0, 30000.0, 7500.0, step=100.0, key="avg_Gold_B")
    avg_Level_B = st.number_input("Rata-rata Level Tim B", 1.0, 15.0, 12.0, step=0.1, key="avg_Level_B")

st.subheader("⏱️ Waktu Pertandingan")
col_avg_menit, col_avg_detik = st.columns(2)
avg_menit = col_avg_menit.number_input("Menit", 0, None, 15, key="avg_waktu_menit")
avg_detik = col_avg_detik.number_input("Detik", 0, 59, 0, key="avg_waktu_detik")
avg_waktu_total_detik = (avg_menit * 60) + avg_detik

if st.button("Prediksi dari Input Rata-rata", use_container_width=True):
    data_avg_prediksi_mentah = pd.DataFrame([{
        'K_1': avg_K_A,
        'D_1': avg_D_A,
        'A_1': avg_A_A,
        'Gold_1': avg_Gold_A,
        'Level_1': avg_Level_A,
        'Time_seconds_1': avg_waktu_total_detik,
        'K_2': avg_K_B,
        'D_2': avg_D_B,
        'A_2': avg_A_B,
        'Gold_2': avg_Gold_B,
        'Level_2': avg_Level_B
    }])

    st.write("Rata-rata :")
    st.dataframe(data_avg_prediksi_mentah)

    data_avg_prediksi_normalized = standarisasi_zscore_prediksi(data_avg_prediksi_mentah, rata_rata_latih, std_dev_latih)
    data_avg_prediksi_final = data_avg_prediksi_normalized.to_numpy().flatten()

    _, _, _, hasil_prob_avg = propagasi_maju(data_avg_prediksi_final, bobot, bias)
    probabilitas_menang_A_avg = hasil_prob_avg[0][0]

    prediksi_label_avg = "Tim A Menang" if probabilitas_menang_A_avg >= 0.5 else "Tim B Menang"

    st.balloons()
    st.success("Prediksi dari input rata-rata berhasil dibuat!")

    col_res_avg1, col_res_avg2 = st.columns(2)
    with col_res_avg1:
        st.metric(label="**Hasil Prediksi**", value=prediksi_label_avg)
    if probabilitas_menang_A_avg >= 0.5:
        with col_res_avg2:
            st.metric(label="**Probabilitas Kemenangan Tim A**", value=f"{probabilitas_menang_A_avg:.2%}")
    else:
        with col_res_avg2:
            st.metric(label="**Probabilitas Kemenangan Tim B**", value=f"{1 - probabilitas_menang_A_avg:.2%}")
