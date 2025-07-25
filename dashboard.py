# import streamlit as st
# import pandas as pd
# from datetime import datetime, date
# from inference import predict_batch
# import time
# import plotly.express as px
# import re

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import re

st.set_page_config(page_title="Customer Service Analytic Playground", layout="wide")

st.markdown("""
    <style>
        .centered-title {text-align: center; font-size: 2.1em; font-weight: bold; padding-bottom: 0.2em;}
        .disclaimer {text-align: center; color: #f2bc4c; font-size: 1.05em; padding-bottom: 1.5em;}
    </style>
    <div class="centered-title">📊 Customer Service Analytic Playground</div>
""", unsafe_allow_html=True)

  # <div class="disclaimer">
    #     ⚠️ <b>Disclaimer:</b> Ini adalah playground analytic. Hasil analisis dapat berbeda tergantung data aktual.<br>
    #     Anda bisa filter, visualisasi, dan download hasil data!
    # </div>

# === Fungsi natural sort untuk kategori dsb ===
def natural_sort_key(s):
    match = re.match(r'^\s*(\d+)', str(s))
    return int(match.group(1)) if match else float('inf')

# === LOAD DATA ===
@st.cache_data
def load_sample():
    # Buat data sample jika user tidak upload file
    data = {
        "Tanggal Tiket": pd.date_range("2024-07-01", periods=50, freq="D"),
        "Kategori": ["1. Kendala KYC", "2. Error Aplikasi", "3. Promo", "4. Reksadana", "5. Lainnya"]*10,
        "Sub Kategori": ["Login", "Registrasi", "Voucher", "Pembelian", "Feedback"]*10,
        "Sub Askes": ["App", "Web", "CS", "Promo", "Email"]*10,
        "Status": ["Open", "Closed", "In Progress", "Follow Up", "Waiting"]*10,
        "SLA": ["5 hari", "3 hari", "2 hari", "4 hari", "1 hari"]*10,
        "Pengaduan": [f"Sample complaint {i+1}" for i in range(50)],
    }
    df = pd.DataFrame(data)
    return df

st.write("Upload file pengaduan (CSV/XLSX) atau gunakan data sample di bawah untuk eksplorasi analisis.")
uploaded_file = st.file_uploader("Upload file pengaduan (CSV/XLSX)", type=["csv", "xlsx"])
if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"File tidak bisa dibaca: {e}")
        df = load_sample()
else:
    df = load_sample()

if len(df) > 20000:
    st.warning(f"Data terlalu besar ({len(df)} baris), Streamlit mungkin lambat. Sebaiknya <20.000 baris untuk responsif.")
# --- Normalisasi kolom agar filter rapi ---
filter_columns = ["Kategori", "Sub Kategori", "Sub Askes", "Status", "SLA"]
for col in filter_columns:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.strip()
            .str.title()
            .replace("Nan", None)
            .replace("None", None)
        )

# --- Filter Data ---
st.markdown("### 🔎 Filter Data")

if "Tanggal Tiket" in df.columns:
    df["Tanggal Tiket"] = pd.to_datetime(df["Tanggal Tiket"], errors="coerce")
    tgl_min, tgl_max = df["Tanggal Tiket"].min(), df["Tanggal Tiket"].max()
    date_range = st.date_input(
        "Tanggal Tiket", value=(tgl_min, tgl_max), min_value=tgl_min, max_value=tgl_max)
    df = df[df["Tanggal Tiket"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))]

if "Kategori" in df.columns:
    kategori_opsi = [x for x in df["Kategori"].dropna().unique() if x]
    kategori_opsi = sorted(kategori_opsi, key=natural_sort_key)
    kategori_pilih = st.multiselect("Kategori", kategori_opsi, default=kategori_opsi)
    df = df[df["Kategori"].isin(kategori_pilih)]

if "Sub Kategori" in df.columns:
    subkat_opsi = [x for x in df["Sub Kategori"].dropna().unique() if x]
    subkat_pilih = st.multiselect("Sub Kategori", subkat_opsi, default=subkat_opsi)
    df = df[df["Sub Kategori"].isin(subkat_pilih)]

if "Sub Askes" in df.columns:
    subaskes_opsi = [x for x in df["Sub Askes"].dropna().unique() if x]
    subaskes_pilih = st.multiselect("Sub Askes", subaskes_opsi, default=subaskes_opsi)
    df = df[df["Sub Askes"].isin(subaskes_pilih)]

if "Status" in df.columns:
    status_opsi = [x for x in df["Status"].dropna().unique() if x]
    status_pilih = st.multiselect("Status", status_opsi, default=status_opsi)
    df = df[df["Status"].isin(status_pilih)]

if "SLA" in df.columns:
    sla_opsi = [x for x in df["SLA"].dropna().unique() if x]
    sla_pilih = st.multiselect("SLA", sla_opsi, default=sla_opsi)
    df = df[df["SLA"].isin(sla_pilih)]

# --- Tampilkan hasil filter dan chart ---
st.write(f"### Hasil Filter: {len(df)} data")
st.dataframe(df, use_container_width=True, height=500)

# --- Chart Otomatis ---
if "Tanggal Tiket" in df.columns and not df.empty:
    count_per_date = df.groupby("Tanggal Tiket").size().reset_index(name="Jumlah Komplain")
    fig = px.bar(count_per_date, x="Tanggal Tiket", y="Jumlah Komplain", title="Jumlah Komplain per Tanggal Tiket")
    st.plotly_chart(fig, use_container_width=True)

if "Kategori" in df.columns and not df.empty:
    count_per_kat = df.groupby("Kategori").size().reset_index(name="Jumlah Komplain")
    count_per_kat = count_per_kat.sort_values(by="Kategori", key=lambda x: x.map(natural_sort_key))
    fig = px.bar(count_per_kat, x="Kategori", y="Jumlah Komplain", title="Jumlah Komplain per Kategori")
    st.plotly_chart(fig, use_container_width=True)

if "Sub Kategori" in df.columns and not df.empty:
    count_per_subkat = df.groupby("Sub Kategori").size().reset_index(name="Jumlah Komplain")
    fig = px.bar(count_per_subkat, x="Sub Kategori", y="Jumlah Komplain", title="Jumlah Komplain per Sub Kategori")
    st.plotly_chart(fig, use_container_width=True)

if "Status" in df.columns and not df.empty:
    count_per_status = df.groupby("Status").size().reset_index(name="Jumlah Komplain")
    fig = px.pie(count_per_status, names="Status", values="Jumlah Komplain", title="Proporsi Komplain per Status")
    st.plotly_chart(fig, use_container_width=True)

if "SLA" in df.columns and not df.empty:
    count_per_sla = df.groupby("SLA").size().reset_index(name="Jumlah Komplain")
    fig = px.bar(count_per_sla, x="SLA", y="Jumlah Komplain", title="Jumlah Komplain per SLA")
    st.plotly_chart(fig, use_container_width=True)

st.download_button(
    "⬇️ Download Hasil Filter (CSV)",
    df.to_csv(index=False).encode("utf-8"),
    file_name="hasil_filter_playground.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("Developed by Irfan Karim • Powered by AI Auto-Labeling | Streamlit UI v4")
