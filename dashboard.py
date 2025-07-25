import streamlit as st
import pandas as pd
from datetime import datetime, date
from inference import predict_batch
import time
import plotly.express as px
import re

def natural_sort_key(s):
    # Ekstrak angka paling depan pada string (misal '1. Kendala KYC' jadi 1)
    match = re.match(r'^\s*(\d+)', str(s))
    return int(match.group(1)) if match else float('inf')

st.set_page_config(page_title="Customer Service Auto-Label Dashboard", layout="wide")

# ===== Title & Disclaimer =====
st.markdown(
    """
    <style>
        .centered-title {
            text-align: center;
            font-size: 2.1em;
            font-weight: bold;
            padding-bottom: 0.2em;
        }
        .disclaimer {
            text-align: center;
            color: #f2bc4c;
            font-size: 1.05em;
            padding-bottom: 1.5em;
        }
    </style>
    <div class="centered-title">
        🧑‍💻 Customer Service Auto-Label Dashboard
    </div>
    <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> Hasil prediksi model bersifat otomatis dan bisa saja tidak akurat. 
        Harap selalu lakukan pengecekan/validasi manual sebelum mengambil keputusan.
    </div>
    """,
    unsafe_allow_html=True
)

mode = st.sidebar.radio("Pilih Mode", ["Manual", "Upload File","Playground Analytic"])

# ===================== MODE MANUAL ===========================
if mode == "Manual":
    st.subheader("Tambah Pengaduan Manual")

    # --- ALWAYS render input form above ---
    with st.form(key="input_form", clear_on_submit=True):
        pengaduan_manual = st.text_area(
            "📝 Pengaduan Customer (1 baris = 1 pengaduan)",
            height=130,
            placeholder="Contoh:\nuser mengeluh karena aplikasi lemot\ngagal KYC tidak bisa login\n..."
        )
        tanggal_ticket = st.date_input("📅 Tanggal Tiket", value=date.today())
        jam_ticket = st.time_input("⏰ Jam Tiket", value=datetime.now().time())
        sla_manual = st.text_input("⏳ SLA", value="5 hari")
        status_manual = st.selectbox("📌 Status", ["Open", "Closed", "In Progress"])
        tanggal_closed = st.date_input("📅 Tanggal Closed", value=date.today())
        col_pred, col_progress = st.columns([1, 0.2])
        with col_pred:
            pred_button = st.form_submit_button("Prediksi Pengaduan")
        with col_progress:
            progress_placeholder = st.empty()

    # --- Session state to store ALL manual dataframes (appendable) ---
    if "manual_df" not in st.session_state or st.session_state["manual_df"] is None:
        st.session_state["manual_df"] = pd.DataFrame()

    # --- After submit, append new rows to df (not overwrite!) ---
    if pred_button and pengaduan_manual.strip():
        pengaduan_list = [line.strip() for line in pengaduan_manual.split('\n') if line.strip() != ""]
        if pengaduan_list:
            # Progress bar
            for i in range(10):
                time.sleep(0.04)
                progress_placeholder.progress((i+1)*10)
            preds = predict_batch(pengaduan_list)
            table_data = {
                "Pengaduan": pengaduan_list,
                "Tanggal Tiket": [tanggal_ticket]*len(pengaduan_list),
                "Pukul Chat  (WIB)": [jam_ticket.strftime('%H:%M')]*len(pengaduan_list),
                "SLA": [sla_manual]*len(pengaduan_list),
                "Status": [status_manual]*len(pengaduan_list),
                "Tanggal Closed": [tanggal_closed]*len(pengaduan_list)
            }
            for col in preds.columns:
                table_data[col] = preds[col].values
            hasil_df = pd.DataFrame(table_data)
            # --- Append, not replace ---
            st.session_state["manual_df"] = pd.concat(
                [st.session_state["manual_df"], hasil_df], ignore_index=True
            )
            progress_placeholder.empty()
        else:
            st.warning("Masukkan minimal 1 pengaduan (1 baris = 1 pengaduan)!")

    # --- SELALU tampilkan data editor & download below form ---
    if not st.session_state["manual_df"].empty:
        st.success("Hasil Prediksi Manual (bisa diedit label):")
        editable_cols = ['Sub Kategori', 'Kategori', 'Sub Askes']
        st.session_state["manual_df"] = st.data_editor(
            st.session_state["manual_df"],
            use_container_width=True,
            num_rows="fixed",
            column_config={col: st.column_config.TextColumn(disabled=False) for col in editable_cols},
            disabled=[col for col in st.session_state["manual_df"].columns if col not in editable_cols]
        )
        st.download_button(
            "⬇️ Download Hasil (CSV)",
            st.session_state["manual_df"].to_csv(index=False),
            file_name="hasil_prediksi_manual.csv",
            mime="text/csv"
        )
        if st.button("Reset Data Manual"):
            st.session_state["manual_df"] = pd.DataFrame()

# =================== MODE UPLOAD FILE (tidak berubah banyak) ===================
elif mode == "Upload File":
    st.subheader("Upload File Pengaduan (CSV/XLSX)")
    uploaded_file = st.file_uploader("Upload file pengaduan", type=["csv", "xlsx"])
    if "edited_df" not in st.session_state:
        st.session_state["edited_df"] = None

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("Preview Data Asli:", df.head(3))

        pengaduan_cols = [
            col for col in df.columns
            if "aduan" in col.lower() or "complain" in col.lower() or "pengaduan" in col.lower()
        ]
        if not pengaduan_cols:
            st.warning("Kolom pengaduan tidak otomatis terdeteksi. Silakan pilih kolom pengaduan dari data Anda.")
            col_pengaduan = st.selectbox("Pilih kolom pengaduan", df.columns.tolist())
        else:
            col_pengaduan = pengaduan_cols[0]
            st.success(f"Otomatis menggunakan kolom: **{col_pengaduan}**")

        metadata_fields = ["Tanggal Tiket", "Pukul Chat  (WIB)", "SLA", "Status", "Tanggal Closed"]
        cols_found = [col for col in metadata_fields if col in df.columns]
        missing_cols = [col for col in metadata_fields if col not in df.columns]
        st.info(f"Kolom meta-data yang ditemukan di file: {', '.join(cols_found) if cols_found else 'Tidak ada'}")
        if missing_cols:
            st.warning(
                f"Beberapa kolom meta-data belum ada di file: {', '.join(missing_cols)}. "
                f"Nanti akan ditambah otomatis dengan nilai default/manual."
            )

        col_pred, col_progress = st.columns([1, 0.2])
        with col_pred:
            pred_button = st.button("Prediksi Otomatis File")
        with col_progress:
            progress_placeholder = st.empty()

        if pred_button:
            texts = df[col_pengaduan].fillna("").astype(str).tolist()
            for i in range(10):
                time.sleep(0.04)
                progress_placeholder.progress((i+1)*10)
            preds = predict_batch(texts)
            for col in ['Sub Kategori', 'Kategori', 'Sub Askes']:
                df[col] = preds[col].values
            df_out = df.rename(columns={col_pengaduan: "Pengaduan"})
            now = datetime.now()
            if "Tanggal Tiket" not in df_out.columns:
                df_out["Tanggal Tiket"] = now.date()
            if "Pukul Chat  (WIB)" not in df_out.columns:
                df_out["Pukul Chat  (WIB)"] = now.strftime("%H:%M")
            if "SLA" not in df_out.columns:
                df_out["SLA"] = "5 hari"
            if "Status" not in df_out.columns:
                df_out["Status"] = "Open"
            if "Tanggal Closed" not in df_out.columns:
                df_out["Tanggal Closed"] = ""
            st.session_state["edited_df"] = df_out
            progress_placeholder.empty()

    if st.session_state["edited_df"] is not None:
        st.success("Hasil Labeling Otomatis (bisa diedit manual):")
        editable_cols = ['Sub Kategori', 'Kategori', 'Sub Askes']
        st.session_state["edited_df"] = st.data_editor(
            st.session_state["edited_df"],
            use_container_width=True,
            num_rows="dynamic",
            column_config={col: st.column_config.TextColumn(disabled=False) for col in editable_cols},
            disabled=[col for col in st.session_state["edited_df"].columns if col not in editable_cols]
        )
        st.download_button(
            "⬇️ Download Hasil Prediksi (CSV)",
            data=st.session_state["edited_df"].to_csv(index=False),
            file_name="hasil_prediksi_batch.csv",
            mime="text/csv"
        )
        if st.button("Reset Data Upload"):
            st.session_state["edited_df"] = None

elif mode == "Playground Analytic":
    st.subheader("📊 Playground Analytic & Visualization")

    uploaded_file = st.file_uploader("Upload file pengaduan (CSV/XLSX)", type=["csv", "xlsx"], key="playground-upload")
    if uploaded_file:
        # Load file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Konversi kolom tanggal jika ada
        if "Tanggal Tiket" in df.columns:
            df["Tanggal Tiket"] = pd.to_datetime(df["Tanggal Tiket"], errors="coerce")
        
        st.write("Preview Data:", df.head(3))
        
        # ========== FILTERS ==========
        st.markdown("### 🔎 Filter Data")
        # Bersihkan & normalisasi kolom filter
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
        # --- Tanggal Tiket (range) ---
        if "Tanggal Tiket" in df.columns:
            tgl_min, tgl_max = df["Tanggal Tiket"].min(), df["Tanggal Tiket"].max()
            date_range = st.date_input(
                "Tanggal Tiket", value=(tgl_min, tgl_max), min_value=tgl_min, max_value=tgl_max
            )
            df = df[df["Tanggal Tiket"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))]

        # --- Kategori ---
        if "Kategori" in df.columns:
            kategori_opsi = [x for x in df["Kategori"].dropna().unique() if x]
            kategori_opsi = sorted(kategori_opsi, key=natural_sort_key)  # <-- Perbaiki sort!
            kategori_pilih = st.multiselect("Kategori", kategori_opsi, default=kategori_opsi)
            df = df[df["Kategori"].isin(kategori_pilih)]

        # --- Sub Kategori ---
        if "Sub Kategori" in df.columns:
            subkat_opsi = sorted(df["Sub Kategori"].dropna().unique())
            subkat_pilih = st.multiselect("Sub Kategori", subkat_opsi, default=subkat_opsi)
            df = df[df["Sub Kategori"].isin(subkat_pilih)]

        # --- Sub Askes ---
        if "Sub Askes" in df.columns:
            subaskes_opsi = sorted(df["Sub Askes"].dropna().unique())
            subaskes_pilih = st.multiselect("Sub Askes", subaskes_opsi, default=subaskes_opsi)
            df = df[df["Sub Askes"].isin(subaskes_pilih)]

        # --- Status ---
        if "Status" in df.columns:
            status_opsi = sorted(df["Status"].dropna().unique())
            status_pilih = st.multiselect("Status", status_opsi, default=status_opsi)
            df = df[df["Status"].isin(status_pilih)]
        
        # --- SLA ---
        if "SLA" in df.columns:
            sla_opsi = sorted(df["SLA"].dropna().unique())
            sla_pilih = st.multiselect("SLA", sla_opsi, default=sla_opsi)
            df = df[df["SLA"].isin(sla_pilih)]

        # --- Tampilkan hasil filter ---
        st.write(f"### Hasil Filter: {len(df)} data")
        st.dataframe(df, use_container_width=True)

        # ========== AUTO CHARTS ==========

        # --- Komplain per tanggal (jika ada kolom tanggal) ---
        if "Tanggal Tiket" in df.columns:
            count_per_date = df.groupby("Tanggal Tiket").size().reset_index(name="Jumlah Komplain")
            fig = px.bar(count_per_date, x="Tanggal Tiket", y="Jumlah Komplain", title="Jumlah Komplain per Tanggal Tiket")
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Komplain per kategori ---
        if "Kategori" in df.columns:
            count_per_kat = df.groupby("Kategori").size().reset_index(name="Jumlah Komplain")
            # Pastikan urutan bar mengikuti urutan angka
            count_per_kat = count_per_kat.sort_values(by="Kategori", key=lambda x: x.map(natural_sort_key))
            fig = px.bar(count_per_kat, x="Kategori", y="Jumlah Komplain", title="Jumlah Komplain per Kategori")
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Komplain per sub kategori ---
        if "Sub Kategori" in df.columns:
            count_per_subkat = df.groupby("Sub Kategori").size().reset_index(name="Jumlah Komplain")
            fig = px.bar(count_per_subkat, x="Sub Kategori", y="Jumlah Komplain", title="Jumlah Komplain per Sub Kategori")
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Komplain per status ---
        if "Status" in df.columns:
            count_per_status = df.groupby("Status").size().reset_index(name="Jumlah Komplain")
            fig = px.pie(count_per_status, names="Status", values="Jumlah Komplain", title="Proporsi Komplain per Status")
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Komplain per SLA ---
        if "SLA" in df.columns:
            count_per_sla = df.groupby("SLA").size().reset_index(name="Jumlah Komplain")
            fig = px.bar(count_per_sla, x="SLA", y="Jumlah Komplain", title="Jumlah Komplain per SLA")
            st.plotly_chart(fig, use_container_width=True)
        
        # --- Export hasil filter ---
        st.download_button(
            "⬇️ Download Hasil Filter (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="hasil_filter_playground.csv",
            mime="text/csv"
        )
    else:
        st.info("Silakan upload file pengaduan terlebih dahulu (format CSV/XLSX).")

st.markdown("---")
st.caption("Developed by Irfan Karim • Powered by AI Auto-Labeling | Streamlit UI v4")
