import streamlit as st
import pandas as pd
import joblib

# —————————————————————————————
# 1. Load pipeline ter‐fitted & data
# —————————————————————————————
pipeline = joblib.load("pipeline_best.pkl")  # Preprocessing + Model

df = pd.read_csv("data.csv", delimiter=";")  # Data asli untuk opsi

# —————————————————————————————
# 2. UI Streamlit: Desain yang ditingkatkan
# —————————————————————————————
st.set_page_config(
    page_title="Deteksi Dropout Siswa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header dengan deskripsi
st.markdown(
    """
    # 📘 Sistem Deteksi Risiko Dropout
    Aplikasi ini memprediksi risiko **dropout** siswa Jaya Jaya Institut.
    Masukkan data siswa di sidebar untuk melihat hasil prediksi.
    """
)

# Sidebar dengan informasi dan input
with st.sidebar:
    st.header("Masukkan Data Siswa")
    # Mapping untuk Gender
    gender_map = {0: "Perempuan", 1: "Laki-laki"}
    gender_label = st.selectbox("Jenis Kelamin", list(gender_map.values()))
    gender_code = [k for k, v in gender_map.items() if v == gender_label][0]

    age = st.slider(
        "Usia Saat Pendaftaran",
        min_value=int(df["Age_at_enrollment"].min()),
        max_value=int(df["Age_at_enrollment"].max()),
        value=int(df["Age_at_enrollment"].median())
    )

    # Dynamic Course mapping
    course_codes = df["Course"].unique().tolist()
    course_codes.sort()
    course_map = {c: f"Course {c}" for c in course_codes}
    course_label = st.selectbox("Course", list(course_map.values()))
    course_code = [k for k, v in course_map.items() if v == course_label][0]

    # Expandable untuk input lain
    with st.expander("Fitur Lanjutan (Opsional)"):
        # Mapping Previous Qualification agar mudah dimengerti
        prev_codes = df["Previous_qualification"].unique().tolist()
        prev_codes.sort()
        prev_map = {code: f"Kode {code}" for code in prev_codes}
        prev_label = st.selectbox("Kualifikasi Sebelumnya", list(prev_map.values()))
        prev_code  = [k for k,v in prev_map.items() if v==prev_label][0]
        
        admission_grade = st.slider(
            "Admission Grade",
            float(df["Admission_grade"].min()), 
            float(df["Admission_grade"].max()), 
            float(df["Admission_grade"].median())
        )

    st.markdown("---")
    pred_button = st.button("Prediksi Dropout")

# —————————————————————————————
# 3. Logic Prediksi & Hasil
# —————————————————————————————
if pred_button:
    # Kumpulkan input
    input_dict = {
        "Gender": [gender_code],
        "Age_at_enrollment": [age],
        "Course": [course_code],
        # Tambah jika menggunakan prev_qual & admission_grade
        "Previous_qualification": [prev_code],
        "Admission_grade": [admission_grade]
    }
    input_df = pd.DataFrame(input_dict)

    # Lengkapi kolom lain dengan default median/modus
    all_cols = pipeline.named_steps['pre'].feature_names_in_
    num_cols = pipeline.named_steps['pre'].transformers_[0][2]
    cat_cols = pipeline.named_steps['pre'].transformers_[1][2]
    for col in all_cols:
        if col not in input_df.columns:
            input_df[col] = df[col].median() if col in num_cols else df[col].mode()[0]
    input_df = input_df[all_cols]

    # Prediksi
    y_pred = pipeline.predict(input_df)[0]
    y_proba = pipeline.predict_proba(input_df)[0, 1]

    # Tampilkan hasil dalam dua kolom
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Probabilitas Dropout", f"{y_proba*100:.1f}%")
    with col2:
        if y_pred == 1:
            st.error("⚠️ Siswa berisiko dropout!")
        else:
            st.success("✅ Siswa tidak berisiko tinggi.")

    # Deskripsi tambahan
    risk_desc = (
        "Intervensi segera diperlukan." if y_proba > 0.7
        else "Monitoring lanjutan direkomendasikan." if y_proba > 0.4
        else "Risiko rendah."
    )
    st.info(risk_desc)

# —————————————————————————————
# 4. Footer
# —————————————————————————————
st.markdown("---")
st.markdown(
    "Dicoding | Project by Julio Aldrin Purba"
)
