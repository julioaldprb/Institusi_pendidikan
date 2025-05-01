# Proyek Akhir: Klasifikasi Institusi Pendidikan Menggunakan Machine Learning

## Business Understanding
### Latar Belakang
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout. Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Tujuan Proyek:
- Melakukan analisis eksploratif terhadap data siswa untuk memahami pola-pola yang berkaitan dengan perilaku dropout.
- Mengembangkan model klasifikasi berbasis machine learning untuk memprediksi kemungkinan siswa akan dropout.
- Membangun aplikasi prediksi berbasis Streamlit agar hasil analisis dan model prediktif dapat digunakan oleh pihak manajemen secara praktis dan efisien.

### Permasalahan Bisnis
**1. Efisiensi Operasional yang Rendah**
Saat ini, proses identifikasi siswa yang berisiko tinggi mengalami dropout masih dilakukan secara manual dan bergantung pada intuisi staf pengajar. Hal ini memakan waktu, tidak efisien, serta berisiko tinggi terhadap kesalahan dalam pengambilan keputusan.

**2. Minimnya Dukungan Keputusan Berbasis Data**
Ketiadaan sistem prediktif membuat manajemen kesulitan dalam membuat kebijakan berbasis data untuk pencegahan dropout. Hal ini dapat mengakibatkan intervensi yang kurang tepat sasaran.

**3. Keterbatasan Akses Teknologi oleh Pengguna Non-Teknis**
Pihak pengelola dan pengajar membutuhkan alat bantu yang intuitif untuk mengakses hasil prediksi dan mengambil keputusan dengan cepat tanpa harus memahami teknis machine learning.

## Cakupan Proyek
### **Eksplorasi Data (EDA)**
    - Visualisasi distribusi fitur numerik (histogram, boxplot).
    - Visualisasi fitur kategorikal (bar chart, pie chart).
### **Preprocessing Data**
    - Imputasi nilai hilang, jika ada.  
    - Encoding kategorikal (One-Hot atau Label Encoding).  
    - Scaling numerik (StandardScaler).  
    - Pembagian data: 80% train, 20% test.
### **Modeling**
    - Pelatihan model XGBoost dengan hyperparameter tuning.  
    - Evaluasi metrik: akurasi, precision, recall, F1-score.
### **Deployment Aplikasi**: Penerapan model ke dalam aplikasi Streamlit yang siap digunakan oleh pengguna.
    - **Google Colab**: Notebook berisi seluruh pipeline (EDA, preprocessing, modeling).  
    - **Streamlit App**: Model berupa web yang memungkinkan prediksi real-time.
    - **Metabase** : Dashboard yang menampilkan faktor penting dalam memonitor performa siswa.

---

## Dokumentasi 
### Sumber Data
Dataset diambil dari [Students Performance](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv).

### Setup Environment (Streamlit Model)
1. **Install Dependencies**
Pastikan Anda menggunakan Python 3.10 dan menginstal dependensi yang tercantum dalam `requirements.txt`:
Nb : - `requirements.txt` pada folder Institusi_Pendidikan untuk model pada streamlit dan `requirements.txt` yang diluar dari folder tersebut untuk google colab
     - file untuk menjalankan secara local terdapat pada folder Institusi_pendidikan
3. **Setup Environment**
```bash
git clone https://github.com/julioaldprb/Institusi_pendidikan
cd Institusi_pendidikan
pip install -r requirements.txt
streamlit run app.py
```
Berikut link dari projek yang sudah di deploy : `https://institusipendidikan-cgnmrx6nykbqqekbtfne6g.streamlit.app/`

### Google Colab
1. **Instal dan Import Library**
2. **Load Data : `data.csv`**
3. **Data Understanding**
   - **Shape**: 4.424 baris × 37 kolom  
   - **Contoh 5 Baris Pertama**:

| Marital_status | Application_mode | Application_order | Course | … | GDP  | Status    |
|---------------:|-----------------:|------------------:|-------:|---|-----:|-----------|
|              1 |               17 |                 5 |    171 | … |  1.74| Dropout   |
|              1 |               15 |                 1 |   9254 | … |  0.79| Graduate  |
|              1 |                1 |                 5 |   9070 | … |  1.74| Dropout   |
|              1 |               17 |                 2 |   9773 | … | -3.12| Graduate  |
|              2 |               39 |                 1 |   8014 | … |  0.79| Graduate  |

**Ringkasan Dataset:**
- Jumlah entri (baris): 4424
- Jumlah fitur (kolom): 37

**Tipe data:**
- 29 kolom bertipe int64 (integer)
- 7 kolom bertipe float64 (desimal)
- 1 kolom bertipe object (kemungkinan berupa teks/kategori, yaitu Status)
  
**Kondisi Data:**
-Tidak ada nilai null (kosong) di seluruh kolom. Artinya, data sudah lengkap dan tidak memerlukan imputasi nilai yang hilang.
- Kolom Status bertipe object, yang kemungkinan besar merupakan label klasifikasi, misalnya Graduate, Dropout, atau mungkin lainnya.

**Distribusi Target**

![image](https://github.com/user-attachments/assets/30e2dc60-a359-4772-8149-589620556995)

Gambar menampilkan distribusi status mahasiswa dalam dataset:
- Graduate (Lulus): sekitar 2200+ mahasiswa
- Dropout (Putus studi): sekitar 1400+ mahasiswa
- Enrolled (Masih aktif): sekitar 800+ mahasiswa
  
Artinya:
- Sebagian besar mahasiswa dalam dataset berhasil lulus.
- Jumlah mahasiswa yang dropout juga cukup signifikan.
- Ada sebagian yang masih terdaftar (enrolled) saat data dikumpulkan.

**Korelasi Fitur Numerik Vs Dropout**
Top 10 fitur dengan korelasi tertinggi terhadap is_dropout:
is_dropout                                      1.000000
Age_at_enrollment                               0.254215
Debtor                                          0.229407
Gender                                          0.203983
Application_mode                                0.198458
Marital_status                                  0.093712
Curricular_units_2nd_sem_without_evaluations    0.079901
Mothers_qualification                           0.064958
Curricular_units_1st_sem_without_evaluations    0.054230
Previous_qualification                          0.049379

Dari hasil korelasi terhadap fitur is_dropout, berikut penjelasan singkat untuk 10 fitur dengan korelasi tertinggi:

    1. is_dropout (1.000)
    Ini adalah variabel target itu sendiri, jadi korelasinya pasti 1.

    2. Age_at_enrollment (0.254)
    Semakin tua usia saat mendaftar, semakin tinggi kemungkinan dropout. Korelasinya cukup kuat secara relatif.

    3. Debtor (0.229)
    Mahasiswa yang memiliki utang kepada institusi lebih berisiko dropout.

    4. Gender (0.204)
    Ada perbedaan tingkat dropout antara gender (kode gender perlu dicek apakah 1 = laki-laki atau perempuan).

    5. Application_mode (0.198)
    Cara mahasiswa mendaftar mungkin berkaitan dengan motivasi atau kesiapan mereka, sehingga memengaruhi dropout.

    6. Marital_status (0.094)
    Status pernikahan juga berkaitan, meskipun tidak terlalu kuat.

    7. Curricular_units_2nd_sem_without_evaluations (0.080)
    Semakin banyak mata kuliah yang tidak diikuti ujian di semester 2, semakin besar kemungkinan dropout.

    8. Mothers_qualification (0.065)
    Tingkat pendidikan ibu sedikit memengaruhi kemungkinan anaknya dropout.

    9. Curricular_units_1st_sem_without_evaluations (0.054)
    Sama seperti semester 2, jika banyak mata kuliah yang tidak diikuti ujian di semester 1, risikonya meningkat.

    10. Previous_qualification (0.049)
    Latar belakang pendidikan sebelumnya punya sedikit pengaruh.

**Analisis Kategori Course**

![image](https://github.com/user-attachments/assets/b9f2196f-9afd-49c2-9993-65e467c24676)


Grafik yang Anda tampilkan menunjukkan distribusi jumlah mahasiswa berdasarkan kode Course (program studi). Berikut beberapa insight dari grafik:

    1. Course dengan kode 9500 memiliki jumlah mahasiswa terbanyak, jauh melampaui yang lain (sekitar 770-an mahasiswa). Ini mungkin merupakan jurusan paling populer.

    2. Setelah itu, Course 9147, 9238, 9085, dan 9773 juga cukup tinggi peminatnya.

    3. Course dengan kode 33 hampir tidak memiliki mahasiswa—mungkin sudah tidak aktif atau hanya ada satu orang.

4. **Data Preparation**
   - **Drop Kolom Status** = Kolom Status dihapus karena target klasifikasi telah dibinerisasi menjadi kolom baru bernama is_dropout.
   - **Identifikasi Kolom**
     1. numeric_cols: kolom numerik (int/float) selain is_dropout.
     2. cat_cols: kolom kategorikal (tipe object) yang akan diproses secara terpisah.
   - **Preprocessing pipeline**
     1. numeric_transformer: menangani data numerik dengan imputasi (nilai median) dan normalisasi (StandardScaler).
     2. cat_transformer: menangani data kategorikal dengan imputasi nilai 'Unknown' lalu encoding ke bentuk numerik (OneHotEncoder).
     3. preprocessor: menggabungkan kedua pipeline di atas dengan ColumnTransformer.
   - **Siapkan Data**
     1. X: semua fitur (drop is_dropout).
     2. y: target biner dropout (is_dropout).
        
5. **Modelling**
   - **Split Data**
   - **Definisikan Model**
     Tiga model didefinisikan:
    1. Logistic Regression (logreg)
    2. Random Forest (rf)
    3. XGBoost (xgb)
    Masing-masing dibungkus dalam pipeline yang mencakup preprocessing (preprocessor) dan model (clf).
   - **Grid Search**
     Untuk masing-masing model, dilakukan tuning hyperparameter menggunakan GridSearchCV dengan skor evaluasi ROC AUC dan 3-fold cross-validation.
     Hasil terbaik dari tiap model disimpan di dictionary best_estimators.
   - **Cek Performa**
     Train ROC-AUC: 0.9999999999999999 Ini berarti model hampir sempurna dalam membedakan antara kelas dropout dan tidak dropout pada data latih. Namun, nilai ini yang sangat mendekati 1 bisa menjadi indikasi         overfitting (model terlalu "menghafal" data latih).

     Test ROC-AUC: 0.929 Artinya, model tetap memiliki kinerja sangat baik pada data uji (data yang belum pernah dilihat sebelumnya). Skor ROC-AUC sebesar 0.929 menunjukkan model dapat membedakan siswa dropout        dan tidak dropout dengan akurasi yang sangat tinggi.

  6. **Evaluation**
     - ROC Curve
       
       ![image](https://github.com/user-attachments/assets/63b73d21-4f8c-4210-a3ac-2c949b421448)

       Menunjukkan ROC curve yang memvisualisasikan performa model klasifikasi.

       - Sumbu X: False Positive Rate (FPR)
       - Sumbu Y: True Positive Rate (TPR)
       - Kurva mendekati sudut kiri atas, yang menunjukkan model memiliki performa yang baik.
       - Semakin luas area di bawah kurva (AUC), semakin baik performa model dalam membedakan kelas.

     - Confusion Matrix
       
       ![image](https://github.com/user-attachments/assets/190125fd-7240-4522-9241-c26fd72b198c)
       
       Confusion matrix dari hasil prediksi model:

       - True Negative (TN): 569 — Negatif yang diprediksi benar 
       - False Positive (FP): 32 — Positif yang diprediksi salah     
       - False Negative (FN): 73 — Negatif yang diprediksi salah  
       - True Positive (TP): 211 — Positif yang diprediksi benar

Model cukup baik karena jumlah TP dan TN jauh lebih tinggi dibandingkan FP dan FN.

     🔍 1. Akurasi
            - logreg: 88%
            - rf: 88%
            - xgb: 89% → tertinggi

     📊 2. Precision (kelas 1 / positif)
            - logreg: 0.90
            - rf: 0.87
            - xgb: 0.87
            → Logreg unggul dalam menghindari false positive.

     🎯 3. Recall (kelas 1 / positif)
            - logreg: 0.72
            - rf: 0.74
            - xgb: 0.76 → terbaik dalam menangkap semua kasus positif.
  
     ⚖️ 4. F1-score (kelas 1 / positif)
            - logreg: 0.80
            - rf: 0.80
            - xgb: 0.81 → seimbang antara precision dan recall.

     📚 Kesimpulan:
        - XGBoost (xgb) menunjukkan performa keseluruhan terbaik, terutama karena akurasi tertinggi dan F1-score yang paling baik.
        - Jika menghindari false positives sangat penting, logreg bisa dipertimbangkan karena precision-nya tertinggi.
        - Jika menangkap semua kasus positif (recall) lebih penting, xgb lebih unggul.

## Akses Dashboard Metabase
Dokumentasi ini menjelaskan cara menjalankan Metabase menggunakan Docker dan menghubungkannya ke file SQLite `students_data.db` yang berasal dari proyek analisis pendidikan kamu. 
Nb : Folder untuk dashboard metabase ada di folder data

---
### 🐳 1. Jalankan Metabase dengan Docker
Metabase dijalankan dengan perintah Docker berikut:

```bash
docker run -d --name metabase2 -p 3000:3000 -v "D:/File Ku/Laskar AI/Belajar Penerapan Data Science/Institusi Pendidikan/metabase.db:/metabase.db" metabase/metabase
```
- 📌 Gunakan path lengkap dan ubah \ menjadi / pada Windows.
- 📌 Metabase akan membaca isi folder metabase.db saja.

### 📁 2. Pastikan File SQLite Berada di Folder yang Terbaca
Agar Metabase dapat mengakses file students_data.db, kamu harus:

Buat folder metabase.db di direktori proyek dan salin ke dalam folder tersebut :
`D:\File Ku\Laskar AI\Belajar Penerapan Data Science\Institusi Pendidikan\data\metabase-data`

### 🔄 3. Restart Kontainer Metabase (Jika Sudah Pernah Dijalankan)
Jika kontainer metabase2 sudah pernah dibuat sebelumnya, jalankan ulang dengan perintah berikut
```bash
docker stop metabase2
docker rm metabase2
docker run -d --name metabase2 -p 3000:3000 -v "D:/File Ku/Laskar AI/Belajar Penerapan Data Science/Institusi Pendidikan/metabase.db:/metabase.db" metabase/metabase
```

### 🌐 4. Akses Metabase via Browser
```http://localhost:3000```

### 🧩 5. Hubungkan File SQLite ke Metabase
1. Di halaman awal Metabase, pilih Add your own data.
2. Pilih SQLite
3. Pada bagian file path, masukkan: `/metabase.db/students_data.db`
4. Klik Next dan tunggu hingga koneksi berhasil.

### 🛑 6. Stop atau Start Metabase
`docker stop metabase2`

## Rekomendasi Action Items
- **Intervensi Dini:** Penting untuk segera menindaklanjuti siswa dengan probabilitas risiko di atas 70%. Setelah prediksi menandakan risiko tinggi, adakan sesi konsultasi satu‑ke‑satu dengan guru pembimbing dan     konselor untuk mengidentifikasi hambatan belajar spesifik, kemudian susun rencana belajar personal yang berfokus pada penanganan kendala tersebut.
- **Program Mentoring:** Membangun hubungan dukungan sosial dan akademik dengan memasangkan siswa berisiko dengan mentor alumni atau senior yang pernah mengalami situasi serupa. Terapkan jadwal mentoring minimal     dua kali sebulan dan sertakan modul psikologis singkat untuk membantu siswa meningkatkan motivasi, kepercayaan diri, serta mengelola stres.
- **Belajar Fleksibel:** Menyediakan materi pembelajaran online berupa video, modul interaktif, dan microlearning remedial untuk topik-topik kritis. Kombinasikan kelas tatap muka dan daring agar siswa dengan         kendala kehadiran tetap dapat mengakses materi kapan saja dan dari mana saja.
- **Monitoring:** Gunakan dashboard Metabase atau Streamlit untuk memantau metrik kunci setiap bulan—misalnya distribusi level risiko, tren dropout per program studi, dan rata‑rata nilai akademik. Selenggarakan     rapat evaluasi triwulanan yang melibatkan tim data, pengajar, dan manajemen untuk mengadaptasi strategi intervensi. Kirim laporan otomatis ke stakeholder terkait sebagai dasar keputusan strategis.






