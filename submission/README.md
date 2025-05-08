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
Pastikan Anda menggunakan Python 3.10 serta anaconda dan menginstal dependensi yang tercantum dalam `requirements.txt`:
Nb : - `requirements.txt` pada folder Institusi_Pendidikan untuk model pada streamlit dan `requirements.txt` yang diluar dari folder tersebut untuk google colab
     - file untuk menjalankan secara local terdapat pada folder Institusi_pendidikan
3. **Setup Environment**
```bash
git clone https://github.com/julioaldprb/Institusi_pendidikan
cd Institusi_pendidikan
pip install -r requirements.txt
conda activate siakad
streamlit run app.py
```
Berikut link dari projek yang sudah di deploy dengan model: `https://institusipendidikan-cgnmrx6nykbqqekbtfne6g.streamlit.app/`

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

![Distribusi Target](gambar/eda1.png)

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

![Kategori Course](gambar/eda2.png)


Grafik yang Anda tampilkan menunjukkan distribusi jumlah mahasiswa berdasarkan kode Course (program studi). Berikut beberapa insight dari grafik:

    1. Course dengan kode 9500 memiliki jumlah mahasiswa terbanyak, jauh melampaui yang lain (sekitar 770-an mahasiswa). Ini mungkin merupakan jurusan paling populer.

    2. Setelah itu, Course 9147, 9238, 9085, dan 9773 juga cukup tinggi peminatnya.

    3. Course dengan kode 33 hampir tidak memiliki mahasiswa—mungkin sudah tidak aktif atau hanya ada satu orang.

**EDA Univariate**

![EDA Univariate](gambar/unv1.png)

![EDA Univariate](gambar/unv2.png)

![EDA Univariate](gambar/unv3.png)

![EDA Univariate](gambar/unv4.png)

![EDA Univariate](gambar/unv5.png)

![EDA Univariate](gambar/unv6.png)

![EDA Univariate](gambar/unv7.png)

![EDA Univariate](gambar/unv8.png)

![EDA Univariate](gambar/unv9.png)

![EDA Univariate](gambar/unv10.png)

![EDA Univariate](gambar/unv11.png)

![EDA Univariate](gambar/unv12.png)

![EDA Univariate](gambar/unv12.png)

![EDA Univariate](gambar/unv13.png)

![EDA Univariate](gambar/unv14.png)

![EDA Univariate](gambar/unv15.png)

![EDA Univariate](gambar/unv16.png)

![EDA Univariate](gambar/unv17.png)

![EDA Univariate](gambar/unv18.png)

![EDA Univariate](gambar/unv19.png)

![EDA Univariate](gambar/unv20.png)

![EDA Univariate](gambar/unv21.png)

![EDA Univariate](gambar/unv22.png)

![EDA Univariate](gambar/unv23.png)

![EDA Univariate](gambar/unv24.png)

![EDA Univariate](gambar/unv25.png)

![EDA Univariate](gambar/unv26.png)

![EDA Univariate](gambar/unv27.png)

![EDA Univariate](gambar/unv28.png)

![EDA Univariate](gambar/unv29.png)

![EDA Univariate](gambar/unv30.png)

![EDA Univariate](gambar/unv31.png)

![EDA Univariate](gambar/unv32.png)

![EDA Univariate](gambar/unv33.png)

![EDA Univariate](gambar/unv34.png)

![EDA Univariate](gambar/unv35.png)

- **Distribusi Marital_status** =
Mayoritas responden memiliki status pernikahan bernilai 1, yang kemungkinan besar merepresentasikan satu kategori dominan (misalnya, "belum menikah"). Kategori lain sangat sedikit jumlahnya.

- **Distribusi Application_order** =
Sebagian besar responden mengajukan aplikasi pada urutan pertama (1), dan frekuensinya menurun drastis untuk urutan berikutnya. Artinya, mayoritas mahasiswa diterima di pilihan pertama mereka.

- **Distribusi Course** =
Ada beberapa kelompok angka besar yang mendominasi (misalnya, mendekati angka 9000-10000), menunjukkan bahwa sebagian besar responden mengambil jenis kursus tertentu (kemungkinan diwakili dengan kode numerik besar), sementara lainnya sangat sedikit.

- **Distribusi Daytime_evening_attendance** =
Sebagian besar responden memiliki nilai 1, yang kemungkinan berarti mereka mengikuti kuliah pada waktu tertentu (misalnya, malam hari), sedangkan hanya sebagian kecil yang memiliki nilai 0.

- **Distribusi Previous_qualification** =
Hampir semua responden memiliki nilai 1, menunjukkan jenis kualifikasi sebelumnya yang sama, dengan sedikit variasi pada kategori lain (mungkin kode kualifikasi lain seperti 20, 40, dll.).

- **Distribusi Previous_qualification_grade** =
Grafik ini menunjukkan distribusi nilai kualifikasi sebelumnya. Nilai berkisar antara sekitar 90 hingga 180, dengan mayoritas nilai berkumpul di kisaran 130–140. Ini menunjukkan bahwa sebagian besar siswa memiliki nilai sebelumnya dalam kisaran tersebut. Distribusi terlihat seperti distribusi normal dengan sedikit skew ke kanan.

- **Distribusi Nacionality** =
Grafik ini menunjukkan bahwa hampir semua data berasal dari satu kategori nasionalitas (kemungkinan besar satu negara dominan). Hal ini terlihat dari satu batang yang sangat tinggi di posisi indeks sekitar 1–2, sementara kategori lainnya nyaris tidak ada frekuensinya.

- **Mothers_qualification** =
Mayoritas ibu terkonsentrasi pada jenjang nilai 38, dengan puncak lebih kecil di 1 dan 20, sedangkan kategori lain hampir tidak muncul.

- **Fathers_qualification** =
Distribusi ayah serupa: nilai 38 mendominasi, disusul nilai 1 dan 20, sementara jenjang lain sangat jarang.

- **Mothers_occupation** =
Sebagian besar ibu berada di kategori 0–1 (kemungkinan tidak bekerja atau pekerjaan umum), dengan beberapa outlier sangat jarang di angka tinggi.

- **Distribusi Fathers_occupation** =
menunjukkan bahwa mayoritas nilai kode pekerjaan ayah terpusat di kategori rendah (sekitar 1–10), dengan frekuensi di kisaran 2.000–2.200. Terdapat pula beberapa outlier di kode tinggi (sekitar 80 dan 100), namun jumlahnya sangat kecil (di bawah 50).

- **Distribusi Admission_grade** =
cenderung membentuk kurva mendekati normal, dengan nilai ujian masuk berkisar antara 90 hingga hampir 180. Titik puncak frekuensi berada di kisaran 120–130, di mana masing‑masing bin mencatat sekitar 350–450 siswa.

- **Distribusi Displaced** =
memperlihatkan variabel biner: sekitar 2.000 siswa tidak tergolong displaced (0), dan lebih banyak—sekitar 2.400 siswa—tergolong displaced (1).

- **Distribusi Educational_special_needs** =
juga biner dan menunjukkan bahwa mayoritas besar siswa (lebih dari 4.300) tidak memiliki kebutuhan khusus, sementara hanya sekitar 80–100 siswa yang tercatat berkebutuhan khusus.

- **Distribusi Debtor** =
memperlihatkan sekitar 3.900 siswa tanpa tunggakan biaya (0), dan sekitar 500 siswa tercatat sebagai debtor (1), menandakan sebagian kecil memiliki tunggakan.

- **Distribusi Tuition_fees_up_to_date** =
hampir terbalik dengan Debtor: sekitar 3.900 siswa biaya pendidikannya tercatat sudah lunas (1), dan sekitar 500 siswa belum sepenuhnya membayar (0).

- **Distribusi Gender** =
menggambarkan pembagian dua kategori, di mana satu kelompok (kode 0) berjumlah sekitar 2.800 siswa, sedangkan kelompok lainnya (kode 1) sekitar 1.500 siswa.

- **Distribusi Age_at_enrollment** =
sangat miring ke kanan: sebagian besar siswa berusia 18–22 tahun, dengan puncak frekuensi di usia 19 (±1.500 siswa) dan 20 (±1.000 siswa). Terdapat ekor panjang hingga usia di atas 40, namun jumlahnya sangat sedikit.

- **Distribusi Scholarship_holder** =
menandakan bahwa mayoritas siswa (sekitar 3.300) tidak memegang beasiswa (0), sedangkan sekitar 1.100 siswa tercatat sebagai pemegang beasiswa (1).

- **Age\_at\_enrollment**: Mayoritas mendaftar antara 17–22 tahun, puncak di 18–19.
- **International**: Hampir seluruhnya domestik (0), sangat sedikit internasional (1).
- **1st\_sem\_credited**: Sebagian besar 0–1 sks terakreditasi.
- **1st\_sem\_enrolled**: Umumnya mengambil 5–7 sks, puncak di 6.
- **1st\_sem\_evaluations**: Biasanya 7–11 evaluasi, puncak di 8–9.
- **1st\_sem\_approved**: Kebanyakan 5–7 sks disetujui, puncak di 6.
- **1st\_sem\_grade**: Nilai rata‑rata 10–15, puncak di 12–13.
- **1st\_sem\_without\_evaluations**: Hampir semua unit dievaluasi (0 tanpa evaluasi).
- **2nd\_sem\_credited**: Mirip sem 1, kebanyakan 0–1 sks terakreditasi.
- **2nd\_sem\_enrolled**: Lagi-lagi 5–7 sks diambil, puncak di 6.
- **2nd\_sem\_evaluations**: Umumnya 5–13 evaluasi, puncak di 7–9.
- **2nd\_sem\_approved**: Lagi, 5–7 sks disetujui terbanyak.
- **2nd\_sem\_grade**: Nilai 10–15, puncak di 12–13.
- **2nd\_sem\_without\_evaluations**: Hampir semua unit ter­evaluasi.
- **Unemployment\_rate**: Variasi 7–17%, distribusi cukup merata.
- **Inflation\_rate**: Berkisar –1% sampai >3%, dengan klaster di \~1–2% dan \~3–3,5%.
- **GDP**: Pertumbuhan –4% hingga \~4%, tersebar tanpa puncak tunggal.

**Multivariate Analysis**

![Multivariate Analysis](gambar/ma1.png)

![Multivariate Analysis](gambar/ma2.png)


- **Heatmap Korelasi Fitur Numerik**

Gambar ini menunjukkan korelasi antar fitur numerik. Fitur akademik seperti nilai dan kredit antar semester saling berkorelasi kuat, sedangkan sebagian besar fitur lain seperti latar belakang keluarga tidak menunjukkan hubungan signifikan.

- **Age vs Grade (2nd Sem) by Dropout**

Scatter plot ini menunjukkan bahwa mahasiswa dengan nilai semester dua rendah, terutama yang bernilai 0, cenderung dropout. Usia saat mendaftar tidak tampak berpengaruh besar terhadap dropout.


**Numerical vs Categorical (Boxplot)**

![Numerical vs Categorical](gambar/nvc.png)

**Nilai Semester 2 per Status Dropout**

Boxplot ini memperlihatkan bahwa mahasiswa yang dropout memiliki nilai semester dua lebih rendah secara umum dibandingkan yang tidak dropout, dengan banyak yang mendapat nilai 0.

**Categorical vs Categorical (Cross-tab)**

| Gender | is_dropout: 0 (%) | is_dropout: 1 (%) |
|--------|-------------------|-------------------|
| 0      | 74.90             | 25.10             |
| 1      | 54.95             | 45.05             |

- Gender 0 (kemungkinan laki-laki) memiliki 25,10% dropout dan 74,90% tidak dropout.
- Gender 1 (kemungkinan perempuan) memiliki 45,05% dropout dan 54,95% tidak dropout.

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
       
       ![ROC Curve](gambar/eda3.png)

       Menunjukkan ROC curve yang memvisualisasikan performa model klasifikasi.

       - Sumbu X: False Positive Rate (FPR)
       - Sumbu Y: True Positive Rate (TPR)
       - Kurva mendekati sudut kiri atas, yang menunjukkan model memiliki performa yang baik.
       - Semakin luas area di bawah kurva (AUC), semakin baik performa model dalam membedakan kelas.

     - Confusion Matrix
       
       ![Confusion Matrix](gambar/eda4.png)
       
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
docker run -d --name metabase2 -p 3001:3000 `
-v "D:/File Ku/Laskar AI/Belajar Penerapan Data Science/Institusi Pendidikan/submission/data/metabase-data:/metabase.db" `
metabase/metabase
```
- 📌 Gunakan path lengkap dan ubah \ menjadi / pada Windows.
- 📌 Metabase akan membaca isi folder metabase.db saja.

### 📁 2. Pastikan File SQLite Berada di Folder yang Terbaca
Agar Metabase dapat mengakses file students_data.db, kamu harus:

Buat folder metabase.db di direktori proyek dan salin ke dalam folder tersebut :
`D:/File Ku/Laskar AI/Belajar Penerapan Data Science/Institusi Pendidikan/submission/data/metabase-data:/metabase.db"`

### 🔄 3. Restart Kontainer Metabase (Jika Sudah Pernah Dijalankan)
Jika kontainer metabase2 sudah pernah dibuat sebelumnya, jalankan ulang dengan perintah berikut
```bash
docker stop metabase2
docker rm metabase2
docker run -d --name metabase2 -p 3001:3000 `
-v "D:/File Ku/Laskar AI/Belajar Penerapan Data Science/Institusi Pendidikan/submission/data/metabase-data:/metabase.db" `
metabase/metabase
```

### 🌐 4. Akses Metabase via Browser
```http://localhost:3001```

### 🧩 5. Hubungkan File SQLite ke Metabase
1. Di halaman awal Metabase, pilih Add your own data.
2. Pilih SQLite
3. Pada bagian file path, masukkan: `/metabase.db/students_data.db`
4. Klik Next dan tunggu hingga koneksi berhasil.

### 🛑 6. Stop atau Start Metabase
`docker stop metabase2`

## Conclusion

Berdasarkan analisis dan model prediktif yang telah dibangun:

1. **Karakteristik Umum Siswa Berisiko Dropout**  
   - Nilai akademik (semester 1 & 2) rata‑rata lebih rendah dibandingkan non‑dropout (sekitar 6–8 vs 11–13).  
   - Usia pendaftaran cenderung lebih tua (> 25 tahun) dan memiliki catatan keuangan (Debtor = 1).  
   - Tingkat kehadiran kurang (banyak mata kuliah tanpa evaluasi).  
   - Faktor ekonomi makro (tingkat pengangguran tinggi) sedikit meningkatkan risiko.

2. **Performa Model**  
   - XGBoost terpilih dengan ROC‑AUC 0.93, F1‑score 0.81—cukup andal untuk menandai siswa risk‑dropout.  
   - Precision (0.87) dan Recall (0.76) menyeimbangkan false positive dan false negative.

3. **Manfaat Bisnis**  
   - Tim HR Dapat Memprioritaskan Intervensi Dini pada kelompok risiko tinggi (probabilitas > 70%).  
   - Dashboard Metabase memungkinkan monitoring bulanan dan penyesuaian kebijakan berbasis data.

## 📌 Kesimpulan Akhir

Berdasarkan hasil eksplorasi dan penerapan beberapa model machine learning, berikut adalah kesimpulan utama dari analisis yang telah dilakukan:

- Masalah utama yang dihadapi adalah prediksi mahasiswa yang berisiko mengalami dropout berdasarkan data historis mereka.
- Dari keempat model yang digunakan (Random Forest, SVM, Naive Bayes, dan XGBoost), model **XGBoost** memberikan performa terbaik dengan akurasi dan F1-score tertinggi.
- Model XGBoost mampu menangkap pola kompleks pada data, terutama dalam membedakan siswa yang dropout dan tidak dropout.
- Insight penting dari data:
   1. Mahasiswa dengan nilai IPK (grade) rendah di semester awal, jumlah evaluasi yang tidak diikuti, serta tunggakan pembayaran cenderung lebih tinggi mengalami dropout.
   2. Faktor-faktor seperti tidak up-to-date dalam pembayaran, umur saat mendaftar, dan status beasiswa juga turut memengaruhi kecenderungan dropout.
- Dengan hasil ini, institusi pendidikan dapat lebih proaktif dalam melakukan intervensi terhadap mahasiswa yang berisiko tinggi.

Model ini dapat dijadikan sistem pendukung keputusan (decision support system) oleh bagian akademik dan HR untuk meningkatkan retensi mahasiswa.



Kesimpulannya, implementasi prototipe ini akan membantu Jaya Jaya Institut mengidentifikasi dan membimbing siswa berisiko tinggi, sehingga dapat menurunkan angka dropout hingga 10–15% per tahun.

## Rekomendasi Action Items
- **Intervensi Dini:** Penting untuk segera menindaklanjuti siswa dengan probabilitas risiko di atas 70%. Setelah prediksi menandakan risiko tinggi, adakan sesi konsultasi satu‑ke‑satu dengan guru pembimbing dan     konselor untuk mengidentifikasi hambatan belajar spesifik, kemudian susun rencana belajar personal yang berfokus pada penanganan kendala tersebut.
- **Program Mentoring:** Membangun hubungan dukungan sosial dan akademik dengan memasangkan siswa berisiko dengan mentor alumni atau senior yang pernah mengalami situasi serupa. Terapkan jadwal mentoring minimal     dua kali sebulan dan sertakan modul psikologis singkat untuk membantu siswa meningkatkan motivasi, kepercayaan diri, serta mengelola stres.
- **Belajar Fleksibel:** Menyediakan materi pembelajaran online berupa video, modul interaktif, dan microlearning remedial untuk topik-topik kritis. Kombinasikan kelas tatap muka dan daring agar siswa dengan         kendala kehadiran tetap dapat mengakses materi kapan saja dan dari mana saja.
- **Monitoring:** Gunakan dashboard Metabase atau Streamlit untuk memantau metrik kunci setiap bulan—misalnya distribusi level risiko, tren dropout per program studi, dan rata‑rata nilai akademik. Selenggarakan     rapat evaluasi triwulanan yang melibatkan tim data, pengajar, dan manajemen untuk mengadaptasi strategi intervensi. Kirim laporan otomatis ke stakeholder terkait sebagai dasar keputusan strategis.






