Nama : Tata Fikri Prayoga
Nim :TI22250042
A.	Tujuan Pengolahan Kode
Secara keseluruhan, kode di atas bertujuan untuk melakukan Eksploratory Data Analysis (EDA) dan persiapan data awal (data preprocessing) pada sebuah dataset mahasiswa (original.csv). Tujuannya adalah untuk memahami karakteristik data, mendeteksi nilai yang hilang (missing values), memeriksa duplikasi data, serta menganalisis distribusi variabel target (status kelulusan mahasiswa: lulus/0 atau gagal/1) sebelum nantinya digunakan untuk membangun model machine learning (seperti Logistic Regression, Decision Tree, atau Random Forest). 
B.	Penjelasan Kode Untuk Setiap Sel
•	Sel 1: Mengimpor pustaka manipulasi data dasar seperti numpy dan pandas.
•	Sel 2: Mengimpor pustaka visualisasi data seperti matplotlib.pyplot dan seaborn.
•	Sel 3: Mengatur konfigurasi global (seperti RANDOM_STATE = 42) agar hasil randomness dapat direproduksi secara konsisten.
•	Sel 4: Mengimpor modul penilaian dari scikit-learn (misalnya accuracy_score, classification_report, confusion_matrix).
•	Sel 5: Mengimpor algoritma pemodelan awal seperti Logistic Regression dan Decision Tree.
•	Sel 6: Membaca dataset utama dari file original.csv menggunakan Pandas.
•	Sel 7: Menampilkan 5 baris pertama (df.head()) untuk melihat gambaran bentuk kolom.
•	Sel 8: Menampilkan 5 baris terakhir (df.tail()) untuk memeriksa konsistensi akhir data.
•	Sel 9: Memeriksa dimensi dataset menggunakan atribut .shape (jumlah baris dan kolom).
•	Sel 10: Menjalankan df.info() untuk melihat tipe data setiap kolom dan mendeteksi adanya memori atau tipe data yang tidak sesuai.
•	Sel 11: Memeriksa jumlah nilai yang hilang (missing values) di setiap kolom menggunakan df.isnull().sum().
•	Sel 12: Memeriksa baris duplikat di dalam dataset menggunakan df.duplicated().sum().
•	Sel 13: Membersihkan spasi tersembunyi pada nama-nama kolom (df.columns.str.strip()).
•	Sel 14: Menghapus baris atau kolom yang sepenuhnya kosong (dropna).
•	Sel 15: Memvalidasi keberadaan kolom target utama, yaitu kolom label (label (fail=1, pass=0)).
•	Sel 16: Membersihkan baris yang memiliki nilai target kosong.
•	Sel 17: Mengubah tipe data kolom tertentu (misalnya dari objek/string ke numerik jika diperlukan).
•	Sel 18: Memeriksa kembali ringkasan statistik deskriptif dasar menggunakan df.describe().
•	Sel 19: Menghitung statistik deskriptif untuk kolom non-numerik (kategorikal) menggunakan df.describe(include=['O']).
•	Sel 20: Menyimpan checkpoint data pertama setelah pembersihan awal.
•	Sel 21: Menghitung frekuensi kemunculan masing-masing kelas pada variabel target (Lulus vs Gagal).
•	Sel 22: Menghitung persentase distribusi kelas target.
•	Sel 23: Membuat visualisasi diagram batang (barchart) untuk menunjukkan ketidakseimbangan kelas (imbalanced class).
•	Sel 24: Memisahkan daftar kolom numerik dari DataFrame utama.
•	Sel 25: Memisahkan daftar kolom kategorikal dari DataFrame utama.
•	Sel 26: Membuat fungsi atau perulangan untuk menghasilkan histogram dari setiap fitur numerik.
•	Sel 27: Menganalisis distribusi usia mahasiswa.
•	Sel 28: Menganalisis distribusi jarak tempat tinggal ke universitas.
•	Sel 29: Menganalisis distribusi jumlah kehadiran atau presensi.
•	Sel 30: Menganalisis distribusi jam belajar mandiri.
•	Sel 31: Menganalisis distribusi jumlah akses atau login ke portal akademik.
•	Sel 32: Menyimpulkan temuan awal dari visualisasi distribusi fitur numerik.
•	Sel 33: Menghitung matriks korelasi Pearson antar variabel numerik ($r$).
•	Sel 34: Membuat visualisasi Heatmap korelasi menggunakan Seaborn.
•	Sel 35: Mengidentifikasi pasangan fitur yang memiliki korelasi tinggi (multikolinearitas).
•	Sel 36: Membuat visualisasi Boxplot untuk mendeteksi adanya nilai pencilan (outliers) pada fitur numerik.
•	Sel 37: Menerapkan metode statistik (seperti Interquartile Range / IQR) untuk menangani outlier ekstrem.
•	Sel 38: Melakukan imputasi atau pengisian nilai yang hilang (missing values) pada kolom numerik (menggunakan rata-rata atau median).
•	Sel 39: Melakukan imputasi pada kolom kategorikal (menggunakan modus).
•	Sel 40: Melakukan Encoding (seperti One-Hot Encoding atau Label Encoding) pada variabel kategorikal agar dapat dibaca oleh model matematika.
•	Sel 41: Membuat fitur baru hasil kombinasi fitur yang sudah ada (jika diperlukan untuk meningkatkan performa model).
•	Sel 42: Menentukan variabel independen ($X$) dan variabel target / dependen ($y$).
•	Sel 43: Membagi dataset menjadi data latih (training set) dan data uji (testing set) menggunakan train_test_split (misalnya rasio 80:20).
•	Sel 44: Memeriksa ukuran dimensi dari $X_{train}$, $X_{test}$, $y_{train}$, dan $y_{test}$.
•	Sel 45: Mengimpor StandardScaler atau MinMaxScaler untuk normalisasi fitur.
•	Sel 46: Melakukan fitting dan transform pada data latih, lalu transform pada data uji untuk menghindari kebocoran data (data leakage).
•	Sel 47: Menyimpan objek scaler atau data yang telah diskalakan ke dalam variabel khusus.
•	Sel 48: Melatih model pertama menggunakan algoritma Logistic Regression.
•	Sel 49: Melatih model kedua menggunakan algoritma Decision Tree Classifier.
•	Sel 50: Melatih model ketiga menggunakan algoritma Random Forest Classifier.
•	Sel 51: Melakukan prediksi pada data uji menggunakan model Logistic Regression.
•	Sel 52: Melakukan prediksi pada data uji menggunakan model Decision Tree.
•	Sel 53: Melakukan prediksi pada data uji menggunakan model Random Forest.
•	Sel 54: Menghitung metrik akurasi, presisi, recall, dan F1-score untuk model Logistic Regression.
•	Sel 55: Menghitung metrik evaluasi yang sama untuk model Decision Tree.
•	Sel 56: Menghitung metrik evaluasi untuk model Random Forest.
•	Sel 57: Membuat visualisasi Confusion Matrix untuk membandingkan performa antar model dan melihat kesalahan prediksi (False Positive / False Negative).
•	Sel 58: Menarik kesimpulan akhir mengenai model terbaik yang dipilih berdasarkan metrik evaluasi untuk diterapkan ke tahap selanjutnya.
