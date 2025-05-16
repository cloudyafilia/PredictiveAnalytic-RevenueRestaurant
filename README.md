# Laporan Proyek Machine Learning - Restaurant Revenue Prediction

## Domain Proyek
Industri restoran merupakan salah satu sektor ekonomi yang memiliki kontribusi signifikan terhadap perekonomian global dan regional. Di tengah persaingan bisnis yang semakin ketat, kemampuan untuk memprediksi pendapatan (revenue) restoran menjadi sangat penting bagi pemilik usaha dan investor dalam merancang strategi bisnis yang efektif (Fatmah et al., 2024). Faktor-faktor seperti lokasi restoran, jenis layanan (cuisine), kapasitas tempat duduk, anggaran pemasaran, kualitas layanan, hingga ulasan pelanggan menjadi variabel kunci yang secara langsung maupun tidak langsung mempengaruhi kinerja finansial sebuah restoran. Dalam konteks manajemen bisnis modern, pengambilan keputusan berbasis data (data-driven decision making) menjadi pendekatan yang semakin krusial untuk mempertahankan daya saing dan meningkatkan efisiensi operasional (Wulandari et al., 2023).

Seiring berkembangnya teknologi kecerdasan buatan dan machine learning, pemanfaatan model prediktif untuk mengestimasi revenue restoran menjadi semakin relevan (Permana et al., 2023). Model machine learning mampu mengenali pola kompleks dan hubungan non-linear antar variabel yang sulit ditangkap oleh analisis konvensional. Hal ini memungkinkan pelaku bisnis untuk mendapatkan proyeksi pendapatan yang lebih akurat, sekaligus mengidentifikasi faktor-faktor utama yang dapat dioptimalkan guna meningkatkan profitabilitas. Dengan kata lain, prediksi revenue berbasis machine learning tidak hanya membantu dalam forecasting, tetapi juga menjadi alat strategis untuk pengambilan keputusan bisnis jangka panjang di sektor restoran (Triansyah et al., 2024).

## Business Understanding

### Problem Statements
1. Bagaimana memprediksi pendapatan restoran dari fitur-fitur yang tersedia?
2. Fitur mana yang paling mempengaruhi revenue?

### Goals
1. Membangun model Machine Learning untuk prediksi revenue restoran.
2. Mengidentifikasi fitur-fitur yang berkontribusi besar terhadap revenue.

### Solution Statements
Berdasarkan masalah dan tujuan di atas, maka dapat diterapkan solusi sebagai berikut:

- Menggunakan dataset yang mencakup beberapa fitur yaitu:

    i. Spesifikasi Restoran: Lokasi (kota), jenis layanan (Cuisine), kapasitas tempat duduk, harga rata-rata makanan, dan pengalaman koki.

    ii. Faktor Bisnis & Operasional: Anggaran pemasaran, jumlah pengikut media sosial, jumlah ulasan pelanggan, skor kualitas layanan, skor suasana restoran, dan jumlah reservasi mingguan.

    iii. Variabel target prediksi: Pendapatan (Revenue) restoran.

- Pembuatan model Machine Learning untuk memprediksi pendapatan restoran dilakukan menggunakan 5 model yaitu:

    - Linear Regression  
    - Decision Tree Regressor  
    - Random Forest Regressor  
    - Gradient Boosting Regressor  
    - K-Nearest Neighbor (KNN) Regressor

## Data Understanding

- Dataset yang digunakan adalah dataset [Restaurant Revenue Prediction Dataset](https://www.kaggle.com/datasets/anthonytherrien/restaurant-revenue-prediction-dataset) yang diambil dari platform penyedia data Kaggle. File yang digunakan berekstensi `.csv`.

- Dataset ini berisi informasi tentang berbagai atribut restoran yang dapat mempengaruhi pendapatan (Revenue). Setiap baris mewakili satu restoran unik dengan fitur-fitur yang menggambarkan lokasi, jenis layanan, aspek operasional, hingga interaksi dengan pelanggan.

### Exploratory Data Analysis (EDA)

#### EDA - Deskripsi Fitur

- Dataset terdiri dari 8368 baris dan 18 kolom yang berisi informasi mengenai karakteristik restoran dan total pendapatan yang dihasilkan.

- Kolom-kolom dalam dataset dijelaskan sebagai berikut:

  - **Name**: Nama restoran.
  - **Location**: Lokasi restoran (contoh: Rural, Downtown).
  - **Cuisine**: Jenis layanan atau kategori masakan yang ditawarkan (contoh: Japanese, Mexican, Italian).
  - **Rating**: Rata-rata penilaian restoran.
  - **Seating Capacity**: Kapasitas tempat duduk restoran.
  - **Average Meal Price**: Harga rata-rata per porsi makanan.
  - **Marketing Budget**: Anggaran pemasaran yang dialokasikan untuk restoran.
  - **Social Media Followers**: Jumlah pengikut di media sosial.
  - **Chef Experience Years**: Lama pengalaman kerja koki utama (dalam tahun).
  - **Number of Reviews**: Jumlah total ulasan yang diterima oleh restoran.
  - **Avg Review Length**: Rata-rata panjang ulasan dari pelanggan.
  - **Ambience Score**: Skor yang merepresentasikan kualitas suasana restoran.
  - **Service Quality Score**: Skor yang merepresentasikan kualitas layanan.
  - **Parking Availability**: Ketersediaan lahan parkir (Yes/No).
  - **Weekend Reservations**: Jumlah reservasi yang dilakukan saat akhir pekan.
  - **Weekday Reservations**: Jumlah reservasi yang dilakukan saat hari kerja.
  - **Revenue**: Total pendapatan yang dihasilkan oleh restoran (target prediksi).

#### EDA - Univariate Analisis

##### Tabel Deskripsi Statistik

| Fitur                   | Count | Mean        | Std         | Min        | 25%        | 50%        | 75%        | Max          |
|-------------------------|--------|-------------|-------------|------------|------------|------------|------------|--------------|
| Rating                  | 8368   | 4.008       | 0.581       | 3.0        | 3.5        | 4.0        | 4.5        | 5.0          |
| Seating Capacity        | 8368   | 60.21       | 17.40       | 30.0       | 45.0       | 60.0       | 75.0       | 90.0         |
| Average Meal Price      | 8368   | 47.90       | 14.34       | 25.0       | 35.49      | 45.53      | 60.30      | 76.0         |
| Marketing Budget        | 8368   | 3218.25     | 1824.90     | 604.0      | 1889.0     | 2846.50    | 4008.50    | 9978.0       |
| Social Media Followers  | 8368   | 36190.62    | 18630.15    | 5277.0     | 22592.50   | 32518.50   | 44566.25   | 103770.0     |
| Chef Experience Years   | 8368   | 10.05       | 5.52        | 1.0        | 5.0        | 10.0       | 15.0       | 19.0         |
| Number of Reviews       | 8368   | 523.01      | 277.22      | 50.0       | 277.0      | 528.0      | 764.25     | 999.0        |
| Avg Review Length       | 8368   | 174.77      | 71.99       | 50.01      | 113.31     | 173.91     | 237.41     | 299.98       |
| Ambience Score          | 8368   | 5.52        | 2.58        | 1.0        | 3.3        | 5.5        | 7.8        | 10.0         |
| Service Quality Score   | 8368   | 5.51        | 2.59        | 1.0        | 3.2        | 5.6        | 7.8        | 10.0         |
| Weekend Reservations    | 8368   | 29.49       | 20.03       | 0.0        | 13.0       | 27.0       | 43.0       | 88.0         |
| Weekday Reservations    | 8368   | 29.24       | 20.04       | 0.0        | 13.0       | 26.0       | 43.0       | 88.0         |
| Revenue                 | 8368   | 656070.56   | 267413.74   | 184708.52  | 454651.40  | 604242.09  | 813094.23  | 1531868.0    |

##### Penjelasan Statistik Deskriptif

1. **Rating:**
   - Rata-rata rating restoran adalah **4.01**, dengan nilai tengah (median) **4.0**.
   - Rentang rating berkisar antara **3.0 hingga 5.0**, dengan standar deviasi **0.58**.
   - Mayoritas restoran memiliki rating di kisaran **3.5 hingga 4.5**.

2. **Seating Capacity:**
   - Rata-rata kapasitas tempat duduk sebesar **60 kursi**, dengan median **60**.
   - Kapasitas berkisar antara **30 hingga 90 kursi**, dengan standar deviasi **17.40**.
   - Sebagian besar restoran memiliki kapasitas antara **45 hingga 75 kursi**.

3. **Average Meal Price:**
   - Harga rata-rata per porsi makanan adalah **47.90**, dengan median **45.53**.
   - Harga berkisar dari **25 hingga 76**, dengan sebaran **14.34**.
   - Sebagian besar restoran memiliki harga rata-rata antara **35.49 hingga 60.30**.

4. **Marketing Budget:**
   - Anggaran pemasaran rata-rata sebesar **3218.25**, dengan median **2846.50**.
   - Rentang anggaran dari **604 hingga 9978**, dengan standar deviasi **1824.90**.
   - Sebagian besar restoran memiliki budget di kisaran **1889 hingga 4008.50**.

5. **Social Media Followers:**
   - Jumlah pengikut media sosial rata-rata **36190.62**, dengan median **32518.50**.
   - Rentang followers dari **5277 hingga 103770**, dengan standar deviasi **18630.15**.
   - Mayoritas restoran memiliki followers di kisaran **22592.50 hingga 44566.25**.

6. **Revenue (Target):**
   - Pendapatan rata-rata restoran adalah **656070.56**, dengan median **604242.09**.
   - Revenue berkisar dari **184708.52 hingga 1531868.0**, dengan standar deviasi **267413.74**.
   - Sebagian besar restoran memiliki pendapatan di antara **454651.40 hingga 813094.23**.

#### EDA - Tabel Deskripsi Fitur Kategorikal

##### Tabel Statistik Deskriptif (Data Kategorikal)

| Fitur                  | Count | Unique | Kategori Terbanyak (Top) | Frekuensi Terbanyak (Freq) |
|------------------------|--------|--------|--------------------------|---------------------------|
| Name                   | 8368   | 8368   | Restaurant 8351           | 1                         |
| Location               | 8368   | 3      | Downtown                  | 2821                      |
| Cuisine                | 8368   | 6      | French                    | 1433                      |
| Parking Availability   | 8368   | 2      | Yes                       | 4189                      |

##### Penjelasan Statistik Deskriptif (Data Kategorikal)

1. **Name (Nama Restoran):**
   - Terdapat **8.368 entri unik** pada fitur Name, dengan total **8.368 data**.
   - Nama restoran seperti **"Restaurant 8351"** muncul hanya sekali.
   - Hal ini menunjukkan bahwa setiap restoran memiliki nama yang berbeda (unik).

2. **Location (Lokasi Restoran):**
   - Lokasi restoran terbagi menjadi **3 kategori berbeda**.
   - Kategori lokasi yang paling sering muncul adalah **Downtown**, sebanyak **2.821 kali**.
   - Ini menunjukkan bahwa sebagian besar restoran dalam dataset berada di pusat kota.

3. **Cuisine (Jenis Layanan / Masakan):**
   - Terdapat **6 jenis layanan/cuisine** dalam dataset ini.
   - Jenis masakan yang paling banyak adalah **French**, muncul sebanyak **1.433 kali**.
   - Ini menunjukkan bahwa restoran dengan layanan masakan French cukup dominan.

4. **Parking Availability (Ketersediaan Parkir):**
   - Fitur ini memiliki **2 kategori**: Yes dan No.
   - Sebanyak **4.189 restoran** tercatat memiliki fasilitas parkir (Yes).
   - Ini menunjukkan bahwa lebih dari separuh restoran dalam data menyediakan fasilitas parkir.

#### EDA - Visualisasi

###### Visualisasi Distribusi Revenue Restaurant
![image](https://github.com/user-attachments/assets/c475a711-e5cf-40fb-a7c2-ca4ad3fa5f91)

###### Visualisasi Distribusi Cuisine Type
![image](https://github.com/user-attachments/assets/55e46994-95af-421a-9ade-1b546f79062a)

#### EDA - Bivariate Analisis

###### Visualisasi Revenue Restaurant berdasarkan Cuisine Type
![image](https://github.com/user-attachments/assets/37e127e5-3c5d-477d-9083-b4b5732c0ac8)

###### Visualisasi Average Meal Price berdasarkan Location
![image](https://github.com/user-attachments/assets/4d2e3c1b-2e07-4613-8f46-f4bc33e7c3e8)

#### EDA - Multivariate Analisis

###### Visualisasi Matriks Korelasi antar Fitur
![image](https://github.com/user-attachments/assets/714e90a0-e94d-4c66-9462-ba6bc11b5682)

###### Visualisasi Pair Plot antar Fitur
![image](https://github.com/user-attachments/assets/6021ebd9-dad1-4257-86fb-7152d35c9f23)

###### Visualisasi Hubungan Seating Capacity dan Revenue berdasarkan Cuisine Type
![image](https://github.com/user-attachments/assets/3db3233b-c2fb-402c-b948-9de99a7a5051)

## Data Preparation

### Menangani Duplikasi Data dalam DataFrame

- Dataset diperiksa untuk mengetahui apakah ada baris duplikat.
- Jika ditemukan, baris tersebut dihapus untuk menghindari bias dalam analisis dan pemodelan.
- Tujuannya adalah menjaga kualitas dan integritas data agar hasil prediksi tidak salah arah.

### Menghapus Kolom Tidak Relevan

- Kolom `Name` dihapus karena hanya berisi identitas unik restoran.
- Kolom ini tidak memiliki nilai prediktif terhadap target `Revenue`.
- Menghapus fitur seperti ini juga membantu menyederhanakan struktur data.

### Penanganan Outlier

- Outlier dideteksi menggunakan metode **Interquartile Range (IQR)**.
- Fokus utama penanganan outlier berada pada kolom **Revenue**.
- Langkah-langkahnya:
  - Hitung nilai **Q1 (25th percentile)** dan **Q3 (75th percentile)**.
  - Hitung **IQR = Q3 - Q1**.
  - Tentukan batas bawah (**Q1 - 1.5 * IQR**) dan batas atas (**Q3 + 1.5 * IQR**).
  - Data di luar batas tersebut dianggap sebagai outlier.
- Outlier pada `Revenue` dihapus agar model tidak bias oleh nilai ekstrem.
- Tujuan: menjaga performa model tetap stabil dan akurat.

### Identifikasi Fitur Kategorikal dan Numerikal

- **Fitur Kategorikal**:
  - Location
  - Cuisine
  - Parking Availability

- **Fitur Numerikal**:
  - Rating
  - Seating Capacity
  - Average Meal Price
  - Marketing Budget
  - Social Media Followers
  - Chef Experience Years
  - Number of Reviews
  - Avg Review Length
  - Ambience Score
  - Service Quality Score
  - Weekend Reservations
  - Weekday Reservations

### Pemisahan Fitur dan Target

- Dataset dipisahkan menjadi:
  - **X** → Semua fitur (independen)
  - **y** → Target prediksi, yaitu kolom `Revenue`

### Pembagian Dataset

- Dataset dibagi menjadi data latih dan data uji dengan rasio **80:20**
- Jumlah data latih: 6694
- Jumlah data uji: 1674

### Encoding Fitur Kategorikal

- Fitur kategorikal diubah ke bentuk numerik menggunakan **OneHotEncoder**
- Encoding dilakukan setelah split untuk menghindari data leakage
- Fitur yang di-encode:
  - Location
  - Cuisine
  - Parking Availability
- Gunakan `handle_unknown='ignore'` agar aman kalau ada kategori baru di data uji

### Standarisasi Fitur Numerikal

- Fitur numerikal dinormalisasi menggunakan **StandardScaler**
- Penting untuk model yang sensitif terhadap skala seperti **KNN**
- Hasilnya: semua fitur numerik punya mean = 0 dan std = 1

## Modeling

Dalam proyek prediksi revenue restoran ini, beberapa algoritma regresi diterapkan untuk memodelkan hubungan antara fitur-fitur restoran dan total pendapatan (Revenue). Linear Regression digunakan sebagai baseline model untuk mengidentifikasi hubungan linier dasar. Decision Tree Regressor diaplikasikan untuk menangkap pola non-linier melalui pemodelan berbasis aturan keputusan. Random Forest Regressor digunakan sebagai ensemble berbasis pohon yang memadukan prediksi banyak pohon untuk meningkatkan akurasi dan mengurangi overfitting. Gradient Boosting Regressor memperbaiki kesalahan prediksi secara bertahap sehingga seringkali menghasilkan performa terbaik. Terakhir, K-Nearest Neighbors (KNN) diterapkan untuk memprediksi revenue berdasarkan kemiripan fitur antar restoran. Setiap model memiliki karakteristik, kelebihan, dan kekurangannya masing-masing. Oleh karena itu, digunakan beberapa model untuk membandingkan efektivitas dan akurasi prediksi revenue. Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan prediksi revenue restoran.

### 1. Linear Regression

**Linear Regression** adalah model dasar yang memprediksi revenue restoran berdasarkan hubungan linier antara fitur-fitur seperti Seating Capacity, Average Meal Price, Marketing Budget, dan lainnya. Model ini berasumsi bahwa hubungan antara fitur dan target bersifat linier.

- **Kelebihan**:
  - Interpretasi koefisien mudah dan intuitif.
  - Implementasi sederhana dan efisien.
  - Komputasi cepat dan ringan.

- **Kekurangan**:
  - Mengasumsikan hubungan linier (tidak fleksibel untuk pola non-linear).
  - Sensitif terhadap outlier.
  - Tidak mampu menangkap interaksi kompleks antar fitur.

### 2. Decision Tree Regressor

**Decision Tree Regressor** memprediksi revenue dengan membangun serangkaian aturan keputusan berbasis nilai fitur. Model ini mampu menangkap pola non-linear dan interaksi antar fitur tanpa memerlukan transformasi data.

- **Kelebihan**:
  - Mampu menangkap hubungan non-linear dan interaksi fitur.
  - Tidak memerlukan penskalaan fitur.
  - Interpretasi mudah melalui visualisasi pohon keputusan.

- **Kekurangan**:
  - Rentan terhadap overfitting (terlalu menyesuaikan dengan data latih).
  - Tidak stabil terhadap perubahan kecil pada data latih.

### 3. Random Forest Regressor

**Random Forest Regressor** adalah model ensemble yang membangun banyak decision tree dan menggabungkan hasilnya untuk memprediksi revenue. Teknik ini membantu mengurangi overfitting dan meningkatkan generalisasi.

- **Kelebihan**:
  - Lebih akurat dibanding single decision tree.
  - Mengurangi overfitting dengan averaging hasil banyak pohon.
  - Dapat memberikan estimasi pentingnya fitur (feature importance).

- **Kekurangan**:
  - Lebih lambat secara komputasi dibanding single tree.
  - Interpretasi hasil model lebih sulit dibanding pohon tunggal.

### 4. Gradient Boosting Regressor

**Gradient Boosting Regressor** membangun model secara bertahap, di mana setiap pohon berikutnya mencoba memperbaiki kesalahan prediksi dari pohon sebelumnya. Model ini sering memberikan hasil terbaik dalam berbagai kompetisi prediksi.

- **Kelebihan**:
  - Akurasi tinggi dengan kemampuan menangkap pola kompleks.
  - Fleksibel untuk berbagai fungsi loss.
  - Dapat mengatasi outlier dengan lebih baik dibanding Random Forest.

- **Kekurangan**:
  - Rentan overfitting jika parameter tidak diatur dengan benar.
  - Waktu pelatihan lebih lama dibanding Random Forest.
  - Interpretasi model lebih sulit.

### 5. K-Nearest Neighbors (KNN)

**K-Nearest Neighbors (KNN)** memprediksi revenue berdasarkan kedekatan fitur dengan data restoran lain. Model ini menghitung jarak antar data untuk menentukan prediksi.

- **Kelebihan**:
  - Sederhana dan mudah dipahami.
  - Tidak mengasumsikan bentuk hubungan linier atau non-linier.
  - Dapat bekerja baik untuk dataset kecil dengan pola lokal.

- **Kekurangan**:
  - Sensitif terhadap skala fitur (memerlukan standarisasi).
  - Pemilihan parameter `k` sangat mempengaruhi hasil.
  - Kurang efisien pada dataset besar karena menghitung jarak ke semua data latih.

## Evaluation

Bagian ini membahas evaluasi performa model prediksi revenue restoran yang telah dikembangkan. Evaluasi dilakukan dengan menggunakan metrik standar regresi seperti **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, dan **R-squared (R²)**. 

- **MSE** mengukur rata-rata kuadrat selisih antara nilai revenue prediksi dan nilai revenue aktual.
- **RMSE** memberikan interpretasi kesalahan dalam satuan yang sama dengan target (revenue).
- **MAE** mengukur rata-rata kesalahan absolut antara prediksi dan nilai aktual.
- **R-squared (R²)** mengukur seberapa baik model menjelaskan variasi data revenue secara keseluruhan.

Melalui analisis metrik ini, diperoleh pemahaman tentang seberapa akurat dan stabil performa model prediksi revenue yang telah dibangun.

#### 1. Mean Squared Error (MSE)

- **Mean Squared Error (MSE)** digunakan sebagai ukuran absolut rata-rata besarnya kesalahan prediksi.
- MSE dihitung dengan cara mengkuadratkan selisih antara nilai revenue prediksi dengan nilai revenue aktual, kemudian dirata-ratakan.
- Nilai MSE yang lebih rendah menunjukkan bahwa model memiliki kesalahan prediksi yang kecil.

**Rumus MSE**:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

- MSE memberikan bobot lebih besar pada kesalahan prediksi yang besar (kuadrat dari error).
- Oleh karena itu, MSE sensitif terhadap outlier.
- Semakin kecil nilai MSE, semakin baik performa model.

#### 2. Root Mean Squared Error (RMSE)

- **RMSE** adalah akar dari nilai MSE.
- RMSE memiliki satuan yang sama dengan target (revenue), sehingga lebih mudah diinterpretasikan.
- Nilai RMSE memberikan estimasi rata-rata seberapa besar deviasi prediksi model terhadap nilai revenue aktual.

**Rumus RMSE**:

$$
RMSE = \sqrt{MSE}
$$

- RMSE sering digunakan bersamaan dengan MSE karena interpretasinya yang lebih intuitif.

#### 3. Mean Absolute Error (MAE)

- **Mean Absolute Error (MAE)** mengukur rata-rata kesalahan absolut antara prediksi dan nilai aktual.
- MAE lebih robust terhadap outlier dibanding MSE.
- Nilai MAE yang rendah menunjukkan bahwa prediksi model mendekati nilai aktual.

**Rumus MAE**:

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

- MAE dihitung tanpa mengkuadratkan selisih error, sehingga memberikan bobot kesalahan yang proporsional.

#### 4. R-squared (R²)

- **R-squared (Koefisien Determinasi)** menunjukkan seberapa baik model mampu menjelaskan variasi pada data revenue.
- Nilai R² berkisar antara 0 dan 1.
  - R² = 1 artinya model menjelaskan 100% variasi data.
  - R² = 0 artinya model tidak menjelaskan variasi data sama sekali.

**Rumus R²**:

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

di mana:
- $SS_{res} = \sum (y_i - \hat{y}_i)^2$ → jumlah kuadrat kesalahan prediksi.
- $SS_{tot} = \sum (y_i - \bar{y})^2$ → total variasi data terhadap rata-rata.

- Nilai R² yang mendekati 1 menunjukkan model yang baik dalam menjelaskan variasi revenue.
- Sebaliknya, nilai R² mendekati 0 menunjukkan model kurang mampu menangkap pola data.

### Evaluasi Model

| Model                     | MSE           | RMSE        | MAE         | R²       | Rank_MSE | Rank_RMSE | Rank_MAE | Rank_R² | Total Score |
|---------------------------|---------------|-------------|-------------|----------|----------|-----------|----------|---------|-------------|
| Linear Regression          | 2.672e+09     | 51695.05    | 39515.84    | 0.956909 | 4        | 4         | 4        | 4       | 16          |
| Decision Tree              | 2.407e+08     | 15513.61    | 11858.45    | 0.996119 | 3        | 3         | 3        | 3       | 12          |
| **Random Forest**          | **6.417e+07** | **8010.74** | **6175.19** | **0.998965** | **1**    | **1**     | **1**    | **1**   | **4**       |
| Gradient Boosting          | 9.407e+07     | 9699.03     | 7470.66     | 0.998483 | 2        | 2         | 2        | 2       | 8           |
| K-Nearest Neighbors (K=5)  | 5.945e+09     | 77101.83    | 61829.45    | 0.904145 | 5        | 5         | 5        | 5       | 20          |

Model terbaik berdasarkan gabungan semua metrik: **Random Forest** (Total Score = 4)

## Feature Importance

Bagian ini membahas tentang fitur-fitur apa saja yang paling berpengaruh terhadap prediksi revenue restoran, berdasarkan hasil feature importance dari beberapa model regresi yang telah diterapkan.

![image](https://github.com/user-attachments/assets/c64879e8-ff51-45c3-bf29-3a29faebf14f)

![image](https://github.com/user-attachments/assets/b235e4fb-b791-4fb8-991b-c1ace958a6ed)

![image](https://github.com/user-attachments/assets/0faa1b95-c085-43c6-a905-e29518e1cd85)

![image](https://github.com/user-attachments/assets/5faee70e-fb88-41a0-8f3d-433623f6230c)

Berdasarkan hasil analisis feature importance dari empat model regresi (Linear Regression, Decision Tree, Random Forest, dan Gradient Boosting), ditemukan bahwa fitur-fitur berikut secara konsisten memiliki pengaruh paling signifikan terhadap prediksi revenue restoran:

- **Marketing Budget**  
  Faktor utama yang mempengaruhi pendapatan restoran. Semakin besar anggaran pemasaran, semakin besar peluang meningkatkan revenue melalui promosi dan jangkauan pelanggan.

- **Average Meal Price**  
  Harga rata-rata makanan menunjukkan kontribusi signifikan terhadap total revenue. Harga yang optimal akan mempengaruhi margin keuntungan dan jumlah transaksi.

- **Seating Capacity**  
  Kapasitas tempat duduk menentukan potensi maksimum pelanggan yang dapat dilayani, sehingga berperan besar dalam skala pendapatan.

Selain ketiga fitur utama tersebut, beberapa fitur lain juga memberikan kontribusi meskipun tidak sebesar faktor di atas, seperti:

- **Social Media Followers**  
  Meningkatkan visibilitas dan brand awareness restoran, yang secara tidak langsung mendorong peningkatan revenue.

- **Service Quality Score** dan **Ambience Score**  
  Meskipun kontribusinya lebih kecil, kualitas layanan dan suasana restoran tetap menjadi faktor yang mempengaruhi pengalaman pelanggan dan loyalitas.


## Conclusion

Berdasarkan hasil eksperimen dan evaluasi terhadap berbagai model regresi yang diterapkan, proyek ini berhasil menjawab rumusan masalah yang telah ditetapkan, yaitu memprediksi revenue restoran berdasarkan fitur-fitur karakteristik restoran serta memahami hubungan antar fitur terhadap pendapatan.

1. Proyek ini telah mengembangkan beberapa model prediksi revenue dengan pendekatan Machine Learning, yaitu Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, dan K-Nearest Neighbors (KNN). Evaluasi performa model menggunakan metrik MSE, RMSE, MAE, dan R² menunjukkan bahwa **Random Forest Regressor** menjadi model terbaik dengan akurasi tertinggi dan kesalahan prediksi terendah. Model ini dipilih karena mampu menggabungkan prediksi dari banyak decision tree secara ensemble, sehingga menghasilkan model yang kuat, akurat, dan tahan terhadap overfitting.

2. Hasil analisis feature importance dari berbagai model mengungkapkan bahwa faktor-faktor berikut memiliki pengaruh terbesar terhadap prediksi revenue restoran:
- **Marketing Budget**: Faktor utama yang berkontribusi signifikan terhadap pendapatan melalui aktivitas promosi dan pemasaran.
- **Average Meal Price**: Harga rata-rata menu yang mempengaruhi margin keuntungan dan daya beli pelanggan.
- **Seating Capacity**: Menentukan potensi jumlah pelanggan yang dapat dilayani secara fisik.
Fitur tambahan seperti **Social Media Followers**, **Service Quality Score**, dan **Ambience Score** juga menunjukkan pengaruh meskipun tidak sebesar tiga faktor utama di atas.

### Insight
Dengan memanfaatkan model Machine Learning, khususnya Random Forest, prediksi revenue restoran dapat dilakukan secara akurat berdasarkan data karakteristik restoran. Selain itu, pemahaman mengenai fitur-fitur yang paling berpengaruh dapat menjadi acuan strategis bagi pelaku bisnis restoran dalam mengoptimalkan pendapatan melalui pengelolaan anggaran pemasaran, penetapan harga menu, dan manajemen kapasitas layanan. Model yang telah dibangun menunjukkan bahwa penerapan kecerdasan buatan dalam analisis bisnis restoran memberikan manfaat nyata dalam mendukung pengambilan keputusan berbasis data.


## Referensi
- Dataset: [Restaurant Revenue Prediction - Kaggle](https://www.kaggle.com/c/restaurant-revenue-prediction)
- Fatmah, F., Supriyanto, E., Budiman, D., Maichal, M., Ghozali, Z., Ismail, H., ... & Musty, B. (2024). UMKM & kewirausahaan: Panduan praktis. PT. Sonpedia Publishing Indonesia.
- Permana, A. A., Darmawan, R., Saputri, F. R., Harto, B., Al-Hakim, R. R., Wijayanti, R. R., ... & Rukmana, A. Y. (2023). Artificial Intelligence Marketing. Padang: Global Eksekutif Teknologi.
- Triansyah, F. A., Hasmirati, S. A., Soleh, S., MSI, M., Asep Deni, M. M., Khasanah, S. P., ... & Triantoro, I. T. (2024). Manajemen Strategi Menghadapi Industri 5.0. Cendikia Mulia Mandiri.
- Wulandari, A. R., Arvi, A. A., Iqbal, M. I., Tyas, F., Kurniawan, I., & Anshori, M. I. (2023). Digital Hr: Digital transformation in increasing productivity in the work environment. Jurnal Publikasi Ilmu Manajemen, 2(4), 29-42.
