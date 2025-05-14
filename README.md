# Laporan Proyek Machine Learning - Restaurant Revenue Prediction

## Domain Proyek
Industri restoran merupakan salah satu sektor ekonomi yang memiliki kontribusi signifikan terhadap perekonomian global dan regional. Di tengah persaingan bisnis yang semakin ketat, kemampuan untuk memprediksi pendapatan (revenue) restoran menjadi sangat penting bagi pemilik usaha dan investor dalam merancang strategi bisnis yang efektif. Faktor-faktor seperti lokasi restoran, jenis layanan (cuisine), kapasitas tempat duduk, anggaran pemasaran, kualitas layanan, hingga ulasan pelanggan menjadi variabel kunci yang secara langsung maupun tidak langsung mempengaruhi kinerja finansial sebuah restoran. Dalam konteks manajemen bisnis modern, pengambilan keputusan berbasis data (data-driven decision making) menjadi pendekatan yang semakin krusial untuk mempertahankan daya saing dan meningkatkan efisiensi operasional (Chathoth et al., 2007).

Seiring berkembangnya teknologi kecerdasan buatan dan machine learning, pemanfaatan model prediktif untuk mengestimasi revenue restoran menjadi semakin relevan. Model machine learning mampu mengenali pola kompleks dan hubungan non-linear antar variabel yang sulit ditangkap oleh analisis konvensional. Hal ini memungkinkan pelaku bisnis untuk mendapatkan proyeksi pendapatan yang lebih akurat, sekaligus mengidentifikasi faktor-faktor utama yang dapat dioptimalkan guna meningkatkan profitabilitas. Dengan kata lain, prediksi revenue berbasis machine learning tidak hanya membantu dalam forecasting, tetapi juga menjadi alat strategis untuk pengambilan keputusan bisnis jangka panjang di sektor restoran (Pereira et al., 2020).

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

- Dari kolom-kolom tersebut, tipe data dapat dirangkum sebagai berikut:
  - **Fitur Kategorikal**:
    - Name
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
    - Revenue (target)

- Kolom **Name** diidentifikasi sebagai kolom identitas dan tidak memiliki kontribusi terhadap prediksi, sehingga dihapus pada tahap preprocessing.

- Tidak ditemukan missing values dalam dataset ini.

- Distribusi target Revenue menunjukkan adanya outlier, sehingga dilakukan penanganan menggunakan metode Interquartile Range (IQR) untuk meminimalkan pengaruh ekstrem pada hasil prediksi.

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
- Menghapus kolom `Name`.
- Outlier handling menggunakan metode IQR.
- OneHotEncoding untuk fitur kategorikal (Location, Cuisine, Parking Availability).
- Passthrough untuk fitur numerikal.
- StandardScaler diterapkan setelah encoding untuk KNN.
- Train-Test Split (80% - 20%).

## Modeling
Model yang digunakan:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. K-Nearest Neighbor (KNN)

## Evaluation
Evaluasi dilakukan menggunakan:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

Setiap model dievaluasi dengan keempat metrik tersebut, ditampilkan dalam tabel ranking performa model. Model terbaik dipilih berdasarkan total skor ranking dari semua metrik.

### Feature Importance
- Dilakukan analisis feature importance untuk Linear Regression berdasarkan nilai koefisien.
- Fitur penting: Seating Capacity, Marketing Budget, Average Meal Price, Customer Reviews, Weekly Reservations, Service Quality Score.

## Conclusion
- Gradient Boosting Regressor menjadi model terbaik untuk prediksi revenue restoran.
- Faktor utama yang mempengaruhi revenue meliputi kapasitas tempat duduk, anggaran marketing, harga rata-rata makanan, ulasan pelanggan, dan kualitas layanan.
- Model memberikan insight berbasis data untuk mendukung keputusan bisnis di industri restoran.

## Referensi
- Dataset: [Restaurant Revenue Prediction - Kaggle](https://www.kaggle.com/c/restaurant-revenue-prediction)
- Géron, A. (2022). *Hands-On Machine Learning*.
- Hastie, T., Tibshirani, R., & Friedman, J. (2017). *The Elements of Statistical Learning*.
