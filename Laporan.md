# Laporan Proyek Machine Learning - Cloudya Filia Putri

## Domain Proyek
Industri restoran merupakan salah satu sektor ekonomi yang memiliki kontribusi besar terhadap pertumbuhan ekonomi global maupun regional. Dalam konteks yang kompetitif dan dinamis seperti saat ini, pelaku bisnis di industri kuliner dituntut untuk mampu membuat keputusan strategis yang cepat dan tepat. Salah satu aspek penting yang menjadi indikator kinerja utama sebuah restoran adalah revenue (pendapatan). Namun, pendapatan restoran dipengaruhi oleh berbagai faktor yang saling berinteraksi, seperti lokasi, jenis masakan (cuisine), kapasitas tempat duduk, harga makanan, strategi pemasaran, dan interaksi pelanggan melalui media sosial. Menurut Fatmah et al. (2024), strategi bisnis berbasis data dapat membantu pelaku usaha UMKM termasuk restoran dalam meningkatkan efisiensi dan profitabilitas usaha mereka.

Sejumlah penelitian terdahulu telah menunjukkan bahwa pendekatan machine learning dapat digunakan secara efektif untuk memprediksi revenue restoran. Sanjana et al. (2021) menerapkan regresi linier dan algoritma machine learning lainnya dalam prediksi pendapatan restoran dengan hasil yang menjanjikan. Bera (2020) menggunakan pendekatan operational analytics berbasis algoritma regresi untuk memodelkan penjualan restoran, dan menemukan bahwa pemrosesan data awal yang tepat sangat penting untuk akurasi model. Selain itu, Gogolev & Ozhgeov (2019) membandingkan berbagai algoritma machine learning untuk prediksi revenue restoran di berbagai kota dan menyimpulkan bahwa model berbasis ensemble seperti Random Forest menghasilkan performa yang unggul. Studi terbaru oleh Jarlöv & Dahl (2023) juga menyoroti pentingnya data sintetik dan pemodelan time-series untuk meningkatkan prediksi revenue harian restoran berbasis data nyata.

Dengan mempertimbangkan temuan-temuan tersebut, pendekatan berbasis machine learning seperti Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, K-Nearest Neighbors (KNN), dan Support Vector Regression (SVR) semakin relevan untuk digunakan dalam proyek ini. Tidak hanya bertujuan untuk memprediksi revenue restoran dengan lebih akurat, tetapi juga untuk mengidentifikasi fitur-fitur yang paling berpengaruh terhadap kinerja finansial. Pendekatan ini selaras dengan prinsip data-driven decision making yang dikemukakan oleh Wulandari et al. (2023), yang menyebutkan bahwa transformasi digital dan pemanfaatan AI dalam manajemen bisnis menjadi kunci dalam meningkatkan daya saing. Oleh karena itu, proyek ini bertujuan untuk membangun, membandingkan, dan mengevaluasi beberapa model prediktif yang dapat memberikan insight strategis bagi pelaku usaha restoran dalam mengoptimalkan pendapatan mereka.

## Business Understanding
Pendapatan merupakan indikator utama keberhasilan operasional restoran. Namun, pendapatan sebuah restoran juga dipengaruhi oleh berbagai faktor yang kompleks dan saling berinteraksi, seperti lokasi, jenis layanan, dan strategi pemasaran, sehingga diperlukan pendekatan prediktif berbasis data untuk memahami dan mengestimasi pendapatan secara lebih akurat.

### Problem Statements
Adapun rumusan masalah dalam analisis ini adalah sebagai berikut.
1. Bagaimana memprediksi pendapatan restoran dari fitur-fitur yang tersedia?
2. Fitur mana yang paling berpengaruh dalam menentukan besarnya revenue?

### Goals
Berdasarkan rumusan masalah yang telah dipaparkan, maka diperoleh tujuan sebagai berikut.
1. Membangun model Machine Learning untuk prediksi revenue restoran.
2. Mengidentifikasi fitur-fitur yang berkontribusi besar terhadap revenue.

### Solution Statements
Berdasarkan masalah dan tujuan di atas, maka dapat diterapkan solusi sebagai berikut:

1. **Menggunakan dataset restoran yang mencakup beberapa fitur**, yaitu:
   
    a. Spesifikasi Restoran: Lokasi (kota), jenis layanan (Cuisine), kapasitas tempat duduk, harga rata-rata makanan, dan pengalaman koki.
   
    b. Faktor Bisnis & Operasional: Anggaran pemasaran, jumlah pengikut media sosial, jumlah ulasan pelanggan, skor kualitas layanan, skor suasana restoran, dan jumlah reservasi mingguan.
   
    c. Variabel target prediksi: Pendapatan (Revenue) restoran.

3. **Membangun dan membandingkan beberapa model Machine Learning untuk memprediksi pendapatan restoran**, antara lain:
    - Linear Regression  
    - Random Forest Regressor  
    - Gradient Boosting Regressor  
    - K-Nearest Neighbor (KNN) Regressor
    - Support Vector Regression (SVR)

4. **Menggunakan beberapa metrik evaluasi untuk mengukur performa model**, yaitu Mean Squared Error (MSE), Root Mean Squared Error (RMSE),  Mean Absolute Error (MAE), dan R-squared (R²).

5. **Peningkatan performa model dilakukan melalui**:
    - Pemilihan fitur terbaik (feature selection atau importance analysis)
    - Standardisasi data numerik (untuk model tertentu seperti KNN dan SVR)
    - Hyperparameter tuning pada model ensemble seperti Random Forest dan Gradient Boosting untuk mendapatkan performa optimal.

## Data Understanding

- Dataset yang digunakan adalah dataset [Restaurant Revenue Prediction Dataset](https://www.kaggle.com/datasets/anthonytherrien/restaurant-revenue-prediction-dataset) yang diambil dari platform penyedia data Kaggle. File yang digunakan berekstensi `.csv`.

- Dataset ini berisi informasi tentang berbagai atribut restoran yang dapat mempengaruhi pendapatan (Revenue). Setiap baris mewakili satu restoran unik dengan fitur-fitur yang menggambarkan lokasi, jenis layanan, aspek operasional, hingga interaksi dengan pelanggan.

### 1. Deskripsi Fitur

- Dataset terdiri dari 8368 baris dengan 17 fitur yang berisi informasi mengenai karakteristik restoran dan total pendapatan yang dihasilkan.

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

### 2. Deskripsi Statistik

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

#### Penjelasan Statistik Deskriptif

| **Fitur**                  | **Penjelasan** |
|----------------------------|----------------|
| **Rating**                 | Rata-rata rating restoran adalah 4.0 dari skala 5, dengan nilai tertinggi mencapai 5.0. |
| **Seating Capacity**       | Kapasitas tempat duduk berkisar antara 29 hingga 90 kursi, dengan rata-rata sekitar 60 kursi. |
| **Average Meal Price**     | Harga rata-rata makanan berkisar dari $25 hingga $76, dengan median sekitar $45.5. |
| **Marketing Budget**       | Rata-rata anggaran pemasaran adalah sekitar $3.218, dengan variasi tinggi dan maksimum hampir $10.000. |
| **Social Media Followers** | Jumlah pengikut media sosial sangat bervariasi, dengan rata-rata 36.169 dan maksimum lebih dari 100.000. |
| **Chef Experience Years**  | Pengalaman koki rata-rata adalah sekitar 19 tahun, dengan kisaran antara 5 hingga 30 tahun. |
| **Number of Reviews**      | Jumlah ulasan per restoran sangat bervariasi, dengan median sekitar 529 dan maksimum hingga 999 ulasan. |
| **Avg Review Length**      | Panjang rata-rata ulasan adalah 72 kata, dengan nilai maksimum 199 kata. |
| **Ambience Score**         | Nilai ambience berkisar dari 1 hingga 10, dengan rata-rata 5.5, menunjukkan sebaran merata. |
| **Service Quality Score**  | Skor kualitas pelayanan rata-rata adalah 5.5 dari skala 10, dengan banyak nilai rendah. |
| **Weekend Reservations**   | Reservasi akhir pekan rata-rata sekitar 29 orang, dengan maksimum mencapai 80. |
| **Weekday Reservations**   | Reservasi hari kerja lebih rendah, rata-rata sekitar 29 dengan maksimum 88. |
| **Revenue**                | Pendapatan restoran bervariasi besar, dengan rata-rata sekitar $650.000 dan maksimum lebih dari $1,5 juta. |

### 3. Deskripsi Fitur Kategorikal

| Fitur                  | Count | Unique | Kategori Terbanyak (Top) | Frekuensi Terbanyak (Freq) |
|------------------------|--------|--------|--------------------------|---------------------------|
| Name                   | 8368   | 8368   | Restaurant 8351           | 1                         |
| Location               | 8368   | 3      | Downtown                  | 2821                      |
| Cuisine                | 8368   | 6      | French                    | 1433                      |
| Parking Availability   | 8368   | 2      | Yes                       | 4189                      |

#### Penjelasan Statistik Deskriptif (Data Kategorikal)

| **Fitur**                | **Penjelasan** |
|--------------------------|----------------|
| **Name**                 | Terdapat 8.368 nama restoran unik, dengan masing-masing hanya muncul satu kali. |
| **Location**             | Terdapat 3 lokasi berbeda, dengan lokasi paling umum adalah *Downtown* (2.821 restoran). |
| **Cuisine**              | Terdapat 6 jenis masakan, dengan *French* sebagai yang paling umum (1.433 restoran). |
| **Parking Availability** | Mayoritas restoran memiliki parkir (*Yes*), sebanyak 4.189 dari total 8.368 restoran. |

### 4. Pemeriksaan Missing Value

 Pengecekan missing values dapat menggunakan kode sebagai berikut.
  ```python
  # Memeriksa missing value
  df.isnull().sum()
  ```

![image](https://github.com/user-attachments/assets/58ea8018-c8f4-4e41-b65e-6f60db064488)

Dataset ini telah dicek terhadap nilai kosong (missing values), dan **tidak ditemukan nilai yang hilang** di seluruh kolom. Ini menunjukkan kualitas data yang baik dari sisi kelengkapan.

### 5. Pemeriksaan Duplikasi Data

  Pengecekan data duplikat dapat menggunakan kode sebagai berikut.
  ```python
  # Memeriksa duplikasi data
  jumlah_duplikat = df.duplicated().sum()
    print(f"Jumlah baris duplikat: {jumlah_duplikat}")
  ```

![image](https://github.com/user-attachments/assets/d793da9c-513d-445f-839d-31bc6f5c30b5)

Pemeriksaan terhadap baris yang duplikat menunjukkan bahwa **tidak ada data duplikat** pada dataset. Setiap restoran memiliki ID unik yang dapat dikenali melalui kolom `Name`.

### 6. Pemeriksaan Outlier

![image](https://github.com/user-attachments/assets/4f4aa2c3-3125-488e-ad6c-47f36497031a)

Beberapa fitur numerik, terutama `Revenue`, mengandung nilai outlier berdasarkan analisis IQR (Interquartile Range). Nilai-nilai ekstrem ini akan ditangani pada tahap data preparation untuk menghindari bias model prediksi.

### Exploratory Data Analysis (EDA)
Analisis eksploratif dilakukan untuk memahami distribusi data dan hubungan antar fitur. Terdiri dari:

#### 1. Univariate Analisis

![image](https://github.com/user-attachments/assets/1d935347-9dec-45b5-89d5-892fb2b538f6)

Visualisasi distribusi variabel kategorik pada dataset restoran menunjukkan sebaran yang cukup merata untuk sebagian besar kategori. Pada fitur **Location**, restoran paling banyak berada di area Suburban (35.3%) dan Rural (35.1%), sementara Downtown menempati porsi lebih kecil (29.6%). Untuk fitur **Cuisine**, jenis masakan terdistribusi relatif seimbang, dengan kategori French (17.3%) dan American (17.2%) mendominasi secara tipis, sedangkan Japanese menjadi yang paling sedikit (15.1%). Sementara itu, variabel **Parking Availability** menunjukkan distribusi yang hampir simetris, dengan proporsi “Yes” (50.0%) dan “No” (50.0%) hampir identik. Temuan ini mengindikasikan bahwa tidak ada dominasi ekstrem pada variabel-variabel kategorik, sehingga seluruh kategori berpotensi memiliki pengaruh terhadap pendapatan restoran yang perlu dianalisis lebih lanjut.

![image](https://github.com/user-attachments/assets/3ca0b3cc-23a7-4448-9746-7d444b47242e)

Visualisasi distribusi variabel numerik pada dataset restoran memperlihatkan bahwa sebagian besar fitur memiliki sebaran yang tidak simetris atau cenderung skewed. Beberapa variabel seperti **Marketing Budget**, **Social Media Followers**, serta **Weekend** dan **Weekday Reservations** menunjukkan pola skewed ke kanan, yang mengindikasikan keberadaan outlier atau nilai ekstrem di sisi kanan distribusi. Sebaliknya, variabel seperti **Rating**, **Seating Capacity**, **Ambience Score**, dan **Service Quality Score** tampak relatif simetris atau mendekati distribusi normal. Beberapa variabel lainnya seperti **Average Meal Price**, **Chef Experience Years**, dan **Number of Reviews** memperlihatkan distribusi multimodal atau sebaran data yang lebar. Potensi keberadaan outlier juga harus ditangani dengan cermat agar tidak mengganggu performa model prediktif yang akan dibangun.

![image](https://github.com/user-attachments/assets/c475a711-e5cf-40fb-a7c2-ca4ad3fa5f91)

Distribusi Revenue divisualisasikan menggunakan histogram yang menunjukkan sebagian besar restoran dalam dataset memiliki pendapatan (revenue) berkisar antara $400.000 hingga $600.000. Grafik memperlihatkan distribusi data yang condong ke kanan (right-skewed), di mana sebagian kecil restoran memiliki revenue di atas $1 juta, namun mayoritas berada di kisaran menengah. Hal ini menunjukkan bahwa pendapatan tinggi hanya dimiliki oleh segelintir restoran, sementara sebagian besar lainnya berada pada tingkat pendapatan menengah.

#### 2. Bivariate Analisis

![image](https://github.com/user-attachments/assets/28b4998a-8865-4751-a2cb-443dd1585066)


Grafik menunjukkan bahwa terdapat hubungan positif antara jumlah pengikut media sosial dengan rata-rata pendapatan (Revenue) restoran. Restoran dengan pengikut sedikit (0–20k) memiliki pendapatan terendah, sementara restoran dengan pengikut sangat banyak (>60k) mencatatkan pendapatan rata-rata tertinggi, mendekati $900.000. Tren ini mengindikasikan bahwa semakin tinggi eksposur restoran di media sosial, semakin besar peluang untuk menarik pelanggan dan meningkatkan pendapatan. Oleh karena itu, strategi pemasaran digital dan peningkatan interaksi di media sosial tampaknya menjadi faktor penting dalam mendukung performa keuangan restoran.


![image](https://github.com/user-attachments/assets/71c55c4a-f594-43a7-a1bb-e7c428cf6760)


Grafik menunjukkan bahwa terdapat hubungan positif yang jelas antara kapasitas tempat duduk restoran dengan rata-rata pendapatan (Revenue). Restoran dengan kapasitas kecil (29–45 kursi) memiliki pendapatan terendah, sementara restoran dengan kapasitas sangat besar (75–90 kursi) mencatatkan pendapatan tertinggi, mendekati $900.000. Tren yang konsisten ini mengindikasikan bahwa semakin besar daya tampung pelanggan, semakin besar pula potensi pemasukan restoran. Hal ini logis karena lebih banyak kursi memungkinkan lebih banyak transaksi dalam satu waktu, sehingga kapasitas tempat duduk merupakan faktor operasional kunci dalam memaksimalkan revenue.


![image](https://github.com/user-attachments/assets/ca80a8d0-a6a6-4654-bd03-aacf61355c22)


Grafik menunjukkan bahwa semakin tinggi rata-rata harga makanan (Average Meal Price), semakin besar pula rata-rata pendapatan (Revenue) yang diperoleh restoran. Restoran dengan kategori harga murah ($25–$35) memiliki pendapatan terendah, sementara kategori sangat mahal (>$60) mencatatkan pendapatan tertinggi, mendekati $900.000. Tren ini mengindikasikan bahwa strategi penetapan harga yang lebih tinggi dapat meningkatkan revenue, kemungkinan karena margin keuntungan yang lebih besar per transaksi atau karena harga yang lebih tinggi sering diasosiasikan dengan nilai tambah seperti kualitas, lokasi premium, atau segmentasi pelanggan menengah ke atas. Namun, efektivitas strategi ini tetap bergantung pada daya beli dan ekspektasi target pasar.


![image](https://github.com/user-attachments/assets/4dee1c46-9a18-438a-8e38-968cf833f953)

Scatterplot antara Marketing Budget dan Revenue menunjukkan adanya kecenderungan hubungan positif, di mana semakin besar alokasi anggaran pemasaran, pendapatan restoran cenderung meningkat. Namun demikian, sebaran data yang cukup lebar mengindikasikan bahwa hubungan ini tidak sepenuhnya linier dan dipengaruhi oleh faktor-faktor lain di luar budget semata. Selain itu, masih ditemukan banyak restoran dengan revenue tinggi meskipun memiliki marketing budget relatif rendah, yang mengisyaratkan bahwa variabel lain seperti jenis masakan, lokasi, atau kualitas layanan juga memainkan peran penting.


![image](https://github.com/user-attachments/assets/7c3adb31-e027-48ff-8c85-f524631f6ad9)

Grafik menunjukkan bahwa peningkatan jumlah reservasi akhir pekan (Weekend Reservations) berbanding lurus dengan peningkatan rata-rata pendapatan restoran. Restoran dengan sedikit reservasi memiliki pendapatan terendah, sementara restoran yang sangat sibuk pada akhir pekan mencatatkan rata-rata pendapatan tertinggi, melebihi $850.000. Tren ini mencerminkan bahwa akhir pekan merupakan momen penting dalam mendongkrak penjualan, kemungkinan karena volume pengunjung yang lebih tinggi. Oleh karena itu, mengoptimalkan strategi reservasi dan promosi khusus di akhir pekan dapat menjadi kunci dalam meningkatkan performa pendapatan restoran secara keseluruhan.

#### 3. Multivariate Analisis

![image](https://github.com/user-attachments/assets/714e90a0-e94d-4c66-9462-ba6bc11b5682)

Heatmap korelasi menunjukkan bahwa Seating Capacity dan Average Meal Price memiliki korelasi positif tertinggi dengan revenue (sekitar 0.68-0.69), diikuti oleh Marketing Budget dan Social Media Followers. Fitur-fitur lain seperti Service Quality Score dan Ambience Score menunjukkan korelasi lemah terhadap revenue. Korelasi tinggi antara Marketing Budget dan Social Media Followers (0.99) mengindikasikan potensi multikolinearitas yang perlu diantisipasi saat pemodelan.

![image](https://github.com/user-attachments/assets/07631749-d16c-4f5d-8f73-737eba5b64c0)

Berdasarkan grafik di atas, dapat disimpulkan bahwa **Average Meal Price** dan **Seating Capacity** memiliki korelasi paling tinggi terhadap pendapatan (Revenue), masing-masing dengan nilai korelasi mendekati 0.7, yang menunjukkan hubungan linier positif yang kuat. Artinya, semakin tinggi harga rata-rata makanan dan kapasitas tempat duduk, cenderung semakin tinggi pula pendapatan restoran. Fitur lain seperti **Marketing Budget**, **Social Media Followers**, dan **Weekend Reservations** juga berkontribusi secara positif, meskipun dengan kekuatan korelasi yang lebih lemah (sekitar 0.3–0.4). Temuan ini menunjukkan bahwa faktor operasional dan pemasaran sama-sama memengaruhi performa finansial restoran, namun faktor harga dan kapasitas memiliki dampak paling dominan.


## Data Preparation

### Menghapus Kolom Tidak Relevan

   ```python
   # Menghapus kolom 'Name' yang tidak dibutuhkan dalam analisis
   df.drop(['Name'], axis=1, inplace=True)
   df.head()
   ```

- Kolom `Name` dihapus karena hanya berisi identitas unik restoran.
- Kolom ini tidak memiliki nilai prediktif terhadap target `Revenue`.
- Menghapus fitur seperti ini juga membantu menyederhanakan struktur data.

### Penanganan Outlier
   ```python
   # Penanganan outlier
   numeric_restaurant = df.select_dtypes(include=np.number)

   Q1 = numeric_restaurant.quantile(0.25)
   Q3 = numeric_restaurant.quantile(0.75)
   IQR = Q3-Q1

   outlier_condition = (numeric_restaurant < (Q1 - 1.5 * IQR)) | (numeric_restaurant > (Q3 + 1.5 * IQR))
   df = df[~outlier_condition.any(axis=1)]

   df.shape
   ```

- Outlier dideteksi menggunakan metode **Interquartile Range (IQR)**.
- Fokus utama penanganan outlier berada pada kolom **Revenue**.
- Langkah-langkahnya:
  - Hitung nilai **Q1 (25th percentile)** dan **Q3 (75th percentile)**.
  - Hitung **IQR = Q3 - Q1**.
  - Tentukan batas bawah (**Q1 - 1.5 * IQR**) dan batas atas (**Q3 + 1.5 * IQR**).
  - Data di luar batas tersebut dianggap sebagai outlier.
- Outlier pada `Revenue` dihapus agar model tidak bias oleh nilai ekstrem.
- Tujuannya adalah menjaga performa model tetap stabil dan akurat.

### Pemisahan Fitur dan Target serta Pembagian Dataset

   ```python
   # Membagi dataset menjadi data latih dan data uji
   df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
   X = df.drop('Revenue', axis=1)
   y = df['Revenue']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

- Dataset dipisahkan menjadi:
  - **X** → Semua fitur (independen)
  - **y** → Target prediksi, yaitu kolom `Revenue`
- Dataset dibagi menjadi data latih dan data uji dengan rasio **80:20**

### Identifikasi Fitur Kategorikal dan Numerikal

Identifikasi ini penting karena fitur kategorikal memerlukan proses encoding agar dapat digunakan dalam model machine learning, sedangkan fitur numerikal perlu distandarisasi untuk model tertentu. Perlakuan yang sesuai pada setiap jenis fitur memastikan proses preprocessing berjalan efektif dan mencegah error pada pipeline model.

   ```python
   # Mengidentifikasi kolom kategorikal
   categorical_features = df.select_dtypes(include='object').columns
   print("Fitur Kategorikal:", categorical_features)
   ```

   ![image](https://github.com/user-attachments/assets/3076d802-df2e-4ee1-b6a1-ec2562a6af36)

Kolom kategorikal diidentifikasi dengan select_dtypes(include='object'), seperti location, cuisine, dan parking availability. Fitur-fitur ini nantinya perlu diubah menjadi representasi numerik melalui encoding sebelum digunakan dalam model machine learning.

   ```python
   # Mengidentifikasi kolom numerik
   numerical_features = X.select_dtypes(include=np.number).columns
   print("Fitur Numerikal:", numerical_features)
   ```

   ![image](https://github.com/user-attachments/assets/963351af-0bcc-4bb9-a9c1-e6fa238f2223)

Tahap ini bertujuan mengidentifikasi kolom numerik dalam DataFrame X menggunakan select_dtypes(include=np.number). Kolom-kolom seperti seating capacity, average meal price, marketing budget, revenue, dan lain-lain termasuk dalam kategori numerik, dan akan diproses lebih lanjut sesuai kebutuhan analisis dan modeling.

### Encoding Fitur Kategorikal

   ```python
   # Menggunakan ColumnTransformer untuk menerapkan transformasi yang berbeda pada kolom yang berbeda
   preprocessor_drop_model = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)],
    remainder='drop')

   X_train_processed_drop = preprocessor_drop_model.fit_transform(X_train)
   X_test_processed_drop = preprocessor_drop_model.transform(X_test)
   # Mengubah hasil encoding menjadi DataFrame
   feature_names_cat = preprocessor_drop_model.named_transformers_['cat'].get_feature_names_out(categorical_features)
   feature_names_processed_drop = list(feature_names_cat) + list(numerical_features)

   X_train_processed_drop_dense = X_train_processed_drop
   X_test_processed_drop_dense = X_test_processed_drop

   X_train_processed_drop_df = pd.DataFrame(X_train_processed_drop_dense,   columns=feature_names_processed_drop)
   X_test_processed_drop_df = pd.DataFrame(X_test_processed_drop_dense, columns=feature_names_processed_drop)

   print(X_train_processed_drop_df.head())
   print(X_test_processed_drop_df.head())
   ```

- Fitur kategorikal diubah ke bentuk numerik menggunakan **OneHotEncoder**
- Encoding dilakukan setelah split untuk menghindari data leakage
- Fitur yang di-encode:
  - Location
  - Cuisine
  - Parking Availability
- Gunakan `handle_unknown='ignore'` agar aman kalau ada kategori baru di data uji

### Standarisasi Fitur Numerikal

   ```python
   # Standardisasi Fitur Numerik untuk model KNN dan SVR
   scaler = StandardScaler()
   # Standardisasi variabel X
   X_train_scaled = scaler.fit_transform(X_train_processed_drop_df)
   X_test_scaled = scaler.transform(X_test_processed_drop_df)
   # Standardisasi target Y
   y_scaler = StandardScaler()
   y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()

   # Ubah kembali ke DataFrame
   X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_processed_drop_df.columns)
   X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_processed_drop_df.columns)
   ```

- Fitur numerikal dinormalisasi menggunakan **StandardScaler**
- Penting untuk model yang sensitif terhadap skala seperti **KNN** dan **SVR**
- Hasilnya: semua fitur numerik punya mean = 0 dan std = 1

## Modeling

Dalam proyek prediksi revenue restoran ini, beberapa algoritma regresi diterapkan untuk memodelkan hubungan antara fitur-fitur restoran dan total pendapatan (Revenue) dengan beberapa alasan yang mendasari pemilihan algoritma tersebut. Linear Regression digunakan sebagai baseline model untuk mengidentifikasi hubungan linier dasar. Support Vector Regression (SVR) diaplikasikan untuk menangkap pola non-linier dengan pendekatan margin dan kernel, sehingga mampu memodelkan hubungan kompleks antar variabel. Random Forest Regressor digunakan sebagai ensemble berbasis pohon yang memadukan prediksi banyak pohon untuk meningkatkan akurasi dan mengurangi overfitting. Gradient Boosting Regressor memperbaiki kesalahan prediksi secara bertahap sehingga seringkali menghasilkan performa terbaik. Terakhir, K-Nearest Neighbors (KNN) Regressor diterapkan untuk memprediksi revenue berdasarkan kemiripan fitur antar restoran. Setiap model memiliki karakteristik, kelebihan, dan kekurangannya masing-masing. Oleh karena itu, digunakan beberapa model untuk membandingkan efektivitas dan akurasi prediksi revenue. Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan prediksi revenue restoran.

### 1. Linear Regression

**Linear Regression** digunakan untuk memprediksi revenue restoran berdasarkan hubungan linear dengan fitur-fitur seperti seating capacity, average meal price, marketing budget, dan lain-lain. Model ini diinisialisasi menggunakan LinearRegression() dari scikit-learn, yang memanfaatkan metode Ordinary Least Squares (OLS) untuk menghitung koefisien terbaik. Parameter umum seperti fit_intercept=True digunakan untuk memperkirakan nilai konstanta (intercept) secara otomatis.

   ```python
   # Model 1: Linear Regression
   linear_model_drop = LinearRegression()
   linear_model_drop.fit(X_train_processed_drop_df, y_train)
   y_pred_linear_drop = linear_model_drop.predict(X_test_processed_drop_df)
   ```

- **Kelebihan**:
  - Interpretasi koefisien mudah dan intuitif.
  - Implementasi sederhana dan efisien.
  - Komputasi cepat dan ringan.

- **Kekurangan**:
  - Mengasumsikan hubungan linier (tidak fleksibel untuk pola non-linear).
  - Sensitif terhadap outlier.
  - Tidak mampu menangkap interaksi kompleks antar fitur.

### 2. Random Forest Regressor

**Random Forest Regressor** adalah model ensemble yang memanfaatkan banyak decision tree untuk memprediksi revenue restoran. Model diinisialisasi dengan parameter n_estimators=200, max_depth=None, dan random_state=42 menggunakan scikit-learn, untuk memperoleh hasil prediksi yang stabil dan akurat. Model dilatih dengan data latih yang sudah diproses, lalu digunakan untuk memprediksi revenue pada data uji.

   ```python
   # Model 2: Random Forest Regressor 
   best_forest_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42
   )
   best_forest_model.fit(X_train_processed_drop_df, y_train)
   y_pred_best_forest = best_forest_model.predict(X_test_processed_drop_df)
   ```
- **Alasan Pemilihan Parameter**:
  - `n_estimators=200`: Dipilih berdasarkan hasil percobaan bertahap (dari 100, 150, hingga 200) yang menunjukkan bahwa peningkatan jumlah pohon sampai 200 memberikan stabilitas prediksi tanpa menambah waktu pelatihan secara signifikan.
  - `max_depth=None`: Tidak dibatasi untuk mengizinkan pohon tumbuh penuh. Pada tahap percobaan, pembatasan kedalaman (max_depth=10–20) justru menurunkan performa.
  - `random_state=42`: Digunakan agar hasil replikasi tetap konsisten.

- **Kelebihan**:
  - Lebih akurat dibanding single decision tree.
  - Mengurangi overfitting dengan averaging hasil banyak pohon.
  - Dapat memberikan estimasi pentingnya fitur (feature importance).

- **Kekurangan**:
  - Lebih lambat secara komputasi dibanding single tree.
  - Interpretasi hasil model lebih sulit dibanding pohon tunggal.

### 3. Gradient Boosting Regressor

**Gradient Boosting Regressor** adalah model ensemble yang membangun decision tree secara bertahap, di mana setiap pohon berfokus memperbaiki kesalahan prediksi dari pohon sebelumnya. Model diinisialisasi dengan GradientBoostingRegressor(random_state=42) dari scikit-learn untuk memastikan hasil yang konsisten. Model ini dilatih menggunakan data latih yang telah diproses, kemudian digunakan untuk memprediksi revenue pada data uji.

   ```python
   # Model 3: Gradient Boosting Regressor
   gb_model_drop = GradientBoostingRegressor(random_state=42)
   gb_model_drop.fit(X_train_processed_drop_df, y_train)
   y_pred_gb_drop = gb_model_drop.predict(X_test_processed_drop_df)
   ```
- **Alasan Pemilihan Parameter**:
   Parameter default digunakan (`learning_rate=0.1, n_estimators=100`) karena hasil awal sudah cukup baik, sehingga tidak perlu dilakukan tuning.

- **Kelebihan**:
  - Akurasi tinggi dengan kemampuan menangkap pola kompleks.
  - Fleksibel untuk berbagai fungsi loss.
  - Dapat mengatasi outlier dengan lebih baik dibanding Random Forest.

- **Kekurangan**:
  - Rentan overfitting jika parameter tidak diatur dengan benar.
  - Waktu pelatihan lebih lama dibanding Random Forest.
  - Interpretasi model lebih sulit.

### 4. K-Nearest Neighbors (KNN) Regressor

**K-Nearest Neighbors (KNN) Regressor** memprediksi revenue restoran dengan mencari k restoran terdekat (paling mirip) dalam data pelatihan berdasarkan fitur-fitur seperti kapasitas tempat duduk, harga makanan, dan anggaran pemasaran. Prediksi revenue dilakukan dengan menghitung rata-rata revenue dari k tetangga terdekat tersebut. Agar perhitungan jarak akurat, penting untuk memastikan semua fitur berada dalam skala yang sama. Pemilihan nilai k harus diperhatikan karena sangat mempengaruhi hasil prediksi.

   ```python
   # Model 4: K-Nearest Neighbor (KNN)
   knn_model = KNeighborsRegressor(n_neighbors=5)
   knn_model.fit(X_train_scaled_df, y_train)
   y_pred_knn = knn_model.predict(X_test_scaled_df)
   ```
- **Alasan Pemilihan Parameter**:
   `n_neighbors=5`: Merupakan nilai default dan juga diuji coba dari k = 3 hingga 10. Nilai 5 memberikan hasil terbaik dibandingkan nilai k lainnya dalam hal keseimbangan antara bias dan variansi. Nilai ini juga sesuai dengan prinsip umum bahwa nilai ganjil (untuk regresi dan klasifikasi) sering menghindari situasi tie atau overfitting.

- **Kelebihan**:
  - Sederhana dan mudah dipahami.
  - Tidak mengasumsikan bentuk hubungan linier atau non-linier.
  - Dapat bekerja baik untuk dataset kecil dengan pola lokal.

- **Kekurangan**:
  - Sensitif terhadap skala fitur (memerlukan standarisasi).
  - Pemilihan parameter `k` sangat mempengaruhi hasil.
  - Kurang efisien pada dataset besar karena menghitung jarak ke semua data latih.

### 5. Support Vector Regression (SVR)

**Support Vector Regression (SVR)** memprediksi revenue restoran dengan mencari fungsi yang paling sesuai untuk memetakan hubungan antara fitur (misalnya kapasitas tempat duduk, harga makanan, anggaran pemasaran) dan target (revenue) menggunakan pendekatan margin. SVR berusaha menemukan garis (atau kurva) terbaik yang memiliki deviasi maksimal ε dari nilai aktual, namun tetap sederhana (maksimal margin minimum error). Model ini sangat efektif dalam menangkap hubungan non-linear melalui penggunaan kernel, misalnya RBF (radial basis function).

   ```python
   # Model 5: Support Vector Regression (SVR)
   svr_model = SVR(kernel='rbf', C=1000, epsilon=0.1)
   svr_model.fit(X_train_scaled_df, y_train_scaled)
   y_pred_scaled = svr_model.predict(X_test_scaled_df)
   y_pred_svr = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
   ```
- **Alasan Pemilihan Parameter**:
  - `C=1000`: Dipilih setelah uji coba beberapa nilai (C=1, 10, 100, 1000), dan C=1000 memberikan margin yang cukup fleksibel untuk menampung variasi revenue yang tinggi.
  - `epsilon=0.1`: Nilai default yang mengatur toleransi kesalahan prediksi. Karena targetnya adalah revenue dalam satuan ribuan, nilai ini masih masuk akal.
  - `kernel='rbf'`: Kernel non-linear ini dipilih karena data menunjukkan pola non-linier antar fitur.

- **Kelebihan**:
  - Mampu menangkap pola non-linear melalui penggunaan kernel.
  - Cukup robust terhadap outlier ringan.
  - Cocok untuk data berdimensi tinggi.

- **Kekurangan**:
  - Memerlukan penskalaan fitur dan target.
  - Sensitif terhadap pemilihan parameter (seperti C dan epsilon).
  - Kurang efisien untuk dataset besar dan interpretasi model sulit.

### Perbandingan Model
| Model               | Kelebihan                                                             | Kekurangan                                                              |
|---------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Linear Regression   | Mudah diinterpretasikan, cepat dihitung                               | Tidak menangkap hubungan non-linear, sensitif terhadap outlier           |
| Random Forest       | Akurasi tinggi, tahan outlier, mendukung feature importance           | Komputasi lebih berat, interpretasi lebih sulit                          |
| Gradient Boosting   | Akurasi sangat tinggi, menangani kesalahan bertahap                   | Rentan overfitting, tuning kompleks                                      |
| KNN                 | Sederhana, tidak membuat asumsi                                      | Sensitif terhadap skala data, tidak efisien untuk dataset besar          |
| SVR                 | Menangani non-linear dengan baik, robust untuk dataset kecil          | Butuh scaling, tuning parameter rumit, tidak efisien untuk dataset besar |

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

| Model                             | MSE              | RMSE      | MAE       | R²     | Rank MSE | Rank RMSE | Rank MAE | Rank R² | Total Score |
|----------------------------------|------------------|-----------|-----------|--------|----------|------------|----------|---------|--------------|
| Linear Regression                | 2,672,378,391.21 | 51,695.05 | 39,515.84 | 0.9569 | 4        | 4          | 4        | 4       | 16           |
| **Random Forest Regressor**      | 64,171,992.36    | 8,010.74  | 6,175.19  | 0.9990 | 1        | 1          | 1        | 1       | 4            |
| Gradient Boosting Regressor      | 94,071,218.79    | 9,699.03  | 7,470.66  | 0.9985 | 2        | 2          | 2        | 2       | 8            |
| K-Nearest Neighbors Regressor (K=5) | 5,944,692,875.42 | 77,101.83 | 61,829.45 | 0.9041 | 5        | 5          | 5        | 5       | 20           |
| Support Vector Regression        | 385,510,196.76   | 19,634.41 | 15,119.92 | 0.9938 | 3        | 3          | 3        | 3       | 12           |


Model terbaik berdasarkan gabungan semua metrik: **Random Forest Regressor** (Total Score = 4)

Pemilihan metrik evaluasi tersebut sesuai dengan konteks regresi pada prediksi nilai numerik seperti revenue, di mana kesalahan prediksi (dalam satuan asli maupun absolut) dan kemampuan model menjelaskan variasi data menjadi indikator utama performa model.

## Feature Importance

Bagian ini membahas tentang fitur-fitur apa saja yang paling berpengaruh terhadap prediksi revenue restoran, berdasarkan hasil feature importance dari beberapa model regresi yang telah diterapkan.

![image](https://github.com/user-attachments/assets/c64879e8-ff51-45c3-bf29-3a29faebf14f)

![image](https://github.com/user-attachments/assets/0faa1b95-c085-43c6-a905-e29518e1cd85)

![image](https://github.com/user-attachments/assets/5faee70e-fb88-41a0-8f3d-433623f6230c)

Berdasarkan hasil analisis feature importance dari tiga model regresi—Linear Regression, Random Forest Regressor, dan Gradient Boosting Regressor—ditemukan bahwa fitur-fitur berikut secara konsisten memiliki pengaruh paling signifikan terhadap prediksi revenue restoran:

- **Average Meal Price** : Merupakan fitur dengan kontribusi tertinggi di semua model. Semakin tinggi harga rata-rata makanan, semakin besar potensi revenue, terutama jika dikombinasikan dengan strategi nilai tambah seperti kualitas dan positioning pasar.

- **Seating Capacity** : Berperan besar dalam menentukan skala operasional restoran. Model Random Forest dan Gradient Boosting sama-sama menunjukkan bahwa kapasitas tempat duduk adalah salah satu penentu utama revenue, karena memengaruhi volume pelanggan yang bisa dilayani.

- **Cuisine Type (khususnya French dan Japanese)** : Dalam Linear Regression, jenis masakan tertentu seperti French dan Japanese memiliki koefisien besar, menunjukkan bahwa preferensi menu juga berkontribusi terhadap pendapatan.

Selain ketiga fitur utama tersebut, beberapa fitur lain juga memberikan kontribusi meskipun tidak sebesar faktor di atas, seperti:

- **Marketing Budget** : Meskipun memiliki pengaruh rendah di Random Forest dan Gradient Boosting, model Linear Regression menunjukkan bahwa alokasi anggaran promosi tetap relevan untuk mendorong pendapatan, terutama dalam konteks visibilitas awal.

- **Social Media Followers** : Memberikan kontribusi kecil namun tetap berperan dalam memperluas jangkauan dan menarik pelanggan baru. 

- **Service Quality Score** dan **Ambience Score** : Meskipun nilainya rendah, kedua fitur ini tetap relevan dalam menciptakan pengalaman pelanggan yang berkesan dan mendukung loyalitas jangka panjang.

## Conclusion

Berdasarkan hasil eksperimen dan evaluasi terhadap berbagai model regresi yang diterapkan, proyek ini berhasil menjawab rumusan masalah yang telah ditetapkan, yaitu memprediksi revenue restoran berdasarkan fitur-fitur karakteristik restoran serta memahami hubungan antar fitur terhadap pendapatan.

1. Proyek ini telah mengembangkan beberapa model prediksi revenue dengan pendekatan Machine Learning, yaitu Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, K-Nearest Neighbors (KNN) Regressor, dan Support Vector Regression (SVR). Evaluasi performa model menggunakan metrik MSE, RMSE, MAE, dan R² menunjukkan bahwa **Random Forest Regressor** menjadi model terbaik dengan akurasi tertinggi dan kesalahan prediksi terendah. Model ini dipilih karena mampu menggabungkan prediksi dari banyak decision tree secara ensemble, sehingga menghasilkan model yang kuat, akurat, dan tahan terhadap overfitting.

2. Hasil analisis feature importance dari Linear Regression, Random Forest Regressor, dan Gradient Boosting Regressor menunjukkan bahwa **Average Meal Price** dan **Seating Capacity** merupakan faktor paling dominan dalam memprediksi revenue restoran. Jenis masakan tertentu seperti **French** dan **Japanese** juga berkontribusi signifikan dalam model linier. Sementara itu, fitur seperti **Marketing Budget**, **Social Media Followers**, serta **Service Quality** dan **Ambience Score** memiliki pengaruh lebih kecil, namun tetap relevan dalam membentuk pengalaman pelanggan dan meningkatkan pendapatan secara tidak langsung.

### Insight
Dengan memanfaatkan model Machine Learning, khususnya Random Forest Regressor, prediksi revenue restoran dapat dilakukan secara akurat berdasarkan data karakteristik restoran. Selain itu, pemahaman mengenai fitur-fitur yang paling berpengaruh dapat menjadi acuan strategis bagi pelaku bisnis restoran dalam mengoptimalkan pendapatan melalui pengelolaan anggaran pemasaran, penetapan harga menu, dan manajemen kapasitas layanan. Model yang telah dibangun menunjukkan bahwa penerapan kecerdasan buatan dalam analisis bisnis restoran memberikan manfaat nyata dalam mendukung pengambilan keputusan berbasis data.


## Referensi
- Dataset: [Restaurant Revenue Prediction - Kaggle](https://www.kaggle.com/c/restaurant-revenue-prediction)
- Bera, S. (2020). An application of operational analytics: for predicting sales revenue of restaurant. In Machine learning algorithms for industrial applications (pp. 209-235). Cham: Springer International Publishing.
- Fatmah, F., Supriyanto, E., Budiman, D., Maichal, M., Ghozali, Z., Ismail, H., ... & Musty, B. (2024). UMKM & kewirausahaan: Panduan praktis. PT. Sonpedia Publishing Indonesia.
- Gogolev, S., & Ozhegov, E. M. (2019, July). Comparison of machine learning algorithms in restaurant revenue prediction. In International Conference on Analysis of Images, Social Networks and Texts (pp. 27-36). Cham: Springer International Publishing.
- Jarlöv, S., & Svensson Dahl, A. (2023). Restaurant Daily Revenue Prediction: Utilizing Synthetic Time Series Data for Improved Model Performance.
- Sanjana Rao, G. P., Aditya Shastry, K., Sathyashree, S. R., & Sahu, S. (2021). Machine learning based restaurant revenue prediction. In Evolutionary Computing and Mobile Sustainable Networks: Proceedings of ICECMSN 2020 (pp. 363-371). Springer Singapore.
- Wulandari, A. R., Arvi, A. A., Iqbal, M. I., Tyas, F., Kurniawan, I., & Anshori, M. I. (2023). Digital Hr: Digital transformation in increasing productivity in the work environment. Jurnal Publikasi Ilmu Manajemen, 2(4), 29-42.
