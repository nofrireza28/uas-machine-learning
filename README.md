# UAS Machine Learning - Estimasi Harga Kendaraan Bekas

**Kasus yang dipilih: Kasus B - Regresi: Estimasi Harga Kendaraan Bekas**

|||
|---|---|
| Nama | Nofri Reza |
| NIM | 1003240028 |
| Program Studi | Informatika |
| Institusi | Institut Teknologi Tangerang Selatan |
| Mata Kuliah | Machine Learning |

---

## 1. Deskripsi Masalah

Sebuah marketplace otomotif ingin menyarankan harga jual yang wajar kepada penjual kendaraan bekas. Saat ini penjual menentukan harga berdasarkan perkiraan pribadi, sehingga sebagian memasang harga terlalu tinggi (unit tidak laku dan menumpuk di daftar) dan sebagian terlalu rendah (penjual dirugikan, marketplace kehilangan kepercayaan).

Proyek ini membangun model regresi yang memprediksi **harga jual kendaraan bekas** (`selling_price`, numerik kontinu) dari karakteristik kendaraan: umur, jarak tempuh, jenis bahan bakar, transmisi, tipe penjual, riwayat kepemilikan, konsumsi bahan bakar, kapasitas mesin, tenaga maksimum, dan jumlah kursi. Model kemudian dilayani sebagai REST API sehingga sistem marketplace dapat memanggilnya secara langsung saat penjual mengisi formulir penjualan.

---

## 2. Sumber dan Lisensi Data

|Item|Keterangan|
|---|---|
| Nama dataset | Car Price Prediction Dataset (CarDekho) |
| URL | https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset |
| Lisensi | **CC0 1.0 Universal** - Public Domain Dedication |
| Jumlah baris | 8.128 |
| Jumlah kolom | 12 |

### Kamus Kolom

| Kolom | Tipe mentah | Keterangan |
|---|---|---|
| `name` | teks | Nama lengkap varian kendaraan (2.058 nilai unik) |
| `year` | int | Tahun pembuatan (1983–2020) |
| `selling_price` | int | **Target.** Harga jual dalam Rupee India (INR) |
| `km_driven` | int | Jarak tempuh dalam kilometer |
| `fuel` | teks | Diesel, Petrol, CNG, LPG |
| `seller_type` | teks | Individual, Dealer, Trustmark Dealer |
| `transmission` | teks | Manual, Automatic |
| `owner` | teks | First / Second / Third / Fourth & Above Owner, Test Drive Car |
| `mileage(km/ltr/kg)` | float | Konsumsi bahan bakar |
| `engine` | float | Kapasitas mesin dalam CC |
| `max_power` | **teks** | Tenaga maksimum dalam bhp — tersimpan sebagai teks |
| `seats` | float | Jumlah kursi |

---

## 3. Lingkungan Pengembangan

Versi berikut adalah versi yang benar-benar dipakai pada saat pengerjaan. Angka ini dicetak otomatis oleh `src/load_data.py` sehingga selalu dapat diverifikasi ulang.

| Komponen | Versi |
|---|---|
| Python | 3.13.3 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.0 |
| numpy | 2.5.1 |
| FastAPI | 0.139.2 (lingkungan serving) |
| Sistem operasi | Windows, shell PowerShell |

Versi `scikit-learn`, `pandas`, dan `numpy` pada `requirements-api.txt` di tulis ke angka yang sama persis dengan tabel di atas. Artefak `.joblib` dihasilkan oleh versi tersebut, dan memuatnya dengan versi lain berisiko gagal atau berubah perilaku secara diam-diam.

---

## 4. Menjalankan Proyek dari Nol

Perintah di bawah ditulis untuk **PowerShell di Windows**, Untuk Linux/macOS disertakan sebagai komentar.

### 4.1 Clone Repository

```powershell
git clone <URL-REPO-ANDA>
cd uas-ml-1003240028
```

### 4.2 Buat dan aktifkan virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
```

Jika PowerShell tidak bisa menjalankan skrip aktivasi, jalankan perintah berikut:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 4.3 Install dependensi training

```powershell
pip install -r requirements.txt
```

### 4.4 Siapkan dataset

Folder `data/` sengaja kosong di repo (lihat bagian 5). Siapkan datanya dengan salah satu cara berikut.

**Cara A - Kaggle CLI (otomatis):**

```powershell
pip install kaggle
# Letakkan kaggle.json di %USERPROFILE%\.kaggle\
kaggle datasets download -d sukhmandeepsinghbrar/car-price-prediction-dataset -p data/raw --unzip
# Pastikan nama file adalah data/raw/cardekho.csv
```

**Cara B - Unduh manual:**

1. Buka https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset
2. Klik **Download** lalu ekstrak arsipnya
3. Salin berkas CSV ke `data/raw/` dan beri nama `cardekho.csv`

### 4.5 Jalankan load_data.py untuk Load data  dan Verifikasi

```powershell
python src/load_data.py
```

Hasil yang dikeluarkan : versi library, jumlah baris, jumlah kolom, tipe tiap kolom, jumlah nilai hilang per kolom. Bila file dataset belum ada, skrip berhenti dengan menampilkan petunjuk untuk download dataset.

### 4.6 Jalankan analisis data eksploratif (EDA)

```powershell
python src/eda.py
```

Skrip ini mencetak tujuh temuan data kotor data beserta angka pendukungnya, lalu menyimpan lima grafik ke `reports/`:

| Berkas | Isi |
|---|---|
| `01_sebaran_target.png` | Sebaran harga, skala asli vs logaritmik |
| `02_umur_vs_harga.png` | Kurva depresiasi terhadap umur kendaraan |
| `03_kategori_vs_harga.png` | Boxplot harga per transmisi, bahan bakar, kepemilikan |
| `04_korelasi.png` | Matriks korelasi Pearson antar variabel numerik |
| `05_outlier_km.png` | Dampak outlier jarak tempuh, skala asli vs log-log |

### 4.7 Latih model

```powershell
python src/train.py
```

Skrip memisahkan data 80/20 (`random_state=42`), membandingkan tiga algoritma dengan 5-fold cross-validation **pada data latih saja**, memilih pemenang berdasarkan MAE, lalu menyentuh test set tepat satu kali. Hasil yang dikeluarkan adalah `models/model.joblib` dan `models/metadata.json`.

### 4.8 Evaluasi dan periksa prakiraan

```powershell
python src/evaluate.py
```

Memuat model yang telah dilatih, menghitung permutation importance, menyimpan tiga grafik diagnostik (`06`–`08`) ke `reports/`, dan memeriksa ketiga prakiraan pra-pelatihan.

**Ringkasan hasil:** model terpilih HistGradientBoosting dengan target log, MAE test set 70.202 INR, R² 0,926, median galat relatif 12,12%.

### 4.9 Jalankan API

Lingkungan serving dipisahkan dari lingkungan training:

```powershell
python -m venv .venv-api
.\.venv-api\Scripts\Activate.ps1
pip install -r requirements-api.txt
uvicorn app.main:app --reload

# Jalankan bila port 8000 sudah digunakan
uvicorn app.main:app --reload --host 127.0.0.1 --port [port yang berbeda]

```

Jalankan dari **Root Repo atau Project Directory**, bukan dari dalam `app/`. Dokumentasi interaktif tersedia di http://127.0.0.1:8000/docs

| Endpoint | Metode | Keterangan |
|---|---|---|
| `/` | GET | Informasi layanan dan daftar endpoint |
| `/health` | GET | Status kesiapan, `status` bernilai `sehat` atau `tidak sehat` |
| `/predict-harga` | POST | Prediksi harga kendaraan |
| `/docs` | GET | Dokumentasi interaktif (otomatis dari FastAPI) |

Contoh permintaan:

```powershell
curl -X POST http://127.0.0.1:8000/predict-harga `
  -H "Content-Type: application/json" `
  -d '{"name":"Maruti Swift Dzire VDI","year":2014,"km_driven":145500,
       "fuel":"Diesel","seller_type":"Individual","transmission":"Manual",
       "owner":"First Owner","mileage_kmpl":23.4,"engine_cc":1248,
       "max_power_bhp":74,"seats":5}'
```

Empat fitur numerikspesifikasi (`mileage_kmpl`, `engine_cc`, `max_power_bhp`, `seats`) boleh dikosongkan. Nilai yang kosong akan diisi secara otomatis menggunakan median dari data latih melalui Pipeline

**Kode status yang dikembalikan:**

| Kode | Kapan |
|---|---|
| 200 | Prediksi berhasil |
| 422 | Masukan gagal validasi: tipe salah, di luar rentang, nilai enum tidak dikenal, medan wajib hilang, atau JSON rusak |
| 503 | Model belum termuat (`models/model.joblib` tidak ada) |
| 500 | Galat tak terduga di dalam kode |

Prediksi dicatat ke `logs/prediksi.log` (folder ini masuk `.gitignore`).

### 4.10 Jalankan test otomatis

Test dijalankan di **lingkungan serving** (`.venv-api`), bukan lingkungan training. Alasannya ada di bagian 6.

```powershell
.\.venv-api\Scripts\Activate.ps1
pip install -r requirements-dev.txt
# Konfigurasi test ada di file pytest.ini
python -m pytest
```

**Hasil: 37 test, seluruhnya lolos** - 31 test mekanis di `tests/test_api.py` dan 6 test behavioral di `tests/test_perilaku.py`.

Test mekanis memeriksa kontrak API (endpoint ada, kode status benar, bentuk jawaban sesuai, 25 kasus validasi menghasilkan 422). Test behavioral memeriksa apakah keluaran modelnya masuk akal secara domain. Contoh kendaraan lebih tua harus lebih murah, jarak tempuh lebih tinggi harus lebih murah, dan seterusnya.

Bila `models/model.joblib` belum ada, seluruh test dilewati dengan pesan yang menyuruh menjalankan `python src/train.py` lebih dulu.

---

## 5. Mengapa `data/` dan `models/` Tidak Di-commit

Berkas dataset dan artefak `.joblib` adalah **hasil dari proses, bukan kode sumber**. Keduanya memiliki ukuran yang lumayan besar dan berubah setiap kali model dilatih ulang,
sehingga meng-commit-nya akan memperbesar ukuran repository Git. Selain itu, dataset berasal dari sumber pihak ketiga sehingga lebih baik diunduh langsung dari sumber resminya daripada didistribusikan ulang melalui repositori., mendistribusikan ulang dataset pihak ketiga di dalam repo sendiri berpotensi bersinggungan dengan ketentuan platform sumbernya, meskipun lisensi datanya CC0.

Meskipun demikian kedua folder tetap dapat dibuat kembali. Dataset pada `data/` dapat diperoleh dengan mengikuti langkah yang dijelaskan pada dokumentasi, sedangkan `models/` dapat dibuat kembali dengan menjalankan `python src/train.py`. Karena proses pelatihan menggunakan `random_state` yang tetap, hasil pelatihan dapat direproduksi secara konsisten.

---

## 6. Mengapa Lingkungan Serving Memakai Versi Terkunci

Repo ini memisahkan dependensi menjadi dua berkas dengan kebijakan versi yang berbeda. 

`requirements.txt` (**training**) digunakan untuk proses training. Berkas ini menggunakan rentang versi sehingga lebih fleksibel dan memudahkan proses eksperimen maupun pemasangan pada berbagai lingkungan.

`requirements-api.txt` (**serving**) digunakan untuk serving dan menggunakan versi yang dikunci (==) agar lingkungan produksi tetap konsisten. Alasannya berbeda secara mendasar:

1. **Menjaga Kompatibilitas Model.** sehingga berkas model (.joblib) dapat dimuat dengan benar tanpa masalah perbedaan versi pustaka.
2. **Menjamin Konsistensi Hasil** API dijalankan tanpa pengawasan dari manusia. Bila pemasangan hari ini menggunakan versi berbeda dari pemasangan bulan lalu, sebuah perubahan perilaku bisa masuk ke lingkungan production tanpa ada satu baris kode pun yang berubah. Versi terkunci membuat konsistensi tersebut lebih terjaga.
3. **Mempermudah proses pemeliharaan,.** Saat terjadi insiden, penyebabnya lebih mudah ditelusuri tanpa harus mempertimbangkan perbedaan versi dependensi, sehingga penelusuran akar masalah jauh lebih cepat.

Singkatnya: lingkungan training dioptimalkan untuk kemudahan bereksperimen, lingkungan serving dioptimalkan untuk keterulangan dan penggunaan jangka panjang.

### Lingkungan ketiga: pengujian

`requirements-dev.txt` mewarisi `requirements-api.txt` melalui perintah `-r`, kemudian menambahkan dependensi `pytest` dan `httpx` untuk pengujian.

Pendekatan ini dipilih karena pengujian dilakukan terhadap API secara menyeluruh, mulai dari memuat model `model.joblib`, menjalankan aplikasi FastAPI, hingga mengirim permintaan HTTP menggunakan `TestClient`. Oleh karena itu, lingkungan pengujian memerlukan seluruh dependensi yang digunakan pada lingkungan `serving`, bukan hanya library untuk pengujian.

Alasan yang lebih penting, penggunaan `requirements-api.txt` sebagai dasar, untuk memastikan bahwa proses pengujian menggunakan versi library yang **persis sama** dengan yang dipakai di produksi. Dengan demikian, hasil pengujian lebih mencerminkan kondisi sebenarnya saat aplikasi dijalankan dan mengurangi risiko perbedaan perilaku akibat perbedaan versi dependensi.

---

## 7. Struktur Proyek

```
uas-ml-1003240028/
├─ src/                    # kode training: load data, EDA, train, evaluate
│  └─ load_data.py         # load & profil dataset
├─ app/                    # kode serving: API FastAPI
│  └─ main.py              # endpoint, validasi Pydantic, lifespan
├─ tests/                  # test otomatis pytest
│  ├─ conftest.py          # fixture bersama (klien API, muatan dasar)
│  ├─ test_api.py          # test mekanis
│  └─ test_perilaku.py     # test behavioral
├─ data/                   # dataset (masuk .gitignore)
│  └─ raw/                 # CSV mentah hasil unduhan
├─ models/                 # artefak .joblib (masuk .gitignore)
├─ reports/                # grafik EDA & evaluasi (PNG di-commit)
├─ pytest.ini              # konfigurasi pytest
├─ requirements.txt        # dependensi training (rentang minor)
├─ requirements-api.txt    # dependensi serving (versi di-pin persis)
├─ requirements-dev.txt    # dependensi pengujian (mewarisi requirements-api)
├─ .gitignore
└─ README.md
```
