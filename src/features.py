"""
features.py - Definisi rekayasa fitur dan pembersihan
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Tahun terbaru pada dataset. Dipakai sebagai acuan tetap agar nilai 'umur'
# yang dilihat model saat pelatihan dan saat inferensi berada pada skala yang
# sama. Angka ini disimpan pula ke metadata.json.
TAHUN_ACUAN = 2020

# Kolom mentah yang wajib ada pada masukan.
KOLOM_MASUKAN = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage(km/ltr/kg)",
    "engine",
    "max_power",
    "seats",
]

# Kolom hasil rekayasa yang keluar dari PembersihFitur.
FITUR_NUMERIK = [
    "umur",
    "log_km_driven",
    "mileage(km/ltr/kg)",
    "engine",
    "max_power",
    "seats",
]
FITUR_KATEGORIKAL = [
    "merek",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

TARGET = "selling_price"


class PembersihFitur(BaseEstimator, TransformerMixin):
    """Membersihkan dan merekayasa fitur mentah menjadi fitur siap-model.

    Tindakan yang dilakukan, seluruhnya per-baris:

    1. `max_power` dikonversi dari teks ke angka. Nilai yang tidak dapat
       dikonversi -- termasuk sel berisi spasi -- menjadi NaN agar ditangani
       imputer di tahap berikutnya.
    2. `mileage` bernilai 0 diubah menjadi NaN. Nol di sini mustahil secara
       fisik dan merupakan nilai hilang yang tercatat sebagai angka; bila
       dibiarkan, imputer akan menganggapnya pengamatan sah.
    3. `year` diubah menjadi `umur` terhadap TAHUN_ACUAN. Umur bermakna
       langsung bagi pengguna dan tidak menjadi usang seiring berjalannya
       waktu kalender.
    4. `km_driven` ditransformasi dengan log1p. Sebarannya sangat menceng dan
       hubungannya dengan harga baru terlihat lurus pada skala logaritmik
       (lihat Grafik 5 pada laporan EDA). log1p bersifat monoton sehingga
       tidak mengubah urutan, dan aman untuk nilai 0.
    5. `name` diringkas menjadi `merek` (kata pertama). Kolom `name` punya
       2.058 nilai unik dari 8.128 baris sehingga nyaris menjadi pengenal
       baris; `merek` hanya 32 nilai dan mempertahankan informasi yang
       berguna.
    6. Kolom `name` dan `year` dibuang setelah fitur turunannya dibentuk.
    """

    def fit(self, X: pd.DataFrame, y=None):
        # Tidak ada parameter yang dipelajari. Metode ini hanya memvalidasi
        # skema masukan agar galat muncul lebih awal dan lebih jelas.
        hilang = [k for k in KOLOM_MASUKAN if k not in X.columns]
        if hilang:
            raise ValueError(f"Kolom masukan tidak lengkap, yang hilang: {hilang}")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        d = X.copy()

        # (1) max_power: teks -> angka
        d["max_power"] = pd.to_numeric(d["max_power"], errors="coerce")

        # (2) mileage: nol mustahil -> NaN
        d["mileage(km/ltr/kg)"] = pd.to_numeric(
            d["mileage(km/ltr/kg)"], errors="coerce"
        ).replace(0, np.nan)

        # (3) year -> umur
        d["umur"] = TAHUN_ACUAN - pd.to_numeric(d["year"], errors="coerce")

        # (4) km_driven -> log1p
        km = pd.to_numeric(d["km_driven"], errors="coerce").clip(lower=0)
        d["log_km_driven"] = np.log1p(km)

        # (5) name -> merek
        d["merek"] = (
            d["name"].astype("string").str.strip().str.split().str[0].str.title()
        )

        # (6) buang kolom yang sudah digantikan
        return d[FITUR_NUMERIK + FITUR_KATEGORIKAL]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FITUR_NUMERIK + FITUR_KATEGORIKAL, dtype=object)
