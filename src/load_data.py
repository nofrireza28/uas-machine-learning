"""
load_data.py : Load dan pemeriksaan awal dataset.

Cara pakai:
    python src/load_data.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_FILE = RAW_DIR / "cardekho.csv"

DATASET_URL = (
    "https://www.kaggle.com/datasets/"
    "sukhmandeepsinghbrar/car-price-prediction-dataset"
)

PETUNJUK_UNDUH = f"""
File dataset tidak ditemukan di: {RAW_FILE}

Dataset tidak disertakan di dalam repo karena data/ masuk .gitignore. Silakan siapkan datanya dengan salah satu cara berikut.

CARA A - Kaggle CLI (otomatis)
    pip install kaggle
    # letakkan kaggle.json di ~/.kaggle/ (Windows: %USERPROFILE%\\.kaggle\\)
    kaggle datasets download -d sukhmandeepsinghbrar/car-price-prediction-dataset -p data/raw --unzip
    # ganti nama file hasil download menjadi cardekho.csv bila perlu

CARA B - Unduh manual
    1. Buka {DATASET_URL}
    2. Klik "Download", kemudian ekstrak
    3. Copy file CSV ke data/raw/ dan gati nama file (bila perlu) menjadi cardekho.csv

Setelah itu jalankan ulang: python src/load_data.py
"""

KOLOM_WAJIB = [
    "name",
    "year",
    "selling_price",
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


def cetak_versi_pustaka() -> None:
    import numpy as np
    import sklearn

    print("=" * 70)
    print("Versi Library")
    print("=" * 70)
    print(f"python       : {sys.version.split()[0]}")
    print(f"pandas       : {pd.__version__}")
    print(f"numpy        : {np.__version__}")
    print(f"scikit-learn : {sklearn.__version__}")
    try:
        import joblib

        print(f"joblib       : {joblib.__version__}")
    except ImportError:
        print("joblib       : belum terpasang")
    try:
        import fastapi

        print(f"fastapi      : {fastapi.__version__}")
    except ImportError:
        print("fastapi      : belum terpasang")
    print()


def muat_data(path: Path = RAW_FILE) -> pd.DataFrame:
    if not path.exists():
        print(PETUNJUK_UNDUH, file=sys.stderr)
        raise SystemExit(1)
    df = pd.read_csv(path, low_memory=False)
    return df


def periksa_kolom(df: pd.DataFrame) -> None:
    hilang = [k for k in KOLOM_WAJIB if k not in df.columns]
    if hilang:
        print(
            f"PERINGATAN: kolom berikut tidak ditemukan: {hilang}\n"
            "Kemungkinan Anda mengunduh varian dataset yang berbeda.",
            file=sys.stderr,
        )


def profil_dataset(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("Informasi Dataset")
    print("=" * 70)
    print(f"Sumber   : {RAW_FILE.relative_to(ROOT_DIR)}")
    print(f"URL      : {DATASET_URL}")
    print(f"Lisensi  : CC0 1.0 Universal (Public Domain Dedication)")
    print()
    print(f"Jumlah baris  : {df.shape[0]:,}")
    print(f"Jumlah kolom  : {df.shape[1]:,}")
    print()

    print("-" * 70)
    print("Tipe Data dan Nilai Hilang Per Kolom")
    print("-" * 70)

    ringkasan = pd.DataFrame(
        {
            "tipe": df.dtypes.astype(str),
            "nilai_hilang": df.isna().sum(),
            "persen_hilang": (df.isna().mean() * 100).round(2),
            "nilai_unik": df.nunique(),
        }
    )
    ringkasan.index.name = "kolom"
    print(ringkasan.to_string())
    print()

    print("-" * 70)
    print("Contoh Data (5 Baris Pertama)")
    print("-" * 70)
    print(df.head().to_string())
    print()


def main() -> None:
    cetak_versi_pustaka()
    df = muat_data()
    periksa_kolom(df)
    profil_dataset(df)
    print("Load data selesai.")


if __name__ == "__main__":
    main()
