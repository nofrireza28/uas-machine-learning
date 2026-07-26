"""
train.py : Pemisahan data, Pembangunan Pipeline, Perbandingan model.

Cara pakai:
    python src/train.py
"""

from __future__ import annotations
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import (
    FITUR_KATEGORIKAL,
    FITUR_NUMERIK,
    KOLOM_MASUKAN,
    TAHUN_ACUAN,
    TARGET,
    PembersihFitur,
)
from load_data import muat_data

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"

RANDOM_STATE = 42
UKURAN_TEST = 0.20
N_LIPATAN = 5

def buat_praproses() -> ColumnTransformer:
    jalur_numerik = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    jalur_kategorikal = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", jalur_numerik, FITUR_NUMERIK),
            ("kat", jalur_kategorikal, FITUR_KATEGORIKAL),
        ],
        remainder="drop",
    )

def buat_pipeline(estimator, transformasi_log_target: bool = True) -> Pipeline:
    if transformasi_log_target:
        estimator = TransformedTargetRegressor(
            regressor=estimator, func=np.log, inverse_func=np.exp
        )

    return Pipeline(
        steps=[
            ("fitur", PembersihFitur()),
            ("praproses", buat_praproses()),
            ("model", estimator),
        ]
    )

def daftar_kandidat() -> dict[str, object]:
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.06,
            random_state=RANDOM_STATE,
        ),
    }

def metrik(y_benar, y_prediksi) -> dict[str, float]:
    mae = mean_absolute_error(y_benar, y_prediksi)
    rmse = float(np.sqrt(mean_squared_error(y_benar, y_prediksi)))
    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": float(r2_score(y_benar, y_prediksi)),
        "rasio_RMSE_per_MAE": rmse / mae,
    }


def jalankan_cv(X_latih, y_latih) -> pd.DataFrame:
    """Bandingkan seluruh kandidat dengan 5-fold CV pada data latih saja."""
    kf = KFold(n_splits=N_LIPATAN, shuffle=True, random_state=RANDOM_STATE)
    skor = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error",
            "R2": "r2"}

    baris = []
    for nama, est in daftar_kandidat().items():
        for pakai_log in (False, True):
            label = f"{nama}{' + log(target)' if pakai_log else ''}"
            pipa = buat_pipeline(est, transformasi_log_target=pakai_log)

            t0 = time.perf_counter()
            hasil = cross_validate(
                pipa, X_latih, y_latih, cv=kf, scoring=skor, n_jobs=1
            )
            durasi = time.perf_counter() - t0

            baris.append(
                {
                    "model": label,
                    "MAE_rerata": -hasil["test_MAE"].mean(),
                    "MAE_simpangan": hasil["test_MAE"].std(),
                    "RMSE_rerata": -hasil["test_RMSE"].mean(),
                    "R2_rerata": hasil["test_R2"].mean(),
                    "R2_simpangan": hasil["test_R2"].std(),
                    "detik": durasi,
                }
            )
            print(
                f"  {label:<38} MAE {-hasil['test_MAE'].mean():>10,.0f}  "
                f"R2 {hasil['test_R2'].mean():>6.3f} "
                f"(±{hasil['test_R2'].std():.3f})  {durasi:>5.1f}s"
            )

    return pd.DataFrame(baris).sort_values("MAE_rerata").reset_index(drop=True)

def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    # --- 1. Muat -----------------------------------------------------------
    df = muat_data()
    print(f"Baris mentah                : {len(df):,}")

    # --- 2. Buang duplikat (sebelum split, dengan alasan) -------------------
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplikat penuh dibuang      : {n_dup:,}")
    print(f"Baris setelah deduplikasi   : {len(df):,}")

    X = df[KOLOM_MASUKAN]
    y = df[TARGET]

    # --- 3. Split SEBELUM preprocessing apa pun -----------------------------
    X_latih, X_uji, y_latih, y_uji = train_test_split(
        X, y, test_size=UKURAN_TEST, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"Data latih                  : {len(X_latih):,}")
    print(f"Data uji (disegel)          : {len(X_uji):,}\n")

    # --- 4-5. Perbandingan model dengan 5-fold CV ---------------------------
    print("=" * 74)
    print(f"Perbandingan Model - {N_LIPATAN}-fold CV pada DATA LATIH saja")
    print("=" * 74)
    tabel = jalankan_cv(X_latih, y_latih)

    print("\n" + "-" * 74)
    print("Peringkat (berdasarkan MAE rerata CV, makin kecil makin baik)")
    print("-" * 74)
    print(tabel.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    juara = tabel.iloc[0]["model"]
    print(f"\nModel terpilih: {juara}")

    # --- 6. Latih ulang juara pada seluruh data latih -----------------------
    nama_dasar = juara.replace(" + log(target)", "")
    pakai_log = "log(target)" in juara
    pipa_final = buat_pipeline(
        daftar_kandidat()[nama_dasar], transformasi_log_target=pakai_log
    )
    pipa_final.fit(X_latih, y_latih)

    # --- 7. Sentuh test set SATU KALI ---------------------------------------
    print("\n" + "=" * 74)
    print("Evaluasi Pada Test Set - dijalankan satu kali")
    print("=" * 74)
    m_uji = metrik(y_uji, pipa_final.predict(X_uji))
    m_latih = metrik(y_latih, pipa_final.predict(X_latih))

    print(f"{'Metrik':<22}{'Data latih':>16}{'Data uji':>16}")
    for k in ["MAE", "RMSE", "R2", "rasio_RMSE_per_MAE"]:
        f = ",.0f" if k in ("MAE", "RMSE") else ".3f"
        print(f"{k:<22}{m_latih[k]:>16{f}}{m_uji[k]:>16{f}}")

    selisih = m_latih["R2"] - m_uji["R2"]
    print(f"\nSelisih R2 latih-uji: {selisih:.3f}", end="  ")
    print("(indikasi overfitting kuat)" if selisih > 0.10 else "(dalam batas wajar)")

    # --- 8. Simpan artefak --------------------------------------------------
    jalur_model = MODELS_DIR / "model.joblib"
    joblib.dump(pipa_final, jalur_model)

    metadata = {
        "nama_model": juara,
        "estimator_dasar": nama_dasar,
        "target_ditransformasi_log": pakai_log,
        "target": TARGET,
        "satuan_target": "INR (Rupee India)",
        "tahun_acuan_umur": TAHUN_ACUAN,
        "kolom_masukan": KOLOM_MASUKAN,
        "fitur_numerik": FITUR_NUMERIK,
        "fitur_kategorikal": FITUR_KATEGORIKAL,
        "random_state": RANDOM_STATE,
        "ukuran_test": UKURAN_TEST,
        "n_lipatan_cv": N_LIPATAN,
        "jumlah_baris_mentah": int(len(df) + n_dup),
        "jumlah_duplikat_dibuang": n_dup,
        "jumlah_baris_latih": int(len(X_latih)),
        "jumlah_baris_uji": int(len(X_uji)),
        "metrik_cv": tabel.to_dict(orient="records"),
        "metrik_data_latih": m_latih,
        "metrik_data_uji": m_uji,
        "dilatih_pada": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versi": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    jalur_meta = MODELS_DIR / "metadata.json"
    jalur_meta.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nArtefak tersimpan:")
    print(f"  {jalur_model.relative_to(ROOT_DIR)}")
    print(f"  {jalur_meta.relative_to(ROOT_DIR)}")
    print("\nPelatihan selesai.")


if __name__ == "__main__":
    main()
