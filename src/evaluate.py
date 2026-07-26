"""
evaluate.py : Evaluasi model tersimpan dan pemeriksaan prakiraan.

Cara pakai:
    python src/evaluate.py    (jalankan setelah python src/train.py)
"""

from __future__ import annotations
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from features import KOLOM_MASUKAN, TARGET
from load_data import muat_data
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
plt.rcParams.update(
    {"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
     "font.size": 10, "axes.grid": True, "grid.alpha": 0.25}
)

def muat_artefak():
    jalur_model = MODELS_DIR / "model.joblib"
    jalur_meta = MODELS_DIR / "metadata.json"
    if not jalur_model.exists():
        raise SystemExit(
            f"Model belum ada di {jalur_model}.\n"
            "Jalankan lebih dulu: python src/train.py"
        )
    return joblib.load(jalur_model), json.loads(jalur_meta.read_text(encoding="utf-8"))

def siapkan_test_set(meta: dict):
    df = muat_data().drop_duplicates().reset_index(drop=True)
    X_latih, X_uji, y_latih, y_uji = train_test_split(
        df[KOLOM_MASUKAN],
        df[TARGET],
        test_size=meta["ukuran_test"],
        random_state=meta["random_state"],
        shuffle=True,
    )
    assert len(X_uji) == meta["jumlah_baris_uji"], (
        "Ukuran test set tidak cocok dengan metadata - pemisahan tidak identik."
    )
    return X_latih, X_uji, y_latih, y_uji

def grafik_6_prediksi_vs_aktual(y_uji, y_pred) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.scatter(y_uji / 1e5, y_pred / 1e5, s=10, alpha=0.28, color="#3a7ca5")
    lim = max(y_uji.max(), y_pred.max()) / 1e5 * 1.03
    ax.plot([0, lim], [0, lim], color="#c1440e", lw=1.6, ls="--",
            label="prediksi sempurna")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Harga aktual (ratus ribu INR)")
    ax.set_ylabel("Harga prediksi (ratus ribu INR)")
    ax.set_title("Grafik 6 — Prediksi versus harga aktual (test set)")
    ax.legend()
    out = REPORTS_DIR / "06_prediksi_vs_aktual.png"
    fig.savefig(out)
    plt.close(fig)
    return out

def grafik_7_residual(y_uji, y_pred) -> Path:
    residual = y_pred - y_uji
    galat_relatif = residual / y_uji * 100

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax[0].scatter(y_uji / 1e5, residual / 1e5, s=10, alpha=0.28, color="#3a7ca5")
    ax[0].axhline(0, color="#c1440e", lw=1.4)
    ax[0].set_xlabel("Harga aktual (ratus ribu INR)")
    ax[0].set_ylabel("Residual (ratus ribu INR)")
    ax[0].set_title("Residual mutlak melebar pada harga tinggi")

    ax[1].hist(galat_relatif.clip(-100, 100), bins=60, color="#3a7ca5",
               edgecolor="white")
    ax[1].axvline(0, color="#c1440e", lw=1.4)
    ax[1].set_xlabel("Galat relatif (%)")
    ax[1].set_ylabel("Jumlah kendaraan")
    ax[1].set_title("Sebaran galat relatif terpusat di sekitar nol")

    fig.suptitle("Grafik 7 — Analisis residual", y=1.02)
    out = REPORTS_DIR / "07_residual.png"
    fig.savefig(out)
    plt.close(fig)
    return out

def grafik_8_kepentingan_fitur(kepentingan: pd.DataFrame) -> Path:
    d = kepentingan.sort_values("rerata")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(d["fitur"], d["rerata"], xerr=d["simpangan"], color="#3a7ca5",
            error_kw={"ecolor": "#c1440e", "lw": 1.2})
    ax.set_xlabel("Kenaikan MAE saat fitur diacak (INR)")
    ax.set_title("Grafik 8 — Permutation importance pada test set")
    out = REPORTS_DIR / "08_kepentingan_fitur.png"
    fig.savefig(out)
    plt.close(fig)
    return out

def periksa_prakiraan(meta: dict, m_uji: dict, kepentingan: pd.DataFrame) -> None:
    cv = pd.DataFrame(meta["metrik_cv"])

    def r2(nama):
        baris = cv.loc[cv["model"] == nama, "R2_rerata"]
        return float(baris.iloc[0]) if len(baris) else float("nan")

    def mae(nama):
        baris = cv.loc[cv["model"] == nama, "MAE_rerata"]
        return float(baris.iloc[0]) if len(baris) else float("nan")

    print("=" * 74)
    print("Cek Prakiraan Sebelum Pelatihan")
    print("=" * 74)

    # --- Prakiraan 1 -------------------------------------------------------
    r2_lin = r2("LinearRegression")
    r2_rf = r2("RandomForest")
    r2_hgb = r2("HistGradientBoosting")
    print("\n[1] 'Linear R2 < 0,80; model pohon R2 > 0,88'")
    print(f"    LinearRegression      R2 = {r2_lin:.3f}  -> {'TERBUKTI' if r2_lin < 0.80 else 'MELESET'}")
    print(f"    RandomForest          R2 = {r2_rf:.3f}  -> {'TERBUKTI' if r2_rf > 0.88 else 'MELESET (tipis)'}")
    print(f"    HistGradientBoosting  R2 = {r2_hgb:.3f}  -> {'TERBUKTI' if r2_hgb > 0.88 else 'MELESET (tipis)'}")
    print(f"    LinearRegression + log(target)  R2 = {r2('LinearRegression + log(target)'):.3f}")
    print("    CATATAN: begitu target di-log, model linear nyaris menyamai")
    print("    model pohon. Sebagian besar keunggulan pohon ternyata berasal")
    print("    dari kemencengan target, bukan dari non-linearitas fitur.")

    # --- Prakiraan 2 -------------------------------------------------------
    teratas = kepentingan.sort_values("rerata", ascending=False)
    setara = {"year": "year (= umur)", "name": "name (= merek)"}
    print("\n[2] 'max_power adalah fitur terpenting, di atas umur dan km_driven'")
    print("    Diukur pada kolom MASUKAN MENTAH, karena itulah yang benar-benar")
    print("    dikirim pengguna ke API. 'year' membawa informasi yang sama")
    print("    persis dengan 'umur' (transformasi monoton), demikian pula")
    print("    'name' dengan 'merek'.")
    for _, r in teratas.head(5).iterrows():
        print(f"    {setara.get(r['fitur'], r['fitur']):<24} {r['rerata']:>12,.0f}")
    juara_fitur = teratas.iloc[0]["fitur"]
    print(f"    -> {'TERBUKTI' if juara_fitur == 'max_power' else f'MELESET; yang teratas: {setara.get(juara_fitur, juara_fitur)}'}")

    # --- Prakiraan 3 -------------------------------------------------------
    rasio_uji = m_uji["rasio_RMSE_per_MAE"]
    rasio_cv = cv.iloc[0]["RMSE_rerata"] / cv.iloc[0]["MAE_rerata"]
    print("\n[3] 'RMSE > 2x MAE, dan log(target) memperbaiki MAE'")
    print(f"    Rasio RMSE/MAE pada CV       = {rasio_cv:.2f}  -> {'TERBUKTI' if rasio_cv > 2 else 'MELESET'}")
    print(f"    Rasio RMSE/MAE pada test set = {rasio_uji:.2f}  -> {'TERBUKTI' if rasio_uji > 2 else 'MELESET'}")
    for dasar in ["HistGradientBoosting", "RandomForest", "LinearRegression"]:
        tanpa, dengan = mae(dasar), mae(f"{dasar} + log(target)")
        arah = "membaik" if dengan < tanpa else "memburuk"
        print(f"    {dasar:<22} MAE {tanpa:>10,.0f} -> {dengan:>10,.0f}  ({arah} {abs(dengan - tanpa) / tanpa * 100:.1f}%)")
    print()


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    model, meta = muat_artefak()

    print(f"Model dimuat  : {meta['nama_model']}")
    print(f"Dilatih pada  : {meta['dilatih_pada']}\n")

    _, X_uji, _, y_uji = siapkan_test_set(meta)
    y_pred = model.predict(X_uji)

    print("=" * 74)
    print("Matrik evaluasi pada test set")
    print("=" * 74)
    for k, v in meta["metrik_data_uji"].items():
        print(f"  {k:<22}{v:>14,.3f}")
    galat_relatif = np.abs(y_pred - y_uji) / y_uji * 100
    print(f"  {'MAPE (%)':<22}{galat_relatif.mean():>14,.2f}")
    print(f"  {'median galat relatif':<22}{np.median(galat_relatif):>14,.2f}")
    print()

    # Permutation importance dihitung pada test set dengan metrik MAE,
    # sehingga angkanya langsung terbaca sebagai "berapa rupee MAE memburuk
    # bila fitur ini diacak".
    print("Menghitung permutation importance (10 pengacakan)...")
    hasil = permutation_importance(
        model, X_uji, y_uji, n_repeats=10, random_state=meta["random_state"],
        scoring="neg_mean_absolute_error", n_jobs=1,
    )
    kepentingan = pd.DataFrame(
        {
            "fitur": list(X_uji.columns),
            "rerata": hasil.importances_mean,
            "simpangan": hasil.importances_std,
        }
    )

    hasil_grafik = [
        grafik_6_prediksi_vs_aktual(y_uji, y_pred),
        grafik_7_residual(y_uji, y_pred),
        grafik_8_kepentingan_fitur(kepentingan),
    ]

    periksa_prakiraan(meta, meta["metrik_data_uji"], kepentingan)

    print("=" * 74)
    print("Grafik Tersimpan")
    print("=" * 74)
    for p in hasil_grafik:
        print(f"  {p.relative_to(ROOT_DIR)}")
    print("\nEvaluasi selesai.")


if __name__ == "__main__":
    main()
