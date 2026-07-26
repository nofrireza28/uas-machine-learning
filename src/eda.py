"""
eda.py : Analisis Data Eksploratif.

Cara pakai:
    python src/eda.py
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from load_data import muat_data

ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"

TAHUN_ACUAN = 2020  # tahun terbaru pada dataset; dipakai untuk menghitung umur

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

def pemeriksaan_wajib(df: pd.DataFrame) -> None:
    print("=" * 74)
    print("Pemeriksaan Wajib")
    print("=" * 74)

    # --- Pemeriksaan 1: Missing Values ---------------------------------
    print("\n[1] Pemeriksaan nilai hilang")
    print(df.isna().sum())
    print()

    # --- Pemeriksaan 2: Ringkasan Statistik ---------------------------------
    print("\n[2] Ringkasan statistik")
    print(df.describe())

    # --- Pemeriksaan 3: Duplikasi data ---------------------------------
    print("\n[3] Pemeriksaan baris duplikat")
    print(f"    Jumlah baris duplikat: {df.duplicated().sum()}")
    print()

def laporkan_kekotoran(df: pd.DataFrame) -> None:
    print("=" * 74)
    print("Temuan Kekotoran Data")
    print("=" * 74)

    # --- Temuan 1: Duplikasi Data penuh ---------------------------------
    n_dup = int(df.duplicated().sum())
    print(f"\n[1] Duplikasi data Penuh: {n_dup:,} baris ({n_dup / len(df) * 100:.1f}%)")
    print("    Seluruh 12 kolom memiliki nilai yang sama. Kemungkinan besar satu iklan yang sama terhimpun berkali-kali saat pengumpulan data.")
    contoh = df[df.duplicated(keep=False)].sort_values(list(df.columns)).head(4)
    print()
    print(contoh[["name", "year", "selling_price", "km_driven"]].to_string())

    # --- Temuan 2: Kolom numerik bertipe teks --------------------------------
    print(f"\n[2] Kolom Numerik Bertipe Teks: 'max_power' bertipe {df['max_power'].dtype}")
    mp_num = pd.to_numeric(df["max_power"], errors="coerce")
    gagal = df["max_power"].notna() & mp_num.isna()
    print(f"    Nilai yang gagal dikonversi ke angka: {int(gagal.sum())}")
    if gagal.any():
        print(f"    Contoh nilai bermasalah: {[repr(v) for v in df.loc[gagal, 'max_power'].unique()[:5]]}")
    contoh2 = df.loc[gagal, ["name", "year", "max_power"]].head(4)
    print()
    print(contoh2[["name", "year", "max_power"]].to_string())

    # --- Temuan 3: Nilai hilang yang menyamar sebagai spasi -------------------
    spasi = df["max_power"].astype("string").str.strip().eq("")
    n_spasi = int(spasi.sum())
    print(f"\n[3] Nilai Hilang yang Menyamar Sebagai Spasi: {n_spasi} baris pada 'max_power'")
    print("    Perintah df.isna() tidak mendeteksi ini karena secara teknis isinya string")
    print("    berisi spasi, bukan NaN. Baru terlihat setelah .str.strip().")
    contoh3 = df.loc[spasi, ["name", "year", "max_power"]].head(4)
    print()
    print(contoh3[["name", "year", "max_power"]].to_string())


    # --- Temuan 4: Kolom dengan nilai Nol yang tidak logis------------------------
    n_mil0 = int((df["mileage(km/ltr/kg)"] == 0).sum())
    print(f"\n[4] Nilai tidak valid: {n_mil0} baris dengan mileage = 0 km/l")
    print("    Nilai 0 km/liter secara logis tidak masuk akal. Jika sebuah kendaraan memiliki efisiensi 0 km/liter,")
    print("    artinya kendaraan tersebut tidak dapat bergerak sama sekali dengan bahan bakar, sehingga nilai tersebut tidak mungkin menjadi data yang valid.")
    contoh4 = df.loc[df["mileage(km/ltr/kg)"] == 0, ["name", "year", "mileage(km/ltr/kg)"]].head(5)
    print()
    print(contoh4[["name", "year", "mileage(km/ltr/kg)"]].to_string())

    # --- Temuan 5: Kolom blank pada beberapa kolom sekaligus---------------------------------
    kol_spek = ["mileage(km/ltr/kg)", "engine", "seats"]
    serempak = int(df[kol_spek].isna().all(axis=1).sum())
    print(f"\n[5] Nilai tidak valid: {serempak} baris pada kolom {kol_spek} tidak memiliki data sama sekali. Hal ini menunjukkan bahwa data yang hilang tidak terjadi secara acak")
    print("    melainkan karena informasi spesifikasi teknis kendaraan pada baris-baris tersebut memang tidak tersedia")
    contoh5 = df.loc[df[kol_spek].isna().all(axis=1), ["name", "year"] + kol_spek].head(5)
    print()
    print(contoh5[["name", "year"] + kol_spek].to_string())


    # --- Temuan 6: Outlier pada kolom jarak (km_driven)--------------------------------
    print("\n[6] Nilai tidak valid :")
    top = df.nlargest(3, "km_driven")[["name", "year", "km_driven", "selling_price"]]
    print(top.to_string())
    print("    Kendaraan penumpang tahun 2007 dengan 2.360.457 km berarti menempuh lebih kurang 180.000 km per tahun selama 13 tahun")
    print("    Ini bisa jadi kesalahan pencatatan, bukan kondisi kendaraan sebenarnya")
    contoh6 = df.loc[df["km_driven"] > 2_000_000, ["name", "year", "km_driven", "selling_price"]].head(5)
    print()
    print(contoh6[["name", "year", "km_driven", "selling_price"]].to_string())

    # --- Temuan 7: Jumlah Kategori berlebihan --------------------------------
    n_nama = df["name"].nunique()
    n_merek = df["name"].str.split().str[0].nunique()
    print(f"\n[7] Jumlah Kategori Berlebihan: Atribut 'name' Memiliki {n_nama:,} nilai unik dari {len(df):,} baris")
    print(f"    Pada proses One-hot encoding akan menghasilkan {n_nama:,} kolom baru yang hampir seluruhnya bernilai nol")
    print(f"    Kata pertama pada setiap nama tersebut adalah merek, dan hanya ada {n_merek} merek -> jauh lebih berguna.")
    print()

def bersihkan_untuk_visualisasi(df: pd.DataFrame) -> pd.DataFrame:
    """Pembersihan minimal agar informasi pada grafik tidak menyesatkan.
    """
    d = df.copy()
    d = d.drop_duplicates()
    d["max_power"] = pd.to_numeric(d["max_power"], errors="coerce")
    d["mileage(km/ltr/kg)"] = d["mileage(km/ltr/kg)"].replace(0, np.nan)
    d["umur"] = TAHUN_ACUAN - d["year"]
    d["merek"] = d["name"].str.split().str[0]
    return d

def grafik_1_sebaran_target(d: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].hist(d["selling_price"] / 1e5, bins=60, color="#3a7ca5", edgecolor="white")
    ax[0].set_xlabel("Harga jual (ratus ribu INR)")
    ax[0].set_ylabel("Jumlah kendaraan")
    ax[0].set_title("Skala asli")
    ax[0].axvline(d["selling_price"].median() / 1e5, color="#c1440e", ls="--", lw=1.5,
                  label=f"median = {d['selling_price'].median():,.0f}")
    ax[0].axvline(d["selling_price"].mean() / 1e5, color="#2a9d8f", ls="--", lw=1.5,
                  label=f"rerata = {d['selling_price'].mean():,.0f}")
    ax[0].legend(fontsize=8)

    ax[1].hist(np.log10(d["selling_price"]), bins=60, color="#3a7ca5", edgecolor="white")
    ax[1].set_xlabel("log10(harga jual)")
    ax[1].set_ylabel("Jumlah kendaraan")
    ax[1].set_title("Skala logaritmik")

    fig.suptitle("Grafik 1 - Sebaran harga jual sangat menceng ke kanan", y=1.02)
    out = REPORTS_DIR / "01_sebaran_target.png"
    fig.savefig(out)
    plt.close(fig)
    return out

def grafik_2_umur_vs_harga(d: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(d["umur"], d["selling_price"] / 1e5, s=7, alpha=0.18, color="#3a7ca5")

    med = d.groupby("umur")["selling_price"].median() / 1e5
    med = med[med.index <= 25]
    ax.plot(med.index, med.values, color="#c1440e", lw=2.2, marker="o", ms=4,
            label="median harga per umur")

    ax.set_xlim(-0.5, 25)
    ax.set_ylim(0, 30)
    ax.set_xlabel(f"Umur kendaraan (tahun, acuan {TAHUN_ACUAN})")
    ax.set_ylabel("Harga jual (ratus ribu INR)")
    ax.set_title("Grafik 2 — Depresiasi terhadap umur bersifat non-linear")
    ax.legend()
    out = REPORTS_DIR / "02_umur_vs_harga.png"
    fig.savefig(out)
    plt.close(fig)
    return out

def grafik_3_kategori_vs_harga(d: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    kolom = [("transmission", "Transmisi"), ("fuel", "Bahan bakar"), ("owner", "Kepemilikan")]

    for ax, (kol, judul) in zip(axes, kolom):
        urut = d.groupby(kol)["selling_price"].median().sort_values().index
        data = [d.loc[d[kol] == k, "selling_price"].values / 1e5 for k in urut]
        bp = ax.boxplot(data, tick_labels=list(urut), showfliers=False, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#a8dadc")
        ax.set_title(judul)
        ax.set_ylabel("Harga (ratus ribu INR)" if kol == "transmission" else "")
        ax.tick_params(axis="x", rotation=35, labelsize=8)

    fig.suptitle("Grafik 3 — Perbedaan harga antar kategori", y=1.02)
    out = REPORTS_DIR / "03_kategori_vs_harga.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def grafik_4_korelasi(d: pd.DataFrame) -> Path:
    kol = ["selling_price", "umur", "km_driven", "mileage(km/ltr/kg)", "engine",
           "max_power", "seats"]
    corr = d[kol].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(kol)), kol, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(kol)), kol, fontsize=8)
    for i in range(len(kol)):
        for j in range(len(kol)):
            nilai = corr.iloc[i, j]
            ax.text(j, i, f"{nilai:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(nilai) > 0.55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Grafik 4 — Korelasi Pearson antar variabel numerik")
    ax.grid(False)
    out = REPORTS_DIR / "04_korelasi.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def grafik_5_outlier_km(d: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].scatter(d["km_driven"] / 1000, d["selling_price"] / 1e5, s=7, alpha=0.2,
                  color="#3a7ca5")
    ax[0].set_xlabel("Jarak tempuh (ribu km)")
    ax[0].set_ylabel("Harga (ratus ribu INR)")
    ax[0].set_title("Skala asli — beberapa titik menekan seluruh sebaran")

    ax[1].scatter(d["km_driven"], d["selling_price"], s=7, alpha=0.2, color="#3a7ca5")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Jarak tempuh (km, skala log)")
    ax[1].set_ylabel("Harga (INR, skala log)")
    ax[1].set_title("Skala log-log — hubungan menurun jadi terlihat")

    fig.suptitle("Grafik 5 — Outlier jarak tempuh menyembunyikan pola sebenarnya",
                 y=1.02)
    out = REPORTS_DIR / "05_outlier_km.png"
    fig.savefig(out)
    plt.close(fig)
    return out

def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    df = muat_data()
    pemeriksaan_wajib(df)
    laporkan_kekotoran(df)

    d = bersihkan_untuk_visualisasi(df)
    print(f"Baris setelah pembersihan untuk visualisasi: {len(d):,} "
          f"(dari {len(df):,})\n")

    hasil = [
        grafik_1_sebaran_target(d),
        grafik_2_umur_vs_harga(d),
        grafik_3_kategori_vs_harga(d),
        grafik_4_korelasi(d),
        grafik_5_outlier_km(d),
    ]

    print("=" * 74)
    print("Grafik yang dihasilkan:")
    print("=" * 74)
    for p in hasil:
        print(f"  {p.relative_to(ROOT_DIR)}")
    print("\nEDA selesai.")


if __name__ == "__main__":
    main()
