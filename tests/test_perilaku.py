"""
test_perilaku.py - Behavioral test.

"""

from __future__ import annotations
from conftest import harga

def test_kendaraan_lebih_tua_diprediksi_lebih_murah(klien, muatan_sah):
    """Depresiasi: makin tua kendaraan, makin murah harganya.

    Ini hubungan paling mendasar pada domain kendaraan bekas. Bila test ini
    gagal, ada yang salah secara fundamental. Kemungkinan besar tanda pada
    perhitungan `umur = TAHUN_ACUAN - year` terbalik.

    Diuji sebagai deret menurun, bukan sekadar dua titik, agar pembalikan
    yang hanya terjadi di sebagian rentang juga tertangkap.
    """
    tahun = [2018, 2015, 2012, 2008, 2004]
    harga_per_tahun = [harga(klien, {**muatan_sah, "year": th}) for th in tahun]

    for lebih_baru, lebih_tua, h_baru, h_tua in zip(
        tahun, tahun[1:], harga_per_tahun, harga_per_tahun[1:]
    ):
        assert h_baru > h_tua, (
            f"Kendaraan {lebih_baru} ({h_baru:,.0f}) seharusnya lebih mahal "
            f"daripada {lebih_tua} ({h_tua:,.0f})"
        )

    # Selisih ujung ke ujung harus nyata, bukan sekadar beda beberapa rupee.
    assert harga_per_tahun[0] > harga_per_tahun[-1] * 2, (
        "Kendaraan 14 tahun lebih tua seharusnya jauh lebih murah, "
        f"tetapi {harga_per_tahun[0]:,.0f} vs {harga_per_tahun[-1]:,.0f}"
    )


def test_jarak_tempuh_lebih_tinggi_diprediksi_lebih_murah(klien, muatan_sah):
    """Makin jauh kendaraan sudah menempuh, makin murah harganya.

    Hubungan ini lebih lemah daripada umur (korelasi Pearson hanya -0,17 pada
    EDA), tetapi arahnya harus tetap konsisten.
    """
    jarak = [10_000, 50_000, 100_000, 200_000]
    harga_per_jarak = [harga(klien, {**muatan_sah, "km_driven": km}) for km in jarak]

    for km_rendah, km_tinggi, h_rendah, h_tinggi in zip(
        jarak, jarak[1:], harga_per_jarak, harga_per_jarak[1:]
    ):
        assert h_rendah > h_tinggi, (
            f"Kendaraan dengan {km_rendah:,} km ({h_rendah:,.0f}) seharusnya "
            f"lebih mahal daripada {km_tinggi:,} km ({h_tinggi:,.0f})"
        )


def test_tenaga_mesin_lebih_besar_diprediksi_lebih_mahal(klien, muatan_sah):
    """Tenaga mesin menandai segmen kendaraan, dan segmen menentukan harga.

    `max_power` adalah fitur terpenting kedua menurut permutation importance,
    jadi pengaruhnya harus terlihat jelas.
    """
    h_kecil = harga(klien, {**muatan_sah, "max_power_bhp": 60.0})
    h_besar = harga(klien, {**muatan_sah, "max_power_bhp": 250.0})

    assert h_besar > h_kecil, (
        f"Kendaraan 250 bhp ({h_besar:,.0f}) seharusnya lebih mahal daripada "
        f"60 bhp ({h_kecil:,.0f})"
    )


def test_transmisi_otomatis_diprediksi_lebih_mahal(klien, muatan_sah):
    """EDA menunjukkan median otomatis 850.000 versus manual 385.000.

    Test ini memastikan sinyal kategorikal benar-benar sampai ke model --
    bila OneHotEncoder salah memetakan kolom, perbedaan ini akan hilang.
    """
    h_manual = harga(klien, {**muatan_sah, "transmission": "Manual"})
    h_otomatis = harga(klien, {**muatan_sah, "transmission": "Automatic"})

    assert h_otomatis > h_manual, (
        f"Transmisi otomatis ({h_otomatis:,.0f}) seharusnya lebih mahal "
        f"daripada manual ({h_manual:,.0f})"
    )


def test_prediksi_berada_pada_rentang_yang_masuk_akal(klien, muatan_sah):
    """Prediksi harus berada dalam rentang harga yang ada pada data latih.

    Menangkap kegagalan transformasi target: bila `inverse_func=np.exp` lupa
    dipasang, model akan mengembalikan nilai log (belasan) alih-alih rupee
    (ratusan ribu). Angka semacam itu tetap positif dan tetap membalas 200,
    sehingga hanya test seperti inilah yang dapat menangkapnya.

    Batas 20.000 dan 15.000.000 diambil dari rentang harga pada dataset,
    sengaja dilonggarkan agar tidak rapuh terhadap pelatihan ulang.
    """
    h = harga(klien, muatan_sah)
    assert 20_000 < h < 15_000_000, (
        f"Prediksi {h:,.0f} berada di luar rentang harga yang wajar. "
        "Periksa apakah transformasi balik target terpasang."
    )


def test_prediksi_konsisten_untuk_masukan_yang_sama(klien, muatan_sah):
    """Masukan identik harus menghasilkan prediksi identik.

    Model ini deterministik. Bila dua panggilan memberi hasil berbeda, ada
    keadaan yang bocor antar-permintaan -- misalnya karena model dimuat ulang
    per permintaan atau ada praproses yang menyimpan keadaan.
    """
    h1 = harga(klien, muatan_sah)
    h2 = harga(klien, muatan_sah)
    assert h1 == h2, f"Prediksi tidak konsisten: {h1:,.2f} lalu {h2:,.2f}"
