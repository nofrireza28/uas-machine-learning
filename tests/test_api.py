"""
test_api.py - Test mekanis.
"""

from __future__ import annotations
import pytest
from conftest import harga

# ===========================================================================
# Endpoint dasar
# ===========================================================================
def test_akar_mengembalikan_informasi_layanan(klien):
    """GET / harus 200 dan memuat daftar endpoint."""
    r = klien.get("/")
    assert r.status_code == 200

    badan = r.json()
    assert "layanan" in badan
    assert "endpoint" in badan
    assert "POST /predict-harga" in badan["endpoint"]


def test_health_melaporkan_model_termuat(klien):
    """GET /health harus 200 dan menyatakan model siap.

    Ini test paling penting untuk pemantauan: bila endpoint ini melaporkan
    'sehat' padahal model gagal dimuat, seluruh sistem pemantauan menjadi
    tidak berguna.
    """
    r = klien.get("/health")
    assert r.status_code == 200

    badan = r.json()
    assert badan["status"] == "sehat"
    assert badan["model_termuat"] is True
    assert badan["nama_model"]

# ===========================================================================
# Jalur sukses
# ===========================================================================
def test_prediksi_sah_mengembalikan_skema_lengkap(klien, muatan_sah):
    """Permintaan valid harus 200 dengan seluruh medan jawaban terisi."""
    r = klien.post("/predict-harga", json=muatan_sah)
    assert r.status_code == 200

    badan = r.json()
    for medan in [
        "harga_prediksi",
        "mata_uang",
        "perkiraan_galat_mae",
        "umur_kendaraan",
        "nama_model",
        "id_permintaan",
    ]:
        assert medan in badan, f"Medan '{medan}' hilang dari jawaban"

    assert badan["harga_prediksi"] > 0, "Harga prediksi tidak boleh nol atau negatif"
    assert badan["mata_uang"] == "INR"
    # muatan_sah bertahun 2018, TAHUN_ACUAN pada model adalah 2020
    assert badan["umur_kendaraan"] == 2

def test_medan_spesifikasi_boleh_dikosongkan(klien, muatan_sah):
    """Empat medan opsional boleh null; imputer di Pipeline yang mengisinya.

    Menegaskan status 200 saja TIDAK cukup untuk test ini. Bila imputer
    diganti dengan pengisi nol -- kesalahan yang mudah terjadi dan tampak
    tidak berbahaya -- API tetap membalas 200 dengan harga positif, hanya
    saja nilainya jatuh ke sekitar setengah dari yang seharusnya karena
    model membaca kendaraan bermesin 0 CC dan bertenaga 0 bhp.

    Karena itu test ini membandingkan prediksi tanpa medan spesifikasi
    terhadap prediksi dengan medan lengkap. Imputasi median menghasilkan
    nilai yang berdekatan; pengisi nol menghasilkan selisih yang jauh.
    Rentang 0,7-1,5 kali dipilih longgar agar tidak rapuh terhadap pelatihan
    ulang, tetapi tetap jauh lebih ketat daripada sekadar "lebih besar dari
    nol".
    """
    minim = {
        k: v
        for k, v in muatan_sah.items()
        if k not in ("mileage_kmpl", "engine_cc", "max_power_bhp", "seats")
    }
    r = klien.post("/predict-harga", json=minim)
    assert r.status_code == 200

    h_tanpa_spek = r.json()["harga_prediksi"]
    h_lengkap = harga(klien, muatan_sah)
    rasio = h_tanpa_spek / h_lengkap

    assert 0.7 < rasio < 1.5, (
        f"Prediksi tanpa medan spesifikasi ({h_tanpa_spek:,.0f}) terlalu jauh "
        f"dari prediksi lengkap ({h_lengkap:,.0f}); rasio {rasio:.2f}. "
        "Periksa apakah imputer pada Pipeline masih memakai strategi median."
    )


def test_id_permintaan_berbeda_setiap_panggilan(klien, muatan_sah):
    """Setiap permintaan harus punya id sendiri agar log dapat ditelusuri."""
    id1 = klien.post("/predict-harga", json=muatan_sah).json()["id_permintaan"]
    id2 = klien.post("/predict-harga", json=muatan_sah).json()["id_permintaan"]
    assert id1 != id2

# ===========================================================================
# Validasi: seluruhnya harus 422, tidak boleh 500
# ===========================================================================
@pytest.mark.parametrize(
    "medan, nilai, jenis",
    [
        ("year", "dua ribu", "tipe"),
        ("km_driven", "banyak", "tipe"),
        ("seats", "lima", "tipe"),
        ("year", 1850, "rentang"),
        ("year", 2050, "rentang"),
        ("km_driven", -5, "rentang"),
        ("km_driven", 5_000_000, "rentang"),
        ("mileage_kmpl", 0, "rentang"),
        ("engine_cc", 50, "rentang"),
        ("max_power_bhp", 5000, "rentang"),
        ("seats", 99, "rentang"),
        ("fuel", "Solar", "enum"),
        ("fuel", "diesel", "enum"),
        ("transmission", "Matic", "enum"),
        ("owner", "Fifth Owner", "enum"),
        ("seller_type", "Showroom", "enum"),
        ("name", "   ", "isi"),
        ("name", "X", "isi"),
    ],
)
def test_masukan_tidak_valid_dibalas_422(klien, muatan_sah, medan, nilai, jenis):
    """Masukan tak valid harus 422, bukan 500.

    Perbedaannya penting: 422 berarti 'kesalahan ada pada permintaan Anda',
    sedangkan 500 berarti 'ada yang rusak di sisi kami'. Membalas 500 untuk
    masukan buruk menyesatkan pemanggil dan mengaburkan galat yang sebenarnya
    saat penelusuran insiden.
    """
    muatan = {**muatan_sah, medan: nilai}
    r = klien.post("/predict-harga", json=muatan)

    assert r.status_code == 422, (
        f"[{jenis}] {medan}={nilai!r} menghasilkan {r.status_code}, seharusnya 422"
    )
    # Pesan galat harus menunjuk medan yang bermasalah, bukan sekadar
    # menyatakan 'permintaan tidak valid'.
    assert medan in str(r.json()["detail"])

@pytest.mark.parametrize(
    "medan_hilang",
    ["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner"],
)
def test_medan_wajib_hilang_dibalas_422(klien, muatan_sah, medan_hilang):
    """Menghilangkan medan wajib harus 422."""
    muatan = {k: v for k, v in muatan_sah.items() if k != medan_hilang}
    r = klien.post("/predict-harga", json=muatan)
    assert r.status_code == 422
    assert medan_hilang in str(r.json()["detail"])


def test_json_rusak_dibalas_422(klien):
    """Badan permintaan yang bukan JSON valid harus 422, bukan 500."""
    r = klien.post(
        "/predict-harga",
        content="ini bukan json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422
