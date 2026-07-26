"""
conftest.py - Fixture bersama untuk seluruh berkas test.

Dijalankan otomatis oleh pytest sebelum test mana pun. Berkas ini juga
menambahkan akar repo ke sys.path sehingga `from app.main import app`
berhasil tanpa perlu memasang proyek sebagai paket.
"""

from __future__ import annotations
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app  # noqa: E402

@pytest.fixture(scope="session")
def klien():
    """Klien HTTP untuk menguji API.

    Dipakai sebagai context manager agar peristiwa lifespan benar-benar
    berjalan -- tanpa ini, model tidak pernah dimuat dan seluruh test
    prediksi akan gagal dengan 503.

    scope="session" berarti model hanya dimuat sekali untuk seluruh berkas
    test, bukan sekali per test.
    """
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="session", autouse=True)
def lewati_bila_model_tidak_ada():
    """Hentikan seluruh test dengan pesan jelas bila model belum dilatih.

    Tanpa ini, kegagalan akan muncul sebagai deretan 503 yang membingungkan
    dan menyamarkan penyebab sebenarnya.
    """
    if not (ROOT_DIR / "models" / "model.joblib").exists():
        pytest.skip(
            "models/model.joblib belum ada. Jalankan 'python src/train.py' "
            "lebih dulu.",
            allow_module_level=True,
        )

@pytest.fixture
def muatan_sah() -> dict:
    """Muatan permintaan yang valid, dipakai sebagai dasar oleh banyak test.

    Test yang perlu menguji satu medan cukup menyalin dict ini dan mengganti
    medan tersebut, sehingga yang berbeda antar-test hanya satu hal.
    """
    return {
        "name": "Maruti Swift VXI",
        "year": 2018,
        "km_driven": 30000,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "mileage_kmpl": 21.0,
        "engine_cc": 1197.0,
        "max_power_bhp": 82.0,
        "seats": 5,
    }

def harga(klien, muatan: dict) -> float:
    """Bantu: kirim permintaan dan kembalikan harga prediksinya.

    Menegaskan status 200 lebih dulu agar kegagalan tak terduga muncul
    sebagai pesan yang jelas, bukan sebagai KeyError di dalam test.
    """
    r = klien.post("/predict-harga", json=muatan)
    assert r.status_code == 200, f"Diharapkan 200, dapat {r.status_code}: {r.text}"
    return r.json()["harga_prediksi"]
