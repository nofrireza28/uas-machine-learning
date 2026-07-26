"""
main.py - Tahap 4: REST API untuk estimasi harga kendaraan bekas.

Menjalankan:
    uvicorn app.main:app --reload
    (dijalankan dari akar repo, bukan dari dalam app/)

Dokumentasi interaktif otomatis tersedia di http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Penyiapan jalur impor.
#
# Berkas model.joblib menyimpan objek PembersihFitur yang didefinisikan di
# src/features.py. Saat joblib memuatnya kembali, Python harus dapat mengimpor
# modul bernama 'features'. Tanpa baris di bawah, joblib.load gagal dengan
# ModuleNotFoundError meski berkas modelnya ada dan utuh.
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "prediksi.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("api.harga-kendaraan")

# ---------------------------------------------------------------------------
# Keadaan aplikasi. Diisi satu kali saat startup, bukan setiap permintaan.
# ---------------------------------------------------------------------------
keadaan: dict[str, object] = {"model": None, "metadata": None, "galat_muat": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Muat model SEKALI saat startup.

    Memuat model di dalam handler endpoint akan membaca ulang berkas dari
    disk dan membangun ulang objek pada SETIAP permintaan -- lambat, boros
    memori, dan membuat waktu tanggap tidak dapat diprediksi. Dengan lifespan,
    biaya itu dibayar satu kali; bila pemuatan gagal, kegagalannya diketahui
    saat startup, bukan saat pengguna pertama datang.
    """
    jalur_model = MODELS_DIR / "model.joblib"
    jalur_meta = MODELS_DIR / "metadata.json"

    try:
        t0 = time.perf_counter()
        keadaan["model"] = joblib.load(jalur_model)
        keadaan["metadata"] = json.loads(jalur_meta.read_text(encoding="utf-8"))
        durasi = (time.perf_counter() - t0) * 1000
        log.info(
            "Model dimuat: %s (%.0f ms)",
            keadaan["metadata"].get("nama_model", "?"),
            durasi,
        )
    except FileNotFoundError as e:
        keadaan["galat_muat"] = f"Berkas tidak ditemukan: {e.filename}"
        log.error(
            "Gagal memuat model: %s. Jalankan 'python src/train.py' lebih dulu.",
            keadaan["galat_muat"],
        )
    except Exception as e:  # noqa: BLE001 - sengaja luas; startup tidak boleh mati
        keadaan["galat_muat"] = f"{type(e).__name__}: {e}"
        log.error("Gagal memuat model: %s", keadaan["galat_muat"])

    yield

    keadaan["model"] = None
    log.info("Aplikasi dihentikan, model dilepas dari memori.")


app = FastAPI(
    title="API Estimasi Harga Kendaraan Bekas",
    description=(
        "Memprediksi harga jual kendaraan bekas dari karakteristik kendaraan. "
        "Proyek UAS Machine Learning — Kasus B (Regresi)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Skema masukan
# ---------------------------------------------------------------------------
# Batas tahun mengikuti rentang data latih. Kendaraan di luar rentang ini
# berarti ekstrapolasi, dan model tidak punya dasar untuk menebaknya.
TAHUN_MIN = 1983
TAHUN_MAKS = 2020


class PermintaanPrediksi(BaseModel):
    """Karakteristik kendaraan yang akan diprediksi harganya."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Maruti Swift Dzire VDI",
                    "year": 2014,
                    "km_driven": 145500,
                    "fuel": "Diesel",
                    "seller_type": "Individual",
                    "transmission": "Manual",
                    "owner": "First Owner",
                    "mileage_kmpl": 23.4,
                    "engine_cc": 1248.0,
                    "max_power_bhp": 74.0,
                    "seats": 5,
                }
            ]
        }
    }

    name: Annotated[
        str,
        Field(
            min_length=2,
            max_length=120,
            description="Nama kendaraan. Kata pertama dibaca sebagai merek.",
        ),
    ]
    year: Annotated[
        int,
        Field(
            ge=TAHUN_MIN,
            le=TAHUN_MAKS,
            description=(
                f"Tahun pembuatan. Dibatasi {TAHUN_MIN}-{TAHUN_MAKS} mengikuti "
                "rentang data latih."
            ),
        ),
    ]
    km_driven: Annotated[
        int, Field(ge=0, le=1_000_000, description="Jarak tempuh dalam kilometer.")
    ]

    # Enum: nilai di luar daftar ini ditolak dengan 422, bukan diteruskan ke
    # model dan menghasilkan prediksi yang diam-diam salah.
    fuel: Literal["Diesel", "Petrol", "CNG", "LPG"]
    seller_type: Literal["Individual", "Dealer", "Trustmark Dealer"]
    transmission: Literal["Manual", "Automatic"]
    owner: Literal[
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car",
    ]

    # Empat field berikut boleh null. Pipeline sudah memiliki imputer yang
    # dilatih pada data latih, sehingga nilai kosong ditangani dengan median
    # yang sah -- bukan dengan angka nol yang akan menyesatkan model.
    mileage_kmpl: Annotated[
        float | None,
        Field(default=None, gt=0, le=50, description="Konsumsi bahan bakar (km/l)."),
    ]
    engine_cc: Annotated[
        float | None,
        Field(default=None, ge=500, le=10_000, description="Kapasitas mesin (CC)."),
    ]
    max_power_bhp: Annotated[
        float | None,
        Field(default=None, gt=0, le=1_000, description="Tenaga maksimum (bhp)."),
    ]
    seats: Annotated[
        int | None, Field(default=None, ge=2, le=14, description="Jumlah kursi.")
    ]

    @field_validator("name")
    @classmethod
    def nama_tidak_boleh_kosong(cls, v: str) -> str:
        """Tolak nama yang hanya berisi spasi.

        min_length saja tidak cukup: string '   ' panjangnya 3 karakter dan
        lolos pemeriksaan panjang, tetapi menghasilkan merek kosong saat
        dipecah. Ini persis jenis nilai yang ditemukan pada kolom max_power
        di dataset mentah.
        """
        if not v.strip():
            raise ValueError("name tidak boleh hanya berisi spasi")
        return v.strip()

    def ke_dataframe(self) -> pd.DataFrame:
        """Ubah menjadi DataFrame dengan nama kolom yang dikenali Pipeline."""
        return pd.DataFrame(
            [
                {
                    "name": self.name,
                    "year": self.year,
                    "km_driven": self.km_driven,
                    "fuel": self.fuel,
                    "seller_type": self.seller_type,
                    "transmission": self.transmission,
                    "owner": self.owner,
                    "mileage(km/ltr/kg)": self.mileage_kmpl,
                    "engine": self.engine_cc,
                    "max_power": self.max_power_bhp,
                    "seats": self.seats,
                }
            ]
        )


# ---------------------------------------------------------------------------
# Skema keluaran
# ---------------------------------------------------------------------------
class JawabanPrediksi(BaseModel):
    harga_prediksi: float = Field(description="Estimasi harga jual.")
    mata_uang: str = "INR"
    perkiraan_galat_mae: float = Field(
        description="MAE model pada test set. Prediksi biasanya meleset sebesar ini."
    )
    umur_kendaraan: int
    nama_model: str
    id_permintaan: str


class JawabanKesehatan(BaseModel):
    status: Literal["sehat", "tidak sehat"]
    model_termuat: bool
    nama_model: str | None = None
    dilatih_pada: str | None = None
    keterangan: str | None = None


# ---------------------------------------------------------------------------
# Dependency: pastikan model siap
# ---------------------------------------------------------------------------
def ambil_model():
    """Kembalikan model, atau lempar 503 bila belum termuat.

    Fungsi ini sengaja TIDAK dipasang sebagai Depends, melainkan dipanggil
    di dalam badan endpoint. Alasannya: FastAPI menyelesaikan dependency
    SEBELUM memvalidasi badan permintaan, sehingga bila dipasang sebagai
    Depends, permintaan dengan enum salah akan dibalas 503 padahal
    seharusnya 422. Permintaan yang cacat tetap cacat terlepas dari keadaan
    server, dan pemanggil berhak tahu kesalahannya ada di sisi mereka.

    503 Service Unavailable adalah kode yang tepat untuk keadaan ini:
    permintaannya sendiri valid, tetapi layanan sedang tidak dapat
    melayaninya. Ini berbeda dari 500, yang berarti ada galat tak terduga
    di dalam kode.
    """
    if keadaan["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "pesan": "Model belum termuat, layanan belum siap.",
                "penyebab": keadaan["galat_muat"],
                "tindakan": "Jalankan 'python src/train.py' lalu mulai ulang API.",
            },
        )
    return keadaan["model"]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.get("/", tags=["umum"])
def akar():
    """Informasi layanan dan daftar endpoint."""
    return {
        "layanan": "API Estimasi Harga Kendaraan Bekas",
        "versi": app.version,
        "kasus": "Kasus B — Regresi",
        "endpoint": {
            "GET /": "informasi layanan ini",
            "GET /health": "status kesehatan dan kesiapan model",
            "POST /predict-harga": "prediksi harga kendaraan",
            "GET /docs": "dokumentasi interaktif",
        },
    }


@app.get("/health", response_model=JawabanKesehatan, tags=["umum"])
def kesehatan():
    """Status kesiapan layanan.

    Mengembalikan 200 pada kedua keadaan -- baik model termuat maupun tidak.
    Endpoint ini melaporkan keadaan, bukan menghakiminya; pemantau eksternal
    membaca medan 'status' untuk mengambil keputusan.
    """
    if keadaan["model"] is None:
        return JawabanKesehatan(
            status="tidak sehat",
            model_termuat=False,
            keterangan=str(keadaan["galat_muat"] or "Model tidak tersedia."),
        )

    meta = keadaan["metadata"] or {}
    return JawabanKesehatan(
        status="sehat",
        model_termuat=True,
        nama_model=meta.get("nama_model"),
        dilatih_pada=meta.get("dilatih_pada"),
    )


@app.post(
    "/predict-harga",
    response_model=JawabanPrediksi,
    tags=["prediksi"],
    responses={
        422: {"description": "Masukan gagal validasi"},
        503: {"description": "Model belum termuat"},
    },
)
def prediksi_harga(permintaan: PermintaanPrediksi):
    """Prediksi harga jual sebuah kendaraan bekas.

    Urutan pemeriksaan: Pydantic memvalidasi badan permintaan lebih dulu
    (menghasilkan 422 bila gagal), baru kemudian kesiapan model diperiksa
    (menghasilkan 503 bila belum termuat).
    """
    model = ambil_model()
    id_permintaan = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    try:
        hasil = float(model.predict(permintaan.ke_dataframe())[0])
    except Exception as e:  # noqa: BLE001
        log.exception("[%s] Prediksi gagal", id_permintaan)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"pesan": "Prediksi gagal diproses.", "galat": type(e).__name__},
        ) from e

    durasi = (time.perf_counter() - t0) * 1000
    meta = keadaan["metadata"] or {}
    tahun_acuan = int(meta.get("tahun_acuan_umur", 2020))

    log.info(
        "[%s] prediksi=%.0f | %s %d | %s/%s | km=%d | %.1f ms",
        id_permintaan,
        hasil,
        permintaan.name,
        permintaan.year,
        permintaan.fuel,
        permintaan.transmission,
        permintaan.km_driven,
        durasi,
    )

    return JawabanPrediksi(
        harga_prediksi=round(hasil, 2),
        perkiraan_galat_mae=round(
            float(meta.get("metrik_data_uji", {}).get("MAE", 0.0)), 2
        ),
        umur_kendaraan=tahun_acuan - permintaan.year,
        nama_model=str(meta.get("nama_model", "tidak diketahui")),
        id_permintaan=id_permintaan,
    )


# ---------------------------------------------------------------------------
# Penanganan galat tak terduga
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def penangan_galat_umum(request: Request, exc: Exception):
    """Cegah jejak tumpukan bocor ke pengguna.

    Rincian galat dicatat ke log untuk penelusuran, sementara pemanggil hanya
    menerima pesan umum. Jejak tumpukan dapat memuat jalur berkas dan potongan
    kode yang tidak seharusnya terlihat dari luar.
    """
    log.exception("Galat tak tertangani pada %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": {"pesan": "Terjadi galat internal."}},
    )
