"""FastAPI dolandırıcılık skorlama servisi (/health, /predict, /predict/batch)."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from src.config import ayarlari_yukle
from src.schemas import Islem, TahminYaniti, TopluTahminYaniti

from .service import DolandiricilikServisi, servisi_al

ayarlar = ayarlari_yukle()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Başlangıçta modeli yükle; artefakt yoksa anlaşılır hata ver."""
    try:
        servisi_al(ayarlar)
    except FileNotFoundError as e:
        # Model henüz eğitilmemiş olabilir; /health bunu raporlar.
        app.state.yukleme_hatasi = str(e)
    else:
        app.state.yukleme_hatasi = None
    yield


app = FastAPI(title=ayarlar.api.baslik, version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Servis ayakta mı, model yüklü mü?"""
    hata = getattr(app.state, "yukleme_hatasi", None)
    return {
        "durum": "ayakta",
        "model_yuklu": hata is None,
        "mesaj": hata or "Model yüklü.",
    }


def _servis() -> DolandiricilikServisi:
    """Model yüklü değilse 503 döndüren yardımcı."""
    try:
        return servisi_al(ayarlar)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/predict", response_model=TahminYaniti)
def predict(
    islem: Islem,
    esik: float | None = Query(default=None, ge=0.0, le=1.0, description="İsteğe bağlı karar eşiği"),
) -> TahminYaniti:
    """Tek bir işlemi skorlar."""
    return _servis().skorla(islem, esik)


@app.post("/predict/batch", response_model=TopluTahminYaniti)
def predict_batch(
    islemler: list[Islem],
    esik: float | None = Query(default=None, ge=0.0, le=1.0),
) -> TopluTahminYaniti:
    """Birden fazla işlemi tek istekte skorlar."""
    servis = _servis()
    return TopluTahminYaniti(sonuclar=[servis.skorla(i, esik) for i in islemler])
