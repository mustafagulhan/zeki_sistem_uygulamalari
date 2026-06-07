"""Ön işleme ve şema birim testleri (model gerektirmez)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import preprocess
from src.config import Ayarlar, ApiAyari, CiktiAyari, DengesizlikAyari, EsikAyari, ModelAyari, OnislemeAyari, VeriAyari
from src.schemas import OZELLIK_ISIMLERI, Islem


def _ayarlar() -> Ayarlar:
    return Ayarlar(
        veri=VeriAyari(csv_yolu="data/creditcard.csv"),
        onisleme=OnislemeAyari(),
        dengesizlik=DengesizlikAyari(),
        model=ModelAyari(),
        esik=EsikAyari(),
        cikti=CiktiAyari(),
        api=ApiAyari(),
    )


def _ornek_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    veri = {ad: rng.normal(size=n) for ad in OZELLIK_ISIMLERI}
    veri["Time"] = rng.uniform(0, 1000, size=n)
    veri["Amount"] = rng.exponential(50, size=n)
    return pd.DataFrame(veri)[OZELLIK_ISIMLERI]


def test_scaler_yalnizca_hedef_kolonlari_degistirir():
    ayarlar = _ayarlar()
    df = _ornek_df()
    scaler = preprocess.scaler_egit(df, ayarlar)
    donusmus = preprocess.donustur(df, scaler, ayarlar)

    # Ölçeklenmeyen kolonlar (örn. V1) aynı kalmalı.
    assert np.allclose(df["V1"].to_numpy(), donusmus["V1"].to_numpy())
    # Ölçeklenen kolonlar (Amount) değişmeli.
    assert not np.allclose(df["Amount"].to_numpy(), donusmus["Amount"].to_numpy())


def test_donustur_orijinali_bozmaz():
    ayarlar = _ayarlar()
    df = _ornek_df()
    orijinal_amount = df["Amount"].copy()
    scaler = preprocess.scaler_egit(df, ayarlar)
    _ = preprocess.donustur(df, scaler, ayarlar)
    assert np.allclose(df["Amount"].to_numpy(), orijinal_amount.to_numpy())


def test_islem_ozellik_vektoru_sirasi():
    degerler = {ad: float(i) for i, ad in enumerate(OZELLIK_ISIMLERI)}
    islem = Islem(**degerler)
    vektor = islem.ozellik_vektoru()
    assert len(vektor) == len(OZELLIK_ISIMLERI)
    assert vektor == [float(i) for i in range(len(OZELLIK_ISIMLERI))]
