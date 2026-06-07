"""Skorlama servisi: artefaktları yükler, işlemi skorlar ve SHAP açıklaması üretir."""
from __future__ import annotations

import joblib
import pandas as pd

from src import explain, preprocess
from src.config import Ayarlar, ayarlari_yukle
from src.schemas import OZELLIK_ISIMLERI, Islem, TahminYaniti


class DolandiricilikServisi:
    """Eğitilmiş artefaktları sarmalayan skorlama servisi."""

    def __init__(self, ayarlar: Ayarlar) -> None:
        self.ayarlar = ayarlar
        self.model = joblib.load(ayarlar.model_yolu())
        self.scaler = joblib.load(ayarlar.scaler_yolu())
        self.explainer = joblib.load(ayarlar.explainer_yolu())
        self.esik = self._esik_yukle()

    def _esik_yukle(self) -> float:
        """Eğitimde seçilen eşiği metrics.json'dan okur; yoksa varsayılanı kullanır."""
        import json

        try:
            with open(self.ayarlar.metrik_yolu(), "r", encoding="utf-8") as f:
                return float(json.load(f).get("esik", self.ayarlar.esik.varsayilan))
        except (FileNotFoundError, ValueError, KeyError):
            return self.ayarlar.esik.varsayilan

    def _on_isle(self, islem: Islem) -> pd.DataFrame:
        """İşlemi tek satırlık, ölçeklenmiş DataFrame'e dönüştürür."""
        satir = pd.DataFrame([islem.ozellik_vektoru()], columns=OZELLIK_ISIMLERI)
        return preprocess.donustur(satir, self.scaler, self.ayarlar)

    def skorla(self, islem: Islem, esik: float | None = None) -> TahminYaniti:
        """Tek işlem için olasılık + karar + SHAP açıklaması döndürür."""
        kullanilan_esik = self.esik if esik is None else esik
        X = self._on_isle(islem)

        olasilik = float(self.model.predict_proba(X)[0, 1])
        karar = int(olasilik >= kullanilan_esik)
        katkilar = explain.ozellik_katkilari(
            self.explainer, X, self.ayarlar.api.top_shap_ozellik
        )

        return TahminYaniti(
            dolandiricilik_olasiligi=olasilik,
            karar=karar,
            esik=kullanilan_esik,
            en_etkili_ozellikler=katkilar,
        )


_servis: DolandiricilikServisi | None = None


def servisi_al(ayarlar: Ayarlar | None = None) -> DolandiricilikServisi:
    """Tekil (singleton) servis örneği döndürür."""
    global _servis
    if _servis is None:
        _servis = DolandiricilikServisi(ayarlar or ayarlari_yukle())
    return _servis
