"""Uçtan uca servis ve API sözleşmesi testleri (sentetik model ile)."""
from __future__ import annotations

import pandas as pd

from api.service import DolandiricilikServisi
from src.schemas import OZELLIK_ISIMLERI, Islem, TahminYaniti


def _ornek_islem(ayarlar) -> Islem:
    """Eğitimde kaydedilen test örneğinden bir işlem alır."""
    ornek = pd.read_csv(ayarlar.metrik_yolu().parent / "test_ornegi.csv")
    satir = ornek.iloc[0]
    return Islem(**{ad: float(satir[ad]) for ad in OZELLIK_ISIMLERI})


def test_servis_skorla_sozlesmesi(egitilmis_ayarlar):
    servis = DolandiricilikServisi(egitilmis_ayarlar)
    yanit = servis.skorla(_ornek_islem(egitilmis_ayarlar))

    assert isinstance(yanit, TahminYaniti)
    assert 0.0 <= yanit.dolandiricilik_olasiligi <= 1.0
    assert yanit.karar in (0, 1)
    # SHAP açıklaması dolu ve istenen sayıda özellik içeriyor.
    assert len(yanit.en_etkili_ozellikler) == egitilmis_ayarlar.api.top_shap_ozellik
    assert all(k.ozellik in OZELLIK_ISIMLERI for k in yanit.en_etkili_ozellikler)


def test_esik_karari_etkiler(egitilmis_ayarlar):
    servis = DolandiricilikServisi(egitilmis_ayarlar)
    islem = _ornek_islem(egitilmis_ayarlar)

    # Eşik 0.0 -> her şey dolandırıcılık; eşik 1.0'ın üstü -> hiçbir şey.
    assert servis.skorla(islem, esik=0.0).karar == 1
    assert servis.skorla(islem, esik=1.01).karar == 0


def test_metrikler_makul(egitilmis_ayarlar):
    """Sentetik veride sinyal öğrenilebilir olduğundan AUC-PR rastgeleden iyi olmalı."""
    import json

    with open(egitilmis_ayarlar.metrik_yolu(), encoding="utf-8") as f:
        metrikler = json.load(f)
    assert metrikler["auc_pr"] > 0.3
    assert metrikler["roc_auc"] > 0.7
