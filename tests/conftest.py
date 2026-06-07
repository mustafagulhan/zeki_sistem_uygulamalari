"""Test fixture'ları: sentetik veri üretir ve geçici bir model eğitir."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import train
from src.config import (
    ApiAyari,
    Ayarlar,
    CiktiAyari,
    DengesizlikAyari,
    EsikAyari,
    ModelAyari,
    OnislemeAyari,
    VeriAyari,
)
from src.schemas import OZELLIK_ISIMLERI


def sentetik_veri(n: int = 3000, tohum: int = 0) -> pd.DataFrame:
    """Dolandırıcılık sınıfı ayırt edilebilir, dengesiz sentetik veri üretir."""
    rng = np.random.default_rng(tohum)
    n_fraud = max(30, n // 50)  # ~%2 dolandırıcılık
    n_normal = n - n_fraud

    def blok(adet: int, kayma: float) -> np.ndarray:
        # V1..V28 normal dağılım; dolandırıcılıkta hafif kaydırılmış (öğrenilebilir sinyal).
        return rng.normal(loc=kayma, scale=1.0, size=(adet, 28))

    normal = blok(n_normal, 0.0)
    fraud = blok(n_fraud, 1.5)
    V = np.vstack([normal, fraud])

    time_kol = rng.uniform(0, 172000, size=n).reshape(-1, 1)
    amount = np.concatenate([
        rng.exponential(50, n_normal),
        rng.exponential(200, n_fraud),
    ]).reshape(-1, 1)
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)]).astype(int)

    df = pd.DataFrame(
        np.hstack([time_kol, V, amount]),
        columns=OZELLIK_ISIMLERI,
    )
    df["Class"] = y
    return df.sample(frac=1.0, random_state=tohum).reset_index(drop=True)


@pytest.fixture(scope="session")
def egitilmis_ayarlar(tmp_path_factory) -> Ayarlar:
    """Sentetik veriyle küçük bir model eğitir ve geçici yollu Ayarlar döndürür."""
    dizin = tmp_path_factory.mktemp("fraud_test")
    csv_yolu = dizin / "creditcard.csv"
    sentetik_veri().to_csv(csv_yolu, index=False)

    ayarlar = Ayarlar(
        veri=VeriAyari(
            csv_yolu=str(csv_yolu),  # mutlak yol
            test_orani=0.2,
            zaman_sirali_bolme=False,  # sentetik veride stratified daha stabil
        ),
        onisleme=OnislemeAyari(),
        dengesizlik=DengesizlikAyari(strateji="class_weight"),
        model=ModelAyari(n_estimators=40, max_depth=4),  # hızlı eğitim
        esik=EsikAyari(),
        cikti=CiktiAyari(model_dizini=str(dizin)),  # mutlak yol
        api=ApiAyari(),
    )
    train.calistir(ayarlar)
    return ayarlar
