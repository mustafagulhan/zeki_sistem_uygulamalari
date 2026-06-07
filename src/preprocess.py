"""Ön işleme: Time ve Amount kolonlarının RobustScaler ile ölçeklenmesi."""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import RobustScaler

from .config import Ayarlar


def scaler_egit(X_train: pd.DataFrame, ayarlar: Ayarlar) -> RobustScaler:
    """Ölçeklenecek kolonlar üzerinde RobustScaler eğitir."""
    kolonlar = ayarlar.onisleme.olceklenecek_kolonlar
    scaler = RobustScaler()
    scaler.fit(X_train[kolonlar])
    return scaler


def donustur(
    X: pd.DataFrame, scaler: RobustScaler, ayarlar: Ayarlar
) -> pd.DataFrame:
    """Eğitilmiş scaler ile kopya üzerinde dönüşüm uygular (orijinali bozmaz)."""
    kolonlar = ayarlar.onisleme.olceklenecek_kolonlar
    X_kopya = X.copy()
    X_kopya[kolonlar] = scaler.transform(X_kopya[kolonlar])
    return X_kopya
