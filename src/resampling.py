"""Dengesiz veri stratejileri: class_weight, SMOTE, undersample, none."""
from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from .config import Ayarlar


def pozitif_agirligi_hesapla(y_train: pd.Series) -> float:
    """scale_pos_weight = negatif sayısı / pozitif sayısı."""
    pozitif = int((y_train == 1).sum())
    negatif = int((y_train == 0).sum())
    if pozitif == 0:
        return 1.0
    return negatif / pozitif


def uygula(
    X_train: pd.DataFrame, y_train: pd.Series, ayarlar: Ayarlar
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Seçilen stratejiyi uygular; (X, y, scale_pos_weight) döndürür."""
    # SMOTE/undersample yalnızca eğitim verisine uygulanır.
    strateji = ayarlar.dengesizlik.strateji.lower()

    if strateji == "smote":
        smote = SMOTE(
            k_neighbors=ayarlar.dengesizlik.smote_komsu_k,
            random_state=ayarlar.veri.rastgele_tohum,
        )
        X_yeni, y_yeni = smote.fit_resample(X_train, y_train)
        return X_yeni, y_yeni, 1.0

    if strateji == "undersample":
        rus = RandomUnderSampler(random_state=ayarlar.veri.rastgele_tohum)
        X_yeni, y_yeni = rus.fit_resample(X_train, y_train)
        return X_yeni, y_yeni, 1.0

    if strateji == "class_weight":
        return X_train, y_train, pozitif_agirligi_hesapla(y_train)

    # "none"
    return X_train, y_train, 1.0
