"""SHAP TreeExplainer ile açıklanabilirlik (XAI)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from .schemas import OzellikKatkisi


def explainer_olustur(model: XGBClassifier) -> shap.TreeExplainer:
    """Eğitilmiş modelden TreeExplainer üretir."""
    return shap.TreeExplainer(model)


def ozellik_katkilari(
    explainer: shap.TreeExplainer,
    X_satir: pd.DataFrame,
    top_n: int,
) -> list[OzellikKatkisi]:
    """Tek işlem için en etkili top_n özelliği döndürür (pozitif = dolandırıcılığa iter)."""
    shap_degerleri = explainer.shap_values(X_satir)
    # Bazı SHAP sürümleri liste döndürür; normalize et.
    if isinstance(shap_degerleri, list):
        shap_degerleri = shap_degerleri[-1]
    katki_vektoru = np.asarray(shap_degerleri).reshape(-1)

    satir = X_satir.iloc[0]
    katkilar = [
        OzellikKatkisi(
            ozellik=str(ad),
            deger=float(satir[ad]),
            shap_katki=float(katki),
        )
        for ad, katki in zip(X_satir.columns, katki_vektoru)
    ]
    katkilar.sort(key=lambda k: abs(k.shap_katki), reverse=True)
    return katkilar[:top_n]
