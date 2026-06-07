"""Veri yükleme ve eğitim/test bölme (zaman-sıralı veya stratified)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Ayarlar


def veri_oku(csv_yolu: str | Path) -> pd.DataFrame:
    """CSV'yi DataFrame olarak okur; dosya yoksa anlaşılır bir hata verir."""
    yol = Path(csv_yolu)
    if not yol.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı: {yol}\n"
            "Kaggle 'Credit Card Fraud Detection' veri setini indirip "
            "proje/data/creditcard.csv konumuna koyun (bkz. README)."
        )
    return pd.read_csv(yol)


def egitim_test_bol(
    df: pd.DataFrame, ayarlar: Ayarlar
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """X_train, X_test, y_train, y_test döndürür."""
    hedef = ayarlar.veri.hedef_kolon
    y = df[hedef]
    X = df.drop(columns=[hedef])

    if ayarlar.veri.zaman_sirali_bolme and "Time" in X.columns:
        sirali = df.sort_values("Time").reset_index(drop=True)
        n_test = int(len(sirali) * ayarlar.veri.test_orani)
        egitim = sirali.iloc[: len(sirali) - n_test]
        test = sirali.iloc[len(sirali) - n_test :]
        X_train = egitim.drop(columns=[hedef])
        y_train = egitim[hedef]
        X_test = test.drop(columns=[hedef])
        y_test = test[hedef]
        return X_train, X_test, y_train, y_test

    # Stratified rastgele bölme
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=ayarlar.veri.test_orani,
        random_state=ayarlar.veri.rastgele_tohum,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test
