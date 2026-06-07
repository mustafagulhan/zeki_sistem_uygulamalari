"""Değerlendirme metrikleri (AUC-PR, ROC-AUC, F1, eşik analizi, Brier)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def en_iyi_f1_esigi(y_true: np.ndarray, olasiliklar: np.ndarray) -> float:
    """precision_recall_curve üzerinden F1'i maksimize eden eşiği bulur."""
    precision, recall, esikler = precision_recall_curve(y_true, olasiliklar)
    # precision/recall dizileri esikler'den bir uzun; son elemanı at.
    p = precision[:-1]
    r = recall[:-1]
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
    if len(esikler) == 0:
        return 0.5
    return float(esikler[int(np.argmax(f1))])


def metrikleri_hesapla(
    y_true: np.ndarray, olasiliklar: np.ndarray, esik: float
) -> dict:
    """Verilen eşik için tüm metrikleri tek sözlükte döndürür."""
    y_pred = (olasiliklar >= esik).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "esik": float(esik),
        "auc_pr": float(average_precision_score(y_true, olasiliklar)),
        "roc_auc": float(roc_auc_score(y_true, olasiliklar)),
        # Brier skoru: olasılık kalibrasyonu kalitesi (düşük = daha iyi).
        "brier": float(brier_score_loss(y_true, olasiliklar)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
