"""Makale için modelden grafik üretir. Kullanım: python -m scripts.rapor_gorselleri"""
from __future__ import annotations

import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # başsız (headless) ortam
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from src import data_loader, evaluate, preprocess, resampling, train
from src.config import PROJE_KOK, ayarlari_yukle

CIKTI = PROJE_KOK / "reports" / "figures"


def _kaydet(fig, ad: str) -> None:
    CIKTI.mkdir(parents=True, exist_ok=True)
    yol = CIKTI / ad
    fig.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   kaydedildi: {yol.relative_to(PROJE_KOK)}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ayarlar = ayarlari_yukle()
    print(">> Model ve veri yükleniyor...")
    model = joblib.load(ayarlar.model_yolu())
    scaler = joblib.load(ayarlar.scaler_yolu())
    explainer = joblib.load(ayarlar.explainer_yolu())

    df = data_loader.veri_oku(ayarlar.csv_mutlak_yolu())
    X_train, X_test, y_train, y_test = data_loader.egitim_test_bol(df, ayarlar)
    X_test_s = preprocess.donustur(X_test, scaler, ayarlar)
    y_test_arr = y_test.to_numpy()

    olasiliklar = model.predict_proba(X_test_s)[:, 1]
    esik = evaluate.en_iyi_f1_esigi(y_test_arr, olasiliklar)
    y_pred = (olasiliklar >= esik).astype(int)

    # 1) Confusion matrix
    print(">> Confusion matrix...")
    cm = confusion_matrix(y_test_arr, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Dolandırıcılık"]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False
    )
    ax.set_title(f"Karmaşıklık Matrisi (eşik={esik:.3f})")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    _kaydet(fig, "confusion_matrix.png")

    # 2) PR + ROC eğrileri
    print(">> PR / ROC eğrileri...")
    fig, eksenler = plt.subplots(1, 2, figsize=(11, 4.2))
    PrecisionRecallDisplay.from_predictions(y_test_arr, olasiliklar, ax=eksenler[0])
    eksenler[0].set_title("Precision-Recall Eğrisi")
    RocCurveDisplay.from_predictions(y_test_arr, olasiliklar, ax=eksenler[1])
    eksenler[1].set_title("ROC Eğrisi")
    _kaydet(fig, "pr_roc_egrileri.png")

    # 3) SHAP özet (örneklem ile hız)
    print(">> SHAP özet grafiği...")
    ornek = X_test_s.sample(min(2000, len(X_test_s)), random_state=42)
    shap_degerleri = explainer.shap_values(ornek)
    if isinstance(shap_degerleri, list):
        shap_degerleri = shap_degerleri[-1]
    plt.figure()
    shap.summary_plot(shap_degerleri, ornek, show=False, max_display=12)
    fig = plt.gcf()
    fig.suptitle("SHAP Özet: En Etkili Özellikler", y=1.02)
    _kaydet(fig, "shap_ozet.png")

    # 4) Dengesizlik stratejisi karşılaştırması
    print(">> Strateji karşılaştırması (3 model eğitiliyor)...")
    scaler2 = preprocess.scaler_egit(X_train, ayarlar)
    X_train_s = preprocess.donustur(X_train, scaler2, ayarlar)
    stratejiler = ["class_weight", "smote", "undersample"]
    auc_pr, f1s, recalls = [], [], []
    for s in stratejiler:
        ayarlar.dengesizlik.strateji = s
        X_d, y_d, spw = resampling.uygula(X_train_s, y_train, ayarlar)
        m = train.model_egit(X_d, y_d, spw, ayarlar)
        ol = m.predict_proba(X_test_s)[:, 1]
        e = evaluate.en_iyi_f1_esigi(y_test_arr, ol)
        met = evaluate.metrikleri_hesapla(y_test_arr, ol, e)
        auc_pr.append(met["auc_pr"])
        f1s.append(met["f1"])
        recalls.append(met["recall"])

    x = np.arange(len(stratejiler))
    g = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - g, auc_pr, g, label="AUC-PR")
    ax.bar(x, f1s, g, label="F1")
    ax.bar(x + g, recalls, g, label="Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(stratejiler)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor")
    ax.set_title("Dengesizlik Stratejilerinin Karşılaştırması")
    ax.legend()
    for i in range(len(stratejiler)):
        ax.text(x[i] - g, auc_pr[i] + 0.01, f"{auc_pr[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i], f1s[i] + 0.01, f"{f1s[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + g, recalls[i] + 0.01, f"{recalls[i]:.3f}", ha="center", fontsize=8)
    _kaydet(fig, "strateji_karsilastirma.png")

    # 5) Olasılık dağılımı + karar bölgesi güvenilirlik diyagramı
    ham_yolu = ayarlar.model_yolu().parent / "model_ham.pkl"
    if ham_yolu.exists():
        print(">> Olasılık dağılımı ve güvenilirlik diyagramı...")
        ham_model = joblib.load(ham_yolu)
        ham_ol = ham_model.predict_proba(X_test_s)[:, 1]
        kal_ol = model.predict_proba(X_test_s)[:, 1]  # 'model' = dağıtılan (kalibre) model

        fig, eksen = plt.subplots(1, 2, figsize=(12, 4.6))

        # (a) Tahmin edilen olasılık dağılımı (log ölçekli, çünkü çoğu işlem ~0).
        kovalar = np.linspace(0, 1, 41)
        eksen[0].hist(ham_ol, bins=kovalar, alpha=0.6, label="Kalibrasyon öncesi (ham)")
        eksen[0].hist(kal_ol, bins=kovalar, alpha=0.6, label="Kalibrasyon sonrası")
        eksen[0].set_yscale("log")
        eksen[0].set_xlabel("Tahmin edilen dolandırıcılık olasılığı")
        eksen[0].set_ylabel("İşlem sayısı (log)")
        eksen[0].set_title("Olasılık Dağılımı")
        eksen[0].legend()

        # (b) Karar bölgesinde (olasılık > 0,02) güvenilirlik diyagramı.
        for olasilik, etiket in [(ham_ol, "Ham"), (kal_ol, "Kalibre")]:
            maske = olasilik > 0.02
            if maske.sum() > 20:
                frac_poz, ort_tahmin = calibration_curve(
                    y_test_arr[maske], olasilik[maske], n_bins=5, strategy="quantile"
                )
                eksen[1].plot(ort_tahmin, frac_poz, marker="o", label=etiket)
        eksen[1].plot([0, 1], [0, 1], "k--", label="Mükemmel")
        eksen[1].set_xlabel("Ortalama tahmin edilen olasılık")
        eksen[1].set_ylabel("Gerçek dolandırıcılık frekansı")
        eksen[1].set_title("Güvenilirlik (karar bölgesi: olasılık > 0,02)")
        eksen[1].legend(loc="upper left")
        _kaydet(fig, "guvenilirlik_diyagrami.png")

    print(">> Tüm grafikler üretildi.")


if __name__ == "__main__":
    main()
