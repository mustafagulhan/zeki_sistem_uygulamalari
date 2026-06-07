"""Dengesizlik stratejilerini karşılaştırır. Kullanım: python -m scripts.karsilastir_dengesizlik"""
from __future__ import annotations

import sys
import time

from src import data_loader, evaluate, preprocess, resampling, train
from src.config import ayarlari_yukle


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ayarlar = ayarlari_yukle()
    print(">> Veri okunuyor ve bölünüyor (tüm stratejiler aynı bölmeyi kullanır)...")
    df = data_loader.veri_oku(ayarlar.csv_mutlak_yolu())
    X_train, X_test, y_train, y_test = data_loader.egitim_test_bol(df, ayarlar)

    scaler = preprocess.scaler_egit(X_train, ayarlar)
    X_train_s = preprocess.donustur(X_train, scaler, ayarlar)
    X_test_s = preprocess.donustur(X_test, scaler, ayarlar)
    y_test_arr = y_test.to_numpy()

    stratejiler = ["class_weight", "smote", "undersample"]
    sonuclar = []

    for strateji in stratejiler:
        ayarlar.dengesizlik.strateji = strateji
        bas = time.perf_counter()
        X_d, y_d, spw = resampling.uygula(X_train_s, y_train, ayarlar)
        model = train.model_egit(X_d, y_d, spw, ayarlar)
        sure = time.perf_counter() - bas

        olasiliklar = model.predict_proba(X_test_s)[:, 1]
        esik = evaluate.en_iyi_f1_esigi(y_test_arr, olasiliklar)
        m = evaluate.metrikleri_hesapla(y_test_arr, olasiliklar, esik)
        m["strateji"] = strateji
        m["egitim_sn"] = round(sure, 1)
        m["egitim_boyutu"] = len(X_d)
        sonuclar.append(m)
        print(f"   [{strateji}] tamamlandı ({sure:.1f}s)")

    # Tablo
    print("\n" + "=" * 78)
    baslik = f"{'Strateji':<14}{'AUC-PR':>9}{'ROC-AUC':>9}{'Precision':>11}{'Recall':>9}{'F1':>8}{'Eğitim(sn)':>12}"
    print(baslik)
    print("-" * 78)
    for m in sonuclar:
        print(
            f"{m['strateji']:<14}{m['auc_pr']:>9.4f}{m['roc_auc']:>9.4f}"
            f"{m['precision']:>11.4f}{m['recall']:>9.4f}{m['f1']:>8.4f}{m['egitim_sn']:>12.1f}"
        )
    print("=" * 78)
    print("Not: AUC-PR dengesiz veride birincil metriktir; eşik her stratejide F1'i en iyiler.")


if __name__ == "__main__":
    main()
