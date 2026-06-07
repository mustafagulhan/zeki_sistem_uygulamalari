"""Eğitim boru hattı (CLI). Kullanım: python -m src.train --config config.yaml"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

from . import data_loader, evaluate, explain, preprocess, resampling
from .config import Ayarlar, ayarlari_yukle


def model_olustur(scale_pos_weight: float, ayarlar: Ayarlar) -> XGBClassifier:
    """Konfigürasyondaki hiperparametrelerle eğitilmemiş bir XGBoost örneği döndürür."""
    m = ayarlar.model
    return XGBClassifier(
        n_estimators=m.n_estimators,
        max_depth=m.max_depth,
        learning_rate=m.learning_rate,
        subsample=m.subsample,
        colsample_bytree=m.colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=m.rastgele_tohum,
        eval_metric="aucpr",
        n_jobs=-1,
        tree_method="hist",
    )


def model_egit(X_train, y_train, scale_pos_weight: float, ayarlar: Ayarlar) -> XGBClassifier:
    """Temel XGBoost modelini eğitir."""
    model = model_olustur(scale_pos_weight, ayarlar)
    model.fit(X_train, y_train)
    return model


def kalibre_et(X_train, y_train, scale_pos_weight: float, ayarlar: Ayarlar) -> CalibratedClassifierCV:
    """XGBoost'u CalibratedClassifierCV (çapraz doğrulamalı) ile kalibre eder."""
    kalibre = CalibratedClassifierCV(
        model_olustur(scale_pos_weight, ayarlar),
        method=ayarlar.kalibrasyon.yontem,
        cv=ayarlar.kalibrasyon.cv,
    )
    kalibre.fit(X_train, y_train)
    return kalibre


def calistir(ayarlar: Ayarlar) -> dict:
    """Tüm eğitim sürecini çalıştırır ve metrik sözlüğü döndürür."""
    print(">> Veri okunuyor...")
    df = data_loader.veri_oku(ayarlar.csv_mutlak_yolu())
    X_train, X_test, y_train, y_test = data_loader.egitim_test_bol(df, ayarlar)
    print(f"   Eğitim: {len(X_train)} satır | Test: {len(X_test)} satır")
    print(f"   Eğitimde dolandırıcılık oranı: {y_train.mean():.4%}")

    print(">> Ölçekleme (RobustScaler) eğitiliyor...")
    scaler = preprocess.scaler_egit(X_train, ayarlar)
    X_train_s = preprocess.donustur(X_train, scaler, ayarlar)
    X_test_s = preprocess.donustur(X_test, scaler, ayarlar)

    print(f">> Dengesizlik stratejisi: {ayarlar.dengesizlik.strateji}")
    X_train_d, y_train_d, spw = resampling.uygula(X_train_s, y_train, ayarlar)
    print(f"   Eğitim seti boyutu (dengeleme sonrası): {len(X_train_d)} | scale_pos_weight={spw:.2f}")

    print(">> XGBoost eğitiliyor (temel model)...")
    base_model = model_egit(X_train_d, y_train_d, spw, ayarlar)
    y_test_arr = y_test.to_numpy()
    base_olasilik = base_model.predict_proba(X_test_s)[:, 1]
    brier_ham = float(brier_score_loss(y_test_arr, base_olasilik))

    # Dağıtılacak (deploy) model: kalibrasyon etkinse kalibre edilmiş, değilse temel model.
    deploy_model = base_model
    if ayarlar.kalibrasyon.etkin:
        print(f">> Olasılık kalibrasyonu uygulanıyor ({ayarlar.kalibrasyon.yontem})...")
        deploy_model = kalibre_et(X_train_d, y_train_d, spw, ayarlar)

    print(">> Değerlendirme...")
    olasiliklar = deploy_model.predict_proba(X_test_s)[:, 1]

    esik = ayarlar.esik.varsayilan
    en_iyi_esik = evaluate.en_iyi_f1_esigi(y_test_arr, olasiliklar)
    if ayarlar.esik.f1_optimize:
        esik = en_iyi_esik

    metrikler = evaluate.metrikleri_hesapla(y_test_arr, olasiliklar, esik)
    metrikler["varsayilan_esik_metrikleri"] = evaluate.metrikleri_hesapla(
        y_test_arr, olasiliklar, ayarlar.esik.varsayilan
    )
    metrikler["en_iyi_f1_esigi"] = float(en_iyi_esik)
    metrikler["strateji"] = ayarlar.dengesizlik.strateji
    metrikler["kalibrasyon"] = ayarlar.kalibrasyon.etkin
    metrikler["brier_ham"] = brier_ham  # kalibrasyon öncesi
    # metrikler["brier"] zaten kalibrasyon sonrası (deploy) olasılıklardan hesaplandı.

    print(f"   AUC-PR: {metrikler['auc_pr']:.4f} | ROC-AUC: {metrikler['roc_auc']:.4f}")
    print(
        f"   Eşik={esik:.4f} -> Precision: {metrikler['precision']:.4f} | "
        f"Recall: {metrikler['recall']:.4f} | F1: {metrikler['f1']:.4f}"
    )
    print(f"   Brier: ham={brier_ham:.5f} -> kalibre={metrikler['brier']:.5f}")

    print(">> SHAP explainer oluşturuluyor (temel model üzerinden)...")
    explainer = explain.explainer_olustur(base_model)

    print(">> Artefaktlar kaydediliyor...")
    model_dizini = Path(ayarlar.model_yolu()).parent
    model_dizini.mkdir(parents=True, exist_ok=True)
    joblib.dump(deploy_model, ayarlar.model_yolu())
    # Ham (kalibre edilmemiş) model, güvenilirlik diyagramı karşılaştırması için saklanır.
    joblib.dump(base_model, model_dizini / "model_ham.pkl")
    joblib.dump(scaler, ayarlar.scaler_yolu())
    joblib.dump(explainer, ayarlar.explainer_yolu())
    with open(ayarlar.metrik_yolu(), "w", encoding="utf-8") as f:
        json.dump(metrikler, f, ensure_ascii=False, indent=2)

    # Demo örneği: tüm dolandırıcılık + normal örneklem (gerçek etiketler korunur).
    ornek = X_test.copy()
    ornek["Class"] = y_test.to_numpy()
    dolandiricilik = ornek[ornek["Class"] == 1]
    normal = ornek[ornek["Class"] == 0].sample(
        min(1500, int((ornek["Class"] == 0).sum())),
        random_state=ayarlar.veri.rastgele_tohum,
    )
    demo = (
        pd.concat([dolandiricilik, normal])
        .sample(frac=1.0, random_state=ayarlar.veri.rastgele_tohum)  # karıştır
        .reset_index(drop=True)
    )
    ornek_yolu = model_dizini / "test_ornegi.csv"
    demo.to_csv(ornek_yolu, index=False)
    print(
        f"   Demo örneği: {len(demo)} işlem ({len(dolandiricilik)} dolandırıcılık) -> {model_dizini}"
    )
    return metrikler


def main() -> None:
    # Windows konsolunda (cp1252) Türkçe karakterlerin sorunsuz basılması için.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Dolandırıcılık tespit modeli eğitimi")
    parser.add_argument("--config", default=None, help="config.yaml yolu")
    args = parser.parse_args()
    ayarlar = ayarlari_yukle(args.config)
    calistir(ayarlar)
    print(">> Tamamlandı.")


if __name__ == "__main__":
    main()
