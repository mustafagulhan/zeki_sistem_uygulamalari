"""config.yaml'ı tip güvenli Pydantic modellerine yükler."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Proje kök dizini: bu dosya proje/src/config.py konumunda olduğundan iki üst dizin.
PROJE_KOK = Path(__file__).resolve().parents[1]
VARSAYILAN_CONFIG = PROJE_KOK / "config.yaml"


class VeriAyari(BaseModel):
    csv_yolu: str
    hedef_kolon: str = "Class"
    test_orani: float = 0.2
    rastgele_tohum: int = 42
    zaman_sirali_bolme: bool = True


class OnislemeAyari(BaseModel):
    olceklenecek_kolonlar: list[str] = Field(default_factory=lambda: ["Time", "Amount"])


class DengesizlikAyari(BaseModel):
    strateji: str = "class_weight"  # class_weight | smote | undersample | none
    smote_komsu_k: int = 5


class ModelAyari(BaseModel):
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    rastgele_tohum: int = 42


class KalibrasyonAyari(BaseModel):
    etkin: bool = True
    yontem: str = "isotonic"  # isotonic | sigmoid
    cv: int = 3


class EsikAyari(BaseModel):
    varsayilan: float = 0.5
    f1_optimize: bool = True


class CiktiAyari(BaseModel):
    # "model_" ön ekli alanlar Pydantic korumalı ad alanıyla çakışmasın.
    model_config = ConfigDict(protected_namespaces=())

    model_dizini: str = "models"
    model_dosyasi: str = "model.pkl"
    scaler_dosyasi: str = "scaler.pkl"
    explainer_dosyasi: str = "explainer.pkl"
    metrik_dosyasi: str = "metrics.json"


class ApiAyari(BaseModel):
    baslik: str = "Dolandırıcılık Tespit API"
    top_shap_ozellik: int = 6


class Ayarlar(BaseModel):
    """Tüm konfigürasyonun kök modeli."""

    # model_yolu() vb. yardımcı metotlar korumalı ad alanı uyarısı vermesin.
    model_config = ConfigDict(protected_namespaces=())

    veri: VeriAyari
    onisleme: OnislemeAyari
    dengesizlik: DengesizlikAyari
    model: ModelAyari
    # Geriye dönük uyumluluk için varsayılan (eski config.yaml'larda blok olmayabilir).
    kalibrasyon: KalibrasyonAyari = Field(default_factory=KalibrasyonAyari)
    esik: EsikAyari
    cikti: CiktiAyari
    api: ApiAyari

    # --- Türetilmiş yollar (kök dizine göre mutlak) ---
    def model_yolu(self) -> Path:
        return PROJE_KOK / self.cikti.model_dizini / self.cikti.model_dosyasi

    def scaler_yolu(self) -> Path:
        return PROJE_KOK / self.cikti.model_dizini / self.cikti.scaler_dosyasi

    def explainer_yolu(self) -> Path:
        return PROJE_KOK / self.cikti.model_dizini / self.cikti.explainer_dosyasi

    def metrik_yolu(self) -> Path:
        return PROJE_KOK / self.cikti.model_dizini / self.cikti.metrik_dosyasi

    def csv_mutlak_yolu(self) -> Path:
        return PROJE_KOK / self.veri.csv_yolu


def ayarlari_yukle(yol: str | Path | None = None) -> Ayarlar:
    """config.yaml dosyasını okuyup doğrulanmış `Ayarlar` nesnesi döndürür."""
    config_yolu = Path(yol) if yol is not None else VARSAYILAN_CONFIG
    with open(config_yolu, "r", encoding="utf-8") as f:
        ham = yaml.safe_load(f)
    return Ayarlar(**ham)
