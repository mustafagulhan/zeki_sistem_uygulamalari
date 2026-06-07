"""Pydantic veri sözleşmeleri (işlem girdisi ve tahmin yanıtı)."""
from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict

# Modelin beklediği özellik sırası (Class hariç). Tüm katmanlar bu sırayı referans alır.
OZELLIK_ISIMLERI: list[str] = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
)


class Islem(BaseModel):
    """Tek bir kredi kartı işlemi (model girdisi)."""

    # extra="forbid": beklenmeyen alanları reddet (sözleşme hatalarını erken yakala).
    model_config = ConfigDict(extra="forbid")

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

    def ozellik_vektoru(self) -> list[float]:
        """Modelin beklediği sırada özellik listesi döndürür."""
        return [getattr(self, ad) for ad in OZELLIK_ISIMLERI]


class OzellikKatkisi(BaseModel):
    """Tek bir özelliğin SHAP katkısı (açıklanabilirlik için)."""

    ozellik: str
    deger: float          # işlemdeki ham değer
    shap_katki: float     # pozitif => dolandırıcılık olasılığını artırır


class TahminYaniti(BaseModel):
    """/predict yanıtı."""

    dolandiricilik_olasiligi: float = Field(..., ge=0.0, le=1.0)
    karar: int = Field(..., description="1 = dolandırıcılık (eşik aşıldı), 0 = normal")
    esik: float
    en_etkili_ozellikler: list[OzellikKatkisi]


class TopluTahminYaniti(BaseModel):
    """/predict/batch yanıtı."""

    sonuclar: list[TahminYaniti]
