"""Streamlit kontrol paneli (canlı akış, işlem inceleme, model metrikleri)."""
from __future__ import annotations

import os
import time

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import PROJE_KOK, ayarlari_yukle
from src.schemas import OZELLIK_ISIMLERI

API_URL = os.getenv("FRAUD_API_URL", "http://localhost:8000")
ayarlar = ayarlari_yukle()

st.set_page_config(page_title="Dolandırıcılık Tespit Paneli", page_icon="🛡️", layout="wide")


# --------------------------- Yardımcılar ---------------------------
@st.cache_data
def test_ornegini_yukle() -> pd.DataFrame | None:
    """Eğitimde kaydedilen test örneğini yükler (akış simülasyonu için)."""
    yol = ayarlar.metrik_yolu().parent / "test_ornegi.csv"
    if not yol.exists():
        return None
    return pd.read_csv(yol)


@st.cache_data
def metrikleri_yukle() -> dict | None:
    import json

    yol = ayarlar.metrik_yolu()
    if not yol.exists():
        return None
    with open(yol, "r", encoding="utf-8") as f:
        return json.load(f)


def api_saglikli() -> tuple[bool, str]:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3.0)
        d = r.json()
        return bool(d.get("model_yuklu")), d.get("mesaj", "")
    except httpx.HTTPError as e:
        return False, f"API'ye ulaşılamadı: {e}"


def islem_skorla(satir: dict, esik: float | None = None) -> dict:
    """Tek işlemi API'ye gönderip yanıtı döndürür."""
    girdi = {ad: float(satir[ad]) for ad in OZELLIK_ISIMLERI}
    params = {} if esik is None else {"esik": esik}
    r = httpx.post(f"{API_URL}/predict", json=girdi, params=params, timeout=10.0)
    r.raise_for_status()
    return r.json()


# --------------------------- Başlık ---------------------------
st.title("🛡️ Gerçek Zamanlı Kredi Kartı Dolandırıcılık Tespit Paneli")
st.caption("XGBoost + SHAP açıklanabilirlik | FastAPI skorlama servisi | Zeki Sistem Uygulamaları")

saglikli, mesaj = api_saglikli()
if saglikli:
    st.success(f"API bağlı: {API_URL}")
else:
    st.error(f"API hazır değil ({API_URL}). Önce modeli eğitip API'yi başlatın. Detay: {mesaj}")

ornek_df = test_ornegini_yukle()

sekme1, sekme2, sekme3 = st.tabs(["📡 Canlı Akış", "🔍 İşlem İncele", "📊 Model Metrikleri"])


# --------------------------- Sekme 1: Canlı Akış ---------------------------
with sekme1:
    st.subheader("İşlem akışı simülasyonu")
    if ornek_df is None:
        st.warning("test_ornegi.csv bulunamadı. Önce `python -m src.train` ile modeli eğitin.")
    else:
        kol1, kol2, kol3 = st.columns(3)
        adet = kol1.slider("İşlem sayısı", 10, 300, 60, step=10)
        hiz = kol2.slider("Gecikme (sn/işlem)", 0.0, 0.5, 0.05, step=0.05)
        esik = kol3.slider("Karar eşiği", 0.0, 1.0, 0.5, step=0.01)

        if st.button("▶️ Akışı başlat", disabled=not saglikli):
            # Dolandırıcılık örneklerinin görünür olması için karışık örnek al.
            ornek = ornek_df.sample(min(adet, len(ornek_df)), random_state=int(time.time()))
            akis_alani = st.empty()
            uyari_alani = st.container()
            satirlar: list[dict] = []
            uyari_sayisi = 0

            for _, satir in ornek.iterrows():
                yanit = islem_skorla(satir.to_dict(), esik)
                kayit = {
                    "Tutar": round(float(satir["Amount"]), 2),
                    "Olasılık": round(yanit["dolandiricilik_olasiligi"], 4),
                    "Karar": "🚨 DOLANDIRICILIK" if yanit["karar"] else "✅ Normal",
                    "Gerçek": "dolandırıcılık" if int(satir.get("Class", 0)) == 1 else "normal",
                }
                satirlar.append(kayit)
                if yanit["karar"]:
                    uyari_sayisi += 1
                akis_alani.dataframe(
                    pd.DataFrame(satirlar[::-1]), use_container_width=True, height=360
                )
                if hiz > 0:
                    time.sleep(hiz)

            st.metric("Toplam uyarı (eşik aşan işlem)", uyari_sayisi)


# --------------------------- Sekme 2: İşlem İncele ---------------------------
with sekme2:
    st.subheader("Tek işlem skorlama + SHAP açıklaması")
    if ornek_df is None:
        st.warning("Örnek veri yok. Önce modeli eğitin.")
    else:
        idx = st.number_input(
            "İncelenecek işlem indeksi", 0, len(ornek_df) - 1, 0, step=1
        )
        if st.button("🔍 Skorla", disabled=not saglikli):
            satir = ornek_df.iloc[int(idx)].to_dict()
            yanit = islem_skorla(satir)

            k1, k2, k3 = st.columns(3)
            k1.metric("Dolandırıcılık olasılığı", f"{yanit['dolandiricilik_olasiligi']:.2%}")
            k2.metric("Karar", "DOLANDIRICILIK" if yanit["karar"] else "Normal")
            k3.metric("Gerçek etiket", "dolandırıcılık" if int(satir.get("Class", 0)) == 1 else "normal")

            katki_df = pd.DataFrame(yanit["en_etkili_ozellikler"])
            fig = px.bar(
                katki_df,
                x="shap_katki",
                y="ozellik",
                orientation="h",
                color="shap_katki",
                color_continuous_scale="RdBu_r",
                title="En etkili özellikler (SHAP katkısı: pozitif → dolandırıcılığa iter)",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Pozitif SHAP değeri ilgili özelliğin dolandırıcılık olasılığını artırdığını gösterir.")


# --------------------------- Sekme 3: Metrikler ---------------------------
with sekme3:
    st.subheader("Eğitimde kaydedilen model performansı")
    metrikler = metrikleri_yukle()
    if metrikler is None:
        st.warning("metrics.json bulunamadı. Önce modeli eğitin.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("AUC-PR", f"{metrikler['auc_pr']:.4f}")
        k2.metric("ROC-AUC", f"{metrikler['roc_auc']:.4f}")
        k3.metric("F1", f"{metrikler['f1']:.4f}")
        k4.metric("Eşik", f"{metrikler['esik']:.4f}")

        k5, k6 = st.columns(2)
        k5.metric("Precision", f"{metrikler['precision']:.4f}")
        k6.metric("Recall", f"{metrikler['recall']:.4f}")

        cm = metrikler["confusion_matrix"]
        tp, fn, fp, tn = cm["tp"], cm["fn"], cm["fp"], cm["tn"]

        # Tek satırlık özet
        o1, o2, o3 = st.columns(3)
        o1.metric("Yakalanan dolandırıcılık", f"{tp} / {tp + fn}")
        o2.metric("Kaçan (yanlış negatif)", f"{fn}")
        o3.metric("Yanlış alarm (yanlış pozitif)", f"{fp}")

        cm_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Gerçek: Normal", "Gerçek: Dolandırıcılık"],
            columns=["Tahmin: Normal", "Tahmin: Dolandırıcılık"],
        )
        st.write("**Karmaşıklık Matrisi (Confusion Matrix)**")
        st.dataframe(cm_df, use_container_width=True)
        st.caption(
            f"Dengesizlik stratejisi: {metrikler.get('strateji', '-')} | "
            f"Kalibrasyon: {'açık' if metrikler.get('kalibrasyon') else 'kapalı'}"
        )

        # Eğitimde üretilen grafikler (varsa) gömülü gösterilir.
        figur_dizini = PROJE_KOK / "reports" / "figures"
        figurler = [
            ("pr_roc_egrileri.png", "Precision-Recall ve ROC eğrileri"),
            ("shap_ozet.png", "SHAP özeti — en etkili özellikler"),
            ("guvenilirlik_diyagrami.png", "Olasılık dağılımı ve güvenilirlik (kalibrasyon)"),
            ("strateji_karsilastirma.png", "Dengesizlik stratejilerinin karşılaştırması"),
        ]
        mevcut = [(d, b) for d, b in figurler if (figur_dizini / d).exists()]
        if mevcut:
            st.write("---")
            st.write("**Grafikler**")
            # 2'li ızgara: grafikler daha küçük ve yan yana görünsün.
            for i in range(0, len(mevcut), 2):
                kolonlar = st.columns(2)
                for kolon, (dosya, baslik) in zip(kolonlar, mevcut[i : i + 2]):
                    kolon.image(str(figur_dizini / dosya), caption=baslik, use_column_width=True)
        else:
            st.caption("Grafik üretmek için: python -m scripts.rapor_gorselleri")
