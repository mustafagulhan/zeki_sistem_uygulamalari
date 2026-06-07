# 🛡️ Gerçek Zamanlı Kredi Kartı Dolandırıcılık Tespit Sistemi

## Anlatım Videosu

[<img src="https://i.ytimg.com/vi/Hc79sDi3f0U/maxresdefault.jpg" width="50%">]([https://www.youtube.com/watch?v=Hc79sDi3f0U](https://drive.google.com/file/d/1EvY5LAV89mq9QcqEAuSh5-h4mPMMD7eJ/view?usp=sharing))


**Zeki Sistem Uygulamaları – Final Projesi**
MAKÜ Yazılım Mühendisliği Anabilim Dalı, Yüksek Lisans
Danışman: Prof. Dr. Serkan BALLI · Geliştiren: Mustafa GÜLHAN

Dengesiz kredi kartı işlem verisinde **XGBoost** ile dolandırıcılık tespiti yapan,
kararlarını **SHAP** ile açıklayan, **FastAPI** üzerinden skorlayan ve **Streamlit**
panelinde gerçek zamanlı izleme sunan uçtan uca bir sistem.

Bu proje, vize aşamasında incelenen Chen et al. (2025) sistematik derlemesinin işaret
ettiği iki boşluğu kapatır: **(1)** açıklanabilirlik (XAI) entegrasyonu ve **(2)** gerçek
zamanlı sistem değerlendirmesi.

---

## 🏗️ Mimari

```
                ┌────────────────┐
  creditcard.csv│  Eğitim        │   models/  (model.pkl, scaler.pkl,
  ─────────────▶│  Pipeline      │──────────▶ explainer.pkl, metrics.json,
                │  (src/train.py)│            test_ornegi.csv)
                └────────────────┘
                                              │ (artifact'ları yükler)
                                              ▼
  HTTP istek    ┌────────────────┐   skorla   ┌────────────────┐
  ─────────────▶│  FastAPI       │───────────▶│  Servis        │
  /predict      │  (api/main.py) │            │  (service.py)  │
                └────────────────┘            │  XGBoost+SHAP  │
                        ▲                     └────────────────┘
                        │ httpx
                ┌────────────────┐
   Kullanıcı ◀─▶│  Streamlit     │   Canlı akış + uyarılar + SHAP + metrikler
                │  (dashboard/)  │
                └────────────────┘
```

| Katman | Sorumluluk | Dosya |
|---|---|---|
| Konfigürasyon | Tip güvenli ayarlar (`config.yaml`) | `src/config.py` |
| Veri | Okuma + eğitim/test bölme | `src/data_loader.py` |
| Ön işleme | RobustScaler (Time, Amount) | `src/preprocess.py` |
| Dengesizlik | class_weight / SMOTE / undersample | `src/resampling.py` |
| Eğitim | XGBoost + artifact kaydı | `src/train.py` |
| Değerlendirme | AUC-PR, F1, eşik analizi | `src/evaluate.py` |
| Açıklanabilirlik | SHAP TreeExplainer | `src/explain.py` |
| API | `/health`, `/predict`, `/predict/batch` | `api/main.py`, `api/service.py` |
| Panel | Gerçek zamanlı demo | `dashboard/app.py` |

---

## 📦 Kurulum

Gereksinim: **Python 3.11+**

```powershell
# 1) Sanal ortam
python -m venv .venv
.venv\Scripts\Activate.ps1        # Linux/Mac: source .venv/bin/activate

# 2) Bağımlılıklar
pip install -r requirements.txt
```

### Veri setini indirme

[Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
veri setini indirip `creditcard.csv` dosyasını **`data/`** klasörüne koyun:

```
proje/data/creditcard.csv
```

> Veri seti (~150 MB) repoya dahil **değildir** (`.gitignore`). 284.807 işlem,
> 492 dolandırıcılık (%0,172). Özellikler `V1–V28` gizlilik için PCA ile dönüştürülmüştür.

---

## 🚀 Kullanım

### 1. Modeli eğit

```powershell
python -m src.train --config config.yaml
```

`models/` altında `model.pkl`, `scaler.pkl`, `explainer.pkl`, `metrics.json` ve
`test_ornegi.csv` üretilir. Konsolda AUC-PR / F1 / Precision / Recall raporlanır.

### 2. API'yi başlat

```powershell
uvicorn api.main:app --reload
```

- Sağlık: http://localhost:8000/health
- Etkileşimli dokümanlar (Swagger): http://localhost:8000/docs

Örnek istek:

```powershell
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" `
  -d '{"Time":0,"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,"V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,"V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,"V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,"V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,"V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02,"Amount":149.62}'
```

### 3. Paneli başlat

Ayrı bir terminalde:

```powershell
streamlit run dashboard/app.py
```

http://localhost:8501 — **Canlı Akış**, **İşlem İncele** (SHAP) ve **Model Metrikleri**
sekmeleri.

### Docker ile (API + panel birlikte)

```powershell
docker compose up --build
```

> Önce modeli eğitip `models/` klasörünü doldurun; compose bu klasörü konteynerlere bağlar.

---

## 🧪 Testler

```powershell
pytest
```

Testler sentetik veriyle çalışır — gerçek Kaggle veri seti gerektirmez. Ön işleme,
skorlama sözleşmesi, eşik davranışı ve metrik makullüğü doğrulanır.

---

## ⚙️ Konfigürasyon (`config.yaml`)

Öne çıkan ayarlar:

- `dengesizlik.strateji`: `class_weight` | `smote` | `undersample` | `none`
- `model.*`: XGBoost hiperparametreleri
- `kalibrasyon.etkin`: `true` ise olasılıklar isotonic regresyonla kalibre edilir
- `esik.f1_optimize`: `true` ise F1'i en iyileyen karar eşiği otomatik seçilir
- `api.top_shap_ozellik`: yanıtta döndürülecek en etkili özellik sayısı

---

## 📊 Görsel ve Analiz Üretimi

Eğitilmiş modelden panel/rapor görsellerini ve karşılaştırma tablosunu yeniden üretmek için:

```powershell
python -m scripts.rapor_gorselleri        # reports/figures/*.png üretir (panelde kullanılır)
python -m scripts.karsilastir_dengesizlik # dengesizlik stratejisi karşılaştırma tablosu
```

---

## 📁 Proje Yapısı

```
proje/
├── config.yaml              # Tüm ayarlar
├── requirements.txt
├── Dockerfile / docker-compose.yml
├── src/                     # Çekirdek ML pipeline
├── api/                     # FastAPI skorlama servisi
├── dashboard/               # Streamlit panel
├── scripts/                 # Panel görselleri ve karşılaştırma betikleri
├── tests/                   # pytest (sentetik veri)
├── reports/                 # Üretilen görseller (figures/)
├── data/                    # creditcard.csv (gitignore)
└── models/                  # Eğitilmiş artifact'lar (gitignore)
```
