# Ortak imaj: hem API hem panel bu imajdan çalışır (komut compose'ta belirlenir).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Önce bağımlılıklar (katman önbelleği için).
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Uygulama kodu.
COPY . .

# API portu (panel 8501 kullanır).
EXPOSE 8000 8501

# Varsayılan komut: API. Panel servisi compose'ta override eder.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
