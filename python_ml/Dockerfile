# AgriShield AI — Hugging Face Spaces Docker image
# Space: https://huggingface.co/spaces/max1m1ll1en/rwac-v.0.1
#
# HF Spaces requires the app to listen on port 7860.

FROM python:3.11-slim

# system deps (needed by pdfplumber / lxml)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install Python deps first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the source
COPY . .

# HF Spaces always uses port 7860
ENV FLASK_PORT=7860

EXPOSE 7860

# If artifacts/ already exist (committed) → serve immediately.
# Otherwise retrain from scratch on first boot.
CMD ["sh", "-c", \
     "if [ -f artifacts/rain_prediction_rf.pkl ]; then \
        echo 'Artifacts found — starting service …'; \
        python main.py serve; \
      else \
        echo 'No artifacts — retraining (this takes ~2 min) …'; \
        python main.py retrain && python main.py serve; \
      fi"]
