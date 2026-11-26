FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python scripts/fetch_models.py
EXPOSE 7860
CMD ["sh","-c","uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
