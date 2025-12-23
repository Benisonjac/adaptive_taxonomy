FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install fastapi uvicorn

COPY src/ ./src/
COPY data/ ./data/
COPY .env.example ./

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD ["python", "src/api.py"]
