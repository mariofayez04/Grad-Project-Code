# Container for Hugging Face Spaces (Docker SDK). Also works on any Docker host.
FROM python:3.13-slim

# Hugging Face Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch first (the default Linux wheel pulls ~2 GB of CUDA libraries we don't need).
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1

# Download both models at build time so the app starts fast.
RUN python -c "from transformers import pipeline; pipeline('ner', model='Dizex/FoodBaseBERT-NER'); \
from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none"]
