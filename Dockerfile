# Container for Vercel (Container Runtime), Hugging Face Spaces (Docker SDK) and any other Docker host.
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

# The models are in the image now: never contact Hugging Face at runtime (faster start, no network failures).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --chown=user . .

EXPOSE 8501
# Listens on $PORT when the host sets it (Vercel: set PORT=8501 in the project settings), otherwise on 8501.
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none"]
