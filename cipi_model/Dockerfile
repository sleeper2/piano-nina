FROM python:3.10-slim

ARG CIPI_COMMIT=e68961c7012ebbbd3a4f9b0cc0998b832197c3cd

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# The upstream repo (code + pretrained weights), pinned to a commit, plus small parser fixes
# from patches/ (see README).
COPY patches /tmp/patches
RUN git clone https://github.com/PRamoneda/difficulty-prediction-CIPI.git /opt/cipi \
    && git -C /opt/cipi checkout ${CIPI_COMMIT} \
    && git -C /opt/cipi apply /tmp/patches/*.patch \
    && find /opt/cipi -name '__pycache__' -type d -prune -exec rm -rf {} +
ENV PYTHONPATH=/opt/cipi \
    WANDB_MODE=disabled \
    PYTHONUNBUFFERED=1

COPY predict.py evaluate.py tempo_subset.py ./

CMD ["sh", "-c", "python predict.py && python evaluate.py"]
