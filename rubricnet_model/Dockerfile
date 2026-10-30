FROM python:3.10-slim

ARG RUBRICNET_COMMIT=3ff4a9342948b9588541a2a8524eab68045f6001
# Upstream requirements.txt installs music21 from this fork (music21 of May 2023 + 2 small commits).
ARG MUSIC21_COMMIT=21665005c567355cd003a46d81c0d85011f72747

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# music21 fork, pinned, with a parser fix from patches/ (see README).
COPY patches /tmp/patches
RUN git init -q /tmp/music21 \
    && git -C /tmp/music21 fetch -q --depth 1 https://github.com/adhooge/music21.git ${MUSIC21_COMMIT} \
    && git -C /tmp/music21 checkout -q FETCH_HEAD \
    && git -C /tmp/music21 apply /tmp/patches/music21-*.patch \
    && pip install --no-cache-dir /tmp/music21 \
    && rm -rf /tmp/music21

# The upstream repo (code, feature extractor + camera-ready weights), pinned to a commit.
RUN git clone https://github.com/PRamoneda/rubricnet.git /opt/rubricnet \
    && git -C /opt/rubricnet checkout ${RUBRICNET_COMMIT} \
    && find /opt/rubricnet -name '__pycache__' -type d -prune -exec rm -rf {} +
ENV PYTHONPATH=/opt/rubricnet:/opt/rubricnet/extractor/DifficultyFeatures/code \
    WANDB_MODE=disabled \
    PYTHONUNBUFFERED=1

COPY predict.py evaluate.py tempo.py ./

CMD ["sh", "-c", "python predict.py && python evaluate.py"]
