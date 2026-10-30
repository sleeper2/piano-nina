FROM python:3.11-slim
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir numpy music21
WORKDIR /app
CMD ["python", "evaluate_train_data.py"]
