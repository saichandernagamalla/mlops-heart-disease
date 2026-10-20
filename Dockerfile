FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=4
WORKDIR /app
COPY requirements.txt .
ARG PIP_INDEX_URL=https://pypi.org/simple
RUN python -m pip install --no-cache-dir --timeout=60 --retries=4 --index-url "$PIP_INDEX_URL" -r requirements.txt
COPY src ./src
COPY artifacts ./artifacts
EXPOSE 8000
USER 10001
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
