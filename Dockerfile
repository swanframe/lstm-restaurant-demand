FROM python:3.10-slim

WORKDIR /app

# Install system dependencies INCLUDING MAKE
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Sisanya tetap sama...
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

RUN mkdir -p data/raw data/interim data/processed figures

ENV PYTHONPATH=/app

CMD ["python", "scripts/run_preprocessing.py", "--config", "configs/params.yaml", "--paths", "configs/paths.yaml"]