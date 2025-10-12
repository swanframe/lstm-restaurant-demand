.PHONY: setup format lint test data-synth preprocess train predict clean

PYTHON := python

setup:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

format:
	black src/ || true

lint:
	ruff check src/ || true

# Phase 2: generate synthetic CSVs into data/raw
data-synth:
	python scripts/generate_synthetic_data.py --config configs/params.yaml --paths configs/paths.yaml

# Phase 3: run preprocessing
preprocess:
	$(PYTHON) scripts/run_preprocessing.py --config configs/params.yaml --paths configs/paths.yaml

# Phase 4: train model
train:
	$(PYTHON) scripts/run_training.py --config configs/params.yaml --paths configs/paths.yaml

# Phase 5: run prediction
predict:
	$(PYTHON) scripts/run_prediction.py --config configs/params.yaml --paths configs/paths.yaml

clean:
	rm -rf data/interim/* data/processed/* figures/* || true