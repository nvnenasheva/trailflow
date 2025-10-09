.PHONY: install lint fmt test data train eval app api

install:
	pip install -r requirements.txt
	pip install pre-commit ruff mypy pytest
	pre-commit install

lint:
	ruff check src tests
	mypy src

fmt:
	ruff check --fix src tests || true

test:
	pytest -q

data:
	python src/trialflow/ingest_fhir.py

train:
	python src/trialflow/train.py

eval:
	python src/trialflow/evaluate.py

app:
	streamlit run src/trialflow/app_demo.py

api:
	uvicorn src.trialflow.serve:app --reload
