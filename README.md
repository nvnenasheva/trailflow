# trailflow

## Purpose of this project: to reduce a "no-show" in a medical research study

**What it does:** predicts the risk of missed/late appointments and recommends low-cost actions (reminders, rescheduling, televisits, vouchers).

**Business metric:** *Expected Net Benefit (€ / visit)* = Savings - Costs.

**Who are the users:** Pharma/Biotech (sponsors of the research), clinics, telemedicine, and integrators.

## Importance:
1. Visit skipping --> protocol deviations, data processing, and data-lock delays.
2. Low-cost interventions work, but they need to be targeted (budget is limited).
3. We optimize the Euro-effect, not just the AUC: we select the top K visits at risk for a given budget.

## Project structure #TODO
```bash
trialflow/
  README.md
  requirements.txt
  Makefile
  .github/workflows/ci.yaml
  .pre-commit-config.yaml
  src/trialflow/
    __init__.py
    ingest_fhir.py      # загрузка/подготовка данных (заглушка)
    features.py         # фичи и трансформации
    train.py            # обучение и сохранение артефактов
    evaluate.py         # метрики/отчёты
    policy.py           # выбор top-K под бюджет (позже)
    serve.py            # FastAPI (/health, /score)
    app_demo.py         # Streamlit UI
  tests/
    test_features.py
    test_metrics.py
  data/                 # данные (в .gitignore)
  notebooks/
  reports/
    model_card.md       # шаблон карточки модели
```

## Quick start

**Requirements:** Python 3.11+, Linux/Mac/WSL.
```bash
git clone https://github.com/nvnenasheva/trailflow.git
cd trialflow

python -m venv .venv && source .venv/bin/activate
make install
```

### Local demo execution

Streamlit: 
  `streamlit run src/trialflow/app_demo.py`
API: 
  `uvicorn src.trialflow.serve:app --reload`
ROI for the budget Top-K=25% and price of passed visit C: `ENB = prevented_no_show(K) * C - K*N*cost_intervention`.
```bash
python -m src.trialflow.roi --n 1000 --k 0.25 --base_p 0.12 --uplift 0.30 --c 200 --cost 1.5
```
Parameters could be changed in the `notebooks/roi_simulation.ipynb`.

```bash
make api        # FastAPI on http://127.0.0.1:8000
make app        # Run Streamlit client
GET http://127.0.0.1:8000/health  # --> {"status":"ok"} chech the condition
```

### Execute pipeline

```bash
make data    # ingest artificial/CSV (later)
make train   # learning (logreg/LightGBM - later)
make eval    # metrics and report (later)
```
All of the commands already exist in the `Makefile`. They will write log files into the `reports/` folder. 


### Regulation #TODO
Model Card: 

