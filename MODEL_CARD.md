# Model Card — TrialFlow
**Purpose:** Predicting the risk of missed appointments and suggesting low-cost actions (reminders, rescheduling, televisits).
**Not for:** Clinical diagnostics/treatment. Decisions -- auxiliary (human-in-the-loop).

## Data
Synthea synthetics / public no-show; aggregated features: weekday, lead_time_days, is_first_visit, site_id (pseudo).

## Training
Split by time; models: Logistic/LightGBM; calibration (Platt/Isotonic). Tracking: MLflow run_id + git SHA + DVC hash.

## Metrics (valid set)
AUC, PR-AUC, Brier/ECE (calibration); business metrics: Expected Net Benefit (€), Recall@Top-K, PPV@Top-K.

## Limitations and Risks
Bias across sites/seasons; synthetic nature of the data; sensitivity to lead_time distribution.

## Mitigations
Calibration, drift monitoring (Evidently), periodic retraining, "safety floor" (minimum recall on critical visits).

## Versioning
Model version = {git_sha}+{mlflow_run_id}+{dvc_hash}. Returned in the `X-Model-Version` API header.