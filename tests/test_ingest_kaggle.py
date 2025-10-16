"""test for ingest_kaggle.py
Usage:
pytest -q tests/test_ingest_kaggle.py
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(
    "data/processed/visits.parquet"
)  # path should match ingest_kaggle.py output

REQUIRED_COLS = {
    "visit_id",
    "patient_pseudo_id",
    "date",
    "site_id",
    "visit_type",
    "weekday",
    "lead_time_days",
    "is_first_visit",
    "sms_received",
    "scholarship",
    "hipertension",
    "diabetes",
    "alcoholism",
    "handicap",
    "gender",
    "age",
    "no_show",
}


def test_file_exists():
    assert DATA_PATH.exists(), (
        f"Файл не найден: {DATA_PATH}. Сначала запусти ingest_kaggle.py"
    )


def test_schema_and_basic_quality():
    df = pd.read_parquet(DATA_PATH)
    # columns and data types
    assert REQUIRED_COLS.issubset(df.columns), (
        f"Нет колонок: {REQUIRED_COLS - set(df.columns)}"
    )
    assert df["no_show"].isin([0, 1]).all(), "no_show should be 0/1"
    assert df["weekday"].between(0, 6).all(), "weekday должен быть в [0..6]"
    assert (df["lead_time_days"] >= 0).all(), "lead_time_days должен быть >= 0"
    assert df["visit_type"].notna().all(), "visit_type не должен быть пустым"
    # data quality
    assert pd.api.types.is_datetime64_any_dtype(df["date"]), (
        "date should be  datetime64"
    )
    assert len(df) > 1000, (
        "number of strings is too small -- it looks like incorrect import"
    )
    # уникальность визита (для Kaggle обычно уникален AppointmentID)
    assert df["visit_id"].is_unique, "visit_id should be unique (check the input CSV)"


def test_reasonable_rates():
    df = pd.read_parquet(DATA_PATH)
    rate = df["no_show"].mean()
    # для Kaggle но-шоу ~ 0.2; оставим мягкие границы
    assert 0.05 < rate < 0.5, f"suspicious no_show rate={rate:.3f}"
