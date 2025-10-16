"""
ingest_kaggle.py — prepare Kaggle 'Medical Appointment No Shows' for TrialFlow

Usage:
python src/trialflow/ingest_kaggle.py --input data/raw/kaggle_no_show.csv --output data/processed/visits.parquet


Main steps

1) Load CSV (from data/raw/), normalize column names
2) Parse datetimes, drop rows without both dates
3) Feature engineering:
   - lead_time_days: days between booking and appointment (clipped to 0..180)
   - date, weekday  (floor to day)
   - age cleaned (0..120)
   - binary columns → int {0,1}
   - gender normalization (M/F/U)
   - target: 'no_show' -> 1 if Yes
   - site_id from Neighbourhood (hashed)
   - patient_pseudo_id
   - visit_id
   - is_first_visit (within dataset)
   - visit_type -- Kaggle dataset does not ptrovide; put constant
4) Unified output schema
5) Save to Parquet (in data/processed/)"""

import argparse
import hashlib
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _std_columns(cols) -> Dict[str, str]:
    """Lowercase, strip, replace '-' with '_'."""
    return {c: c.strip().lower().replace("-", "_") for c in cols}


def _hash_label(x: str, prefix: str = "site", n: int = 8) -> str:
    """Deterministic short hash for pseudo site_id."""
    return hashlib.sha256((prefix + str(x)).encode("utf-8")).hexdigest()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, help="Path to Kaggle CSV (KaggleV2-May-2016.csv)"
    )
    ap.add_argument(
        "--output", default="data/processed/visits.parquet", help="Output Parquet path"
    )
    args = ap.parse_args()

    path_in = Path(args.input)
    path_out = Path(args.output)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load and normalize column names
    df = pd.read_csv(path_in)
    df = df.rename(columns=_std_columns(df.columns))

    # Expected original columns (case-insensitive; hyphens handled):
    # patientid, appointmentid, gender, scheduledday, appointmentday, age,
    # neighbourhood, scholarship, hipertension, diabetes, alcoholism, handcap,
    # sms_received, no_show

    # 2) Parse datetimes, drop rows without both dates
    for col in ("scheduledday", "appointmentday"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["scheduledday", "appointmentday"]).copy()

    # 3) Feature engineering
    # Lead time (days) between booking and appointment; negatives clipped to 0
    lead = (df["appointmentday"] - df["scheduledday"]).dt.total_seconds() / (24 * 3600)
    df["lead_time_days"] = np.clip(np.round(lead), 0, 180).astype("Int64")

    # Appointment date (floor to day) & weekday
    df["date"] = df["appointmentday"].dt.floor("D")
    df["weekday"] = df["date"].dt.weekday.astype(int)

    # Age cleaned (0..120)
    if "age" in df.columns:
        df["age"] = (
            pd.to_numeric(df["age"], errors="coerce").fillna(0).clip(0, 120).astype(int)
        )
    else:
        df["age"] = 0

    # Binary columns → int {0,1}
    def bin_col(name: str) -> pd.Series:
        return (
            df[name].fillna(0).astype(int)
            if name in df.columns
            else pd.Series(0, index=df.index, dtype=int)
        )

    df["scholarship"] = bin_col("scholarship")
    df["hipertension"] = bin_col("hipertension")
    df["diabetes"] = bin_col("diabetes")
    df["alcoholism"] = bin_col("alcoholism")
    df["handicap"] = bin_col("handcap")
    df["sms_received"] = bin_col("sms_received")

    # Gender normalization (M/F/U)
    if "gender" in df.columns:
        df["gender"] = (
            df["gender"]
            .astype(str)
            .str.upper()
            .str[0]
            .map({"M": "M", "F": "F"})
            .fillna("U")
        )
    else:
        df["gender"] = "U"

    # Target: 'no_show' -> 1 if Yes
    if "no_show" not in df.columns:
        raise ValueError(
            "Column 'No-show' (normalized to 'no_show') not found in input CSV."
        )
    df["no_show"] = (df["no_show"].astype(str).str.strip().str.upper() == "YES").astype(
        int
    )

    # site_id from Neighbourhood (hashed)
    if "neighbourhood" in df.columns:
        df["site_id"] = (
            df["neighbourhood"]
            .astype(str)
            .apply(lambda x: _hash_label(x, prefix="site"))
        )
    else:
        df["site_id"] = "site0001"

    # patient_pseudo_id
    df["patient_pseudo_id"] = (
        df["patientid"].astype(str) if "patientid" in df.columns else "p0"
    )

    # visit_id
    df["visit_id"] = (
        df["appointmentid"].astype(str)
        if "appointmentid" in df.columns
        else (df.index + 1).astype(str)
    )

    # first visit flag (within dataset)
    df = df.sort_values(["patient_pseudo_id", "date", "visit_id"])
    df["is_first_visit"] = (
        ~df.duplicated(subset=["patient_pseudo_id"], keep="first")
    ).astype(int)

    # visit_type — Kaggle сет не даёт; ставим константу
    df["visit_type"] = "visit"

    # 4) Unified output schema
    out_cols = [
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
    ]
    # Ensure all present
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    out = df[out_cols].dropna(subset=["date"]).copy()
    out["lead_time_days"] = out["lead_time_days"].fillna(0).astype(int)
    out["weekday"] = out["weekday"].astype(int).clip(0, 6)

    # 5) Save
    path_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path_out, index=False)
    # Quick console summary
    print(
        f"Saved: {path_out} (rows={len(out)}, no_show_rate={out['no_show'].mean():.3f})"
    )


if __name__ == "__main__":
    main()
