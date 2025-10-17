"""
reliability_diagram.py — калибровочная диаграмма "до/после" + Brier/ECE

Usage:
  python src/trialflow/reliability_diagram.py \
    --input data/processed/visits.parquet \
    --reports-dir reports \
    --method sigmoid \
    --bins 10

Как интерпретировать результаты:
- Brier score: среднеквадратичная ошибка вероятностей (0 — идеально, 0.25 — случайный прогноз для бинарной задачи)
- ECE (Expected Calibration Error): средневзвешенное |acc - conf| по бинам вероятности (0 — идеально) 
    (см. https://arxiv.org/abs/1706.04599)
- Калибровочная диаграмма: линия идеала — диагональ; чем ближе к ней, тем лучше калибровка
    (см. https://scikit-learn.org/stable/modules/calibration.html#calibration-curves)
    (на диаграмме: "Predicted probability (bin mean)" — средняя предсказанная вероятность в бине;
    "Observed frequency" — доля положительных объектов в бине)

Что видим на графике:
- До калибровки (Uncal) модель может быть плохо откалибрована (линия далеко от диагонали), имея высокие Brier/ECE
- После калибровки (Calibrated-sigmoid или Calibrated-isotonic) линия приближается к диагонали, Brier/ECE уменьшаются
- Разные методы калибровки (sigmoid vs isotonic) могут давать разные результаты в зависимости от данных и модели    

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

# headless-отрисовка (важно для CI/серверов)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """ECE: средневзвешенное |acc - conf| по бинам вероятности."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = (y_prob >= lo) & (y_prob < hi)
        if idx.sum() == 0:
            continue
        acc = y_true[idx].mean()
        conf = y_prob[idx].mean()
        ece += idx.mean() * abs(acc - conf)
    return float(ece)


def make_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-aware split: 70/15/15 по возрастанию даты."""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    n_train = int(0.70 * n)
    n_valid = int(0.85 * n)
    return df.iloc[:n_train], df.iloc[n_train:n_valid], df.iloc[n_valid:]


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int
) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return pd.DataFrame({"bin_pred": prob_pred, "bin_true": prob_true})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/visits.parquet")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--method", choices=["sigmoid", "isotonic"], default="sigmoid")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    data_path = Path(args.input)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1) Данные
    df = pd.read_parquet(data_path)
    train, valid, test = make_splits(df)

    target = "no_show"
    num_cols: List[str] = ["age", "weekday", "lead_time_days"]
    cat_cols: List[str] = [
        "is_first_visit",
        "site_id",
        "visit_type",
        "sms_received",
        "scholarship",
        "hipertension",
        "diabetes",
        "alcoholism",
        "handicap",
        "gender",
    ]

    X_train = train[num_cols + cat_cols]
    y_train = train[target].astype(int).values
    X_valid = valid[num_cols + cat_cols]
    y_valid = valid[target].astype(int).values
    X_test = test[num_cols + cat_cols]
    y_test = test[target].astype(int).values

    # 2) Модель: препроцессинг + логрег
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    pipe = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=300))])
    pipe.fit(X_train, y_train)

    # 3) Предсказания "до калибровки" (на тесте)
    proba_uncal = pipe.predict_proba(X_test)[:, 1]

    # 4) Калибровка Platt/Isotonic на валидации (cv="prefit")
    try:
        # sklearn>=1.6: используем FrozenEstimator, чтобы не переобучать base
        from sklearn.frozen import FrozenEstimator

        cal = CalibratedClassifierCV(estimator=FrozenEstimator(pipe), method="sigmoid")
    except ImportError:
        # совместимость со старыми версиями sklearn
        cal = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv="prefit")

    cal.fit(X_valid, y_valid)

    # 5) Предсказания "после калибровки" (на тесте)
    proba_cal = cal.predict_proba(X_test)[:, 1]

    # 6) Метрики калибровки (Brier/ECE) на тесте
    brier_uncal = brier_score_loss(y_test, proba_uncal)
    brier_cal = brier_score_loss(y_test, proba_cal)
    ece_uncal = expected_calibration_error(y_test, proba_uncal, n_bins=args.bins)
    ece_cal = expected_calibration_error(y_test, proba_cal, n_bins=args.bins)

    # 7) Таблицы reliability
    tab_uncal = reliability_table(y_test, proba_uncal, args.bins)
    tab_cal = reliability_table(y_test, proba_cal, args.bins)
    tab_uncal.to_csv(reports_dir / "reliability_uncal.csv", index=False)
    tab_cal.to_csv(reports_dir / "reliability_calibrated.csv", index=False)

    # 8) Плот
    plt.figure()
    # линия идеала
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    # до/после
    plt.plot(
        tab_uncal["bin_pred"],
        tab_uncal["bin_true"],
        marker="o",
        label=f"Uncal (Brier={brier_uncal:.3f}, ECE={ece_uncal:.3f})",
    )
    plt.plot(
        tab_cal["bin_pred"],
        tab_cal["bin_true"],
        marker="o",
        label=f"Calibrated-{args.method} (Brier={brier_cal:.3f}, ECE={ece_cal:.3f})",
    )
    plt.xlabel("Predicted probability (bin mean)")
    plt.ylabel("Observed frequency")
    plt.title("Reliability diagram (test)")
    plt.legend()
    out_png = reports_dir / "calibration_diagram.png"
    plt.savefig(out_png, bbox_inches="tight", dpi=120)
    plt.close()

    # 9) Короткий summary
    summary = {
        "bins": int(args.bins),
        "method": args.method,
        "test": {
            "brier_uncalibrated": float(brier_uncal),
            "brier_calibrated": float(brier_cal),
            "ece_uncalibrated": float(ece_uncal),
            "ece_calibrated": float(ece_cal),
        },
        "artifacts": {
            "diagram_png": str(out_png),
            "reliability_uncal_csv": str(reports_dir / "reliability_uncal.csv"),
            "reliability_calibrated_csv": str(
                reports_dir / "reliability_calibrated.csv"
            ),
        },
    }
    import json

    with open(reports_dir / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
