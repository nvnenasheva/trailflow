import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# from __future__ import annotations


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = (y_prob >= lo) & (y_prob < hi)
        if idx.sum() == 0:
            continue
        acc = y_true[idx].mean()
        conf = y_prob[idx].mean()
        ece += (idx.mean()) * abs(acc - conf)
    return float(ece)


def business_table(
    y_true: np.ndarray,
    proba: np.ndarray,
    topk_list: list[int],
    cost_no_show: float,
    cost_intervention: float,
    uplift_rate: float,
) -> pd.DataFrame:
    n = len(y_true)
    order = np.argsort(-proba)
    y_sorted = y_true[order]
    p_sorted = proba[order]
    rows = []
    total_pos = int(y_true.sum())

    for K in topk_list:
        m = max(1, int(n * (K / 100.0)))
        y_top = y_sorted[:m]
        p_top = p_sorted[:m]

        recall_at_k = (y_top.sum() / total_pos) if total_pos > 0 else 0.0
        ppv_at_k = float(y_top.mean())
        exp_no_shows_in_target = float(p_top.sum())

        benefit = uplift_rate * exp_no_shows_in_target * cost_no_show
        cost = m * cost_intervention
        enb = float(benefit - cost)

        rows.append(
            dict(
                K_percent=K,
                Targeted_n=m,
                Recall_at_K=recall_at_k,
                PPV_at_K=ppv_at_k,
                Exp_no_shows_in_target=exp_no_shows_in_target,
                ENB_euros=enb,
            )
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/visits.parquet")
    ap.add_argument("--model", default="models/baseline_calibrated.pkl")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--cost-no-show", type=float, default=200.0)
    ap.add_argument("--cost-intervention", type=float, default=1.5)
    ap.add_argument("--uplift", type=float, default=0.30)
    args = ap.parse_args()

    data_path = Path(args.input)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Данные
    df = pd.read_parquet(data_path).sort_values("date").reset_index(drop=True)

    target = "no_show"
    num_cols = ["age", "weekday", "lead_time_days"]
    cat_cols = [
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

    X = df[num_cols + cat_cols]
    y = df[target].astype(int).values

    # 2) Time-aware split 70/15/15
    n = len(df)
    n_train = int(0.70 * n)
    n_valid = int(0.85 * n)
    X_train, y_train = X.iloc[:n_train], y[:n_train]
    X_valid, y_valid = X.iloc[n_train:n_valid], y[n_train:n_valid]
    X_test, y_test = X.iloc[n_valid:], y[n_valid:]

    # 3) Пайплайн: препроцессинг + логрег
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    pipe = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=300))])
    pipe.fit(X_train, y_train)

    # 4) Калибровка вероятностей (Platt) на валидации
    """ модель выдает скоры (0..1), но они не обязаны быть настояшими вероятностями.
    калибровка нужна, чтобы скор соответствовал вероятности события (no-show) => k
    делаем надстройку, чтобы перевести скор в вероятность так, чтобы среди объектов с прогнозом ~p доля положительных ≈ p """

    # base уже обучен на train
    # подгоняем калибратор на валидации (Platt sigmoid)
    # затем на любых данных можно вызывать cal.predict_proba() для получения вероятностей -- на валидации, тесте

    # base_estimator VS estimator
    try:
        # sklearn>=1.6: используем FrozenEstimator, чтобы не переобучать pipe
        from sklearn.frozen import FrozenEstimator

        cal = CalibratedClassifierCV(estimator=FrozenEstimator(pipe), method="sigmoid")
    except ImportError:
        # совместимость со старыми версиями sklearn
        cal = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv="prefit")

    cal.fit(X_valid, y_valid)

    proba_valid = cal.predict_proba(X_valid)[:, 1]
    proba_test = cal.predict_proba(X_test)[:, 1]

    """Почему это важно для бизнеса?
    - ML-метрики (ROC AUC, PR AUC) не зависят от калибровки, т.к. ранжируют объекты
    - но бизнес-метрики (ENB, Recall@K, PPV@K) мы считаем, выбирая топ-K% объектов по вероятности no-show
    - если вероятности плохо откалиброваны, то топ-K% может быть не самым оптимальным выбором для бизнеса
    - поэтому калибровка важна для достижения лучших бизнес-результатов """

    # 5) ML-метрики
    metrics = {
        ("valid", "roc_auc"): roc_auc_score(
            y_valid, proba_valid
        ),  # измеряет, насколько хорошо модель ранжирует объекты, чем выше — тем лучше (не зависит от порога классификации)
        ("valid", "pr_auc"): average_precision_score(
            y_valid, proba_valid
        ),  # полезна при дисбалансе классов, чем выше — тем лучше (ловит редкие no-show)
        ("valid", "brier"): brier_score_loss(
            y_valid, proba_valid
        ),  # среднеквадратичная ошибка вероятностей, чем ниже — тем лучше
        ("valid", "ece"): expected_calibration_error(
            y_valid, proba_valid
        ),  # насколько хорошо откалиброваны вероятности, чем ниже — тем лучше
        ("test", "roc_auc"): roc_auc_score(y_test, proba_test),
        ("test", "pr_auc"): average_precision_score(y_test, proba_test),
        ("test", "brier"): brier_score_loss(y_test, proba_test),
        ("test", "ece"): expected_calibration_error(y_test, proba_test),
    }
    # используем явные имена для уровней
    s = pd.Series(metrics)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["split", "metric"])
    metrics_df = s.unstack(
        "split"
    ).reset_index()  # гарантированно есть колонка 'metric'

    # 6) Бизнес-метрики
    """ У нас есть бюджет/ресурс (например, отправка SMS или звонок), который ограничен, поэтому будем таргетировать на самых рискованных на таргетирование K% пациентов.
    Для разных K (5, 10, 15, ..., 50):
    сортируем пациентов по убыванию вероятности no-show,
    выбираем верхние топ-K% как таргетируемые,
    
    считаем бизнес-метрики: 
        - ENB (ожидаемая чистая выгода), 
        - Recall@K (какую долю всех no-show мы поймали среди таргетируемых),
        - PPV@K (точность таргетирования среди таргетируемых -- сколько таргетируемых реально оказались no-show) """

    topk = list(range(5, 55, 5))
    # uplift_rate -- оценка, насколько интервенция (напоминание/телевизит/ваучер) снижает no-show, напр. 30%
    biz_valid = business_table(
        y_valid,
        proba_valid,
        topk,
        args.cost_no_show,
        args.cost_intervention,
        args.uplift,
    )
    biz_test = business_table(
        y_test, proba_test, topk, args.cost_no_show, args.cost_intervention, args.uplift
    )
    biz_valid_csv = reports_dir / "business_valid.csv"
    biz_test_csv = reports_dir / "business_test.csv"
    biz_valid.to_csv(biz_valid_csv, index=False)
    biz_test.to_csv(biz_test_csv, index=False)

    # выбрать K по максимуму ENB на валидации и показать качество на тесте
    best_k = int(biz_valid.loc[biz_valid["ENB_euros"].idxmax(), "K_percent"])
    row_test = biz_test[biz_test["K_percent"] == best_k].iloc[0].to_dict()

    # 7) ROI-кривая
    plt.figure()
    plt.plot(biz_valid["K_percent"], biz_valid["ENB_euros"], marker="o", label="valid")
    plt.plot(biz_test["K_percent"], biz_test["ENB_euros"], marker="o", label="test")
    plt.xlabel("Target budget K (%)")
    plt.ylabel("Expected Net Benefit (€)")
    plt.title("ROI curve — ENB vs Top-K%")
    plt.legend()
    roi_path = reports_dir / "roi_curve.png"
    plt.savefig(roi_path, bbox_inches="tight")
    plt.close()

    # 8) Сохранить модель и summary
    joblib.dump(cal, model_path)

    metrics_csv = reports_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    valid_dict = metrics_df.set_index("metric")["valid"].astype(float).to_dict()
    test_dict = metrics_df.set_index("metric")["test"].astype(float).to_dict()

    summary = {
        "counts": {
            "train": int(len(X_train)),
            "valid": int(len(X_valid)),
            "test": int(len(X_test)),
        },
        "metrics": {"valid": valid_dict, "test": test_dict},
        "best_policy": {
            "K_percent": best_k,
            "ENB_test": float(row_test["ENB_euros"]),
            "Recall@K_test": float(row_test["Recall_at_K"]),
            "PPV@K_test": float(row_test["PPV_at_K"]),
        },
        "artifacts": {
            "model": str(model_path),
            "metrics_csv": str(metrics_csv),
            "biz_valid_csv": str(biz_valid_csv),
            "biz_test_csv": str(biz_test_csv),
            "roi_plot": str(roi_path),
        },
        "assumptions": {
            "COST_NO_SHOW": args.cost_no_show,
            "COST_INTERVENTION": args.cost_intervention,
            "UPLIFT_RATE": args.uplift,
        },
    }

    with open(reports_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    """ expected_no_shows_in_target = Σ калиброванных вероятностей в таргете
        prevented = uplift_rate × expected_no_shows_in_target
        benefit   = prevented × COST_NO_SHOW
        cost      = (кол-во таргетируемых) × COST_INTERVENTION
        ENB       = benefit − cost
    """


if __name__ == "__main__":
    main()

"""
1. Как выбирается K% для таргетирования и почему? 
На валидации выбираем K, при котором ENB максимальна (то есть лучшая отдача в евро), чтобы оптимизировать бизнес-результат.
Затем этот K фиксируем и применяем на тесте, чтобы оценить реальную эффективность.

2. Почему важна калибровка вероятностей для бизнес-метрик?
ML-метрики (ROC AUC, PR AUC) не зависят от калибровки, так как они оценивают ранжирование объектов.
Однако бизнес-метрики (ENB, Recall@K, PPV@K) рассчитываются на основе выбора топ-K% объектов по вероятности no-show.
Если вероятности плохо откалиброваны, то топ-K% может не соответствовать оптимальному выбору для бизнеса. Поэтому калибровка важна для достижения лучших бизнес-результатов.

3. Как интерпретировать ENB?
ENB (Expected Net Benefit) измеряет ожидаемую чистую выгоду от таргетирования пациентов с высоким риском no-show.
Положительное значение ENB означает, что выгода от предотвращенных no-show превышает затраты на интервенции, что свидетельствует о рентабельности стратегии таргетирования.

4. Как вообще работает калибровка Platt?
Калибровка Platt использует логистическую регрессию для преобразования сырых скорингов модели в откалиброванные вероятности.
Это парметрическая S-образная функция, которая подгоняется на отложенной выборке (валидации), чтобы скоринговые значения соответствовали истинным вероятностям событий.
p_cal = 1 / (1 + exp(A * score + B)), где A и B — параметры, обученные на валидационной выборке, score - сырые выходы модели (predict_proba for LogisticRegression).
A и B подбираются так, чтобы предсказания совпали с реальными вероятностями.
Cумма калиброванных вероятностей по группе (например, Top-K) — это и есть ожидаемое число no-show в этой группе. 
Именно эту сумму мы видим в Exp_no_shows_in_target и используем в ENB.

То есть в ьизнес-метриках у нас будет:
    S = сумма p_cal в таргете  → ожидаемое число no-show
    предотвращено ≈ uplift × S
    выгода = (uplift × S) × COST_NO_SHOW

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Toy example:

Пусть у нас 1000 пациентов, из них 200 обычно не приходят (no-show rate = 20%).
Сумма калиброванных вероятностей по всем визитам, которые мы будем таргетировать.
То есть:
    модель дает (после калибровки) для каждого визита вероятность p_i (например, 0.18, 0.27..),
    берем top-K% самых рискованных визитов -- это будет m штук,
    считаем сумму (S) вероятностей этих m визитов.
Эта полученная сумма и есть ожидаемое количество no-show в этой группе. 
Пусть средняя калиброванная вероятность пропуска внутри таргетируемой группы равна 20%, тогда S = 200 * 0.2 = 40. Значит  ожидается 40 no-show среди таргетируемых.
Если uplift_rate = 0.30, то предотвращенных no-show = 40 * 0.30 = 12.
Если стоимость пропуска COST_NO_SHOW = 200€, то выгода = 12 * 200€ = 2400€.
Цена интервенции COST_INTERVENTION = 1.5€, если мы таргетируем 200 пациентов (K=20%), то затраты = 200 * 1.5€ = 300€.
Тогда ENB = 2400€ - 300€ = 2100€, что означает значительную чистую выгоду от таргетирования этой группы пациентов.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Note:
Когда выбрать isotonic вместо sigmoid:
1. Валидации много (десятки тысяч записей) и видно, что S-образной кривой мало — пробуй method="isotonic".
2. Если валидация маленькая/шумная — оставайся на sigmoid (стабильнее).

"""


"""
train_baseline.py -- time-aware baseline + business metrics

Usage:
  python src/trialflow/train_baseline.py \
    --input data/processed/visits.parquet \
    --model models/baseline_calibrated.pkl \
    --reports-dir reports \
    --cost-no-show 200 --cost-intervention 1.5 --uplift 0.30

Idea:
- Create a time-aware baseline model to predict no-shows
- Evaluate the model using both ML and business metrics

Main steps
1) Load data (data/processed/visits.parquet)
2) Time-aware split 70/15/15
3) Preprocessing + Logistic Regression baseline
4) Calibration (Platt) on validation
5) ML metrics: ROC AUC, PR AUC, Brier, ECE
6) Business metrics: ENB, Recall@K, PPV@K
7) ROI curve
8) Save model, metrics, summary (JSON)

Assumptions:
- cost_no_show: e.g. 200€
- cost_intervention: e.g. 1.5€ (SMS, call)
- uplift: e.g. 0.30 (30% relative reduction of no-show in targeted group)
"""

"""
Признаки:
- числовые: age, weekday, lead_time_days;
- категориальные: is_first_visit, site_id, visit_type, sms_received,
  scholarship, hipertension, diabetes, alcoholism, handicap, gender

  Делаем timne-aware split 70/15/15 по дате визита:
- train: 70% первых визитов
- valid: следующие 15%
- test: последние 15%

Модель: логрег с L2, max_iter=300
Калибровка: Platt (CalibratedClassifierCV(method="sigmoid"))
"""
# from __future__ import annotations