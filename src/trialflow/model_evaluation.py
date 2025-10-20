# python model_evaluation.py --model models/model.skops --save_csv reports/coef_or.csv
import argparse, joblib, numpy as np, re
from pathlib import Path

def load_model(path):
    if str(path).endswith(".skops"):
        from skops.io import load as skops_load
        return skops_load(path, trusted=True)
    return joblib.load(path)

def strip_prefix(name: str) -> str:
    # 'preprocess__onehot__gender_F' → 'gender_F'
    return re.sub(r"^[^_]+__([^_]+__)?", "", name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to fitted sklearn Pipeline or LogisticRegression")
    ap.add_argument("--save_csv", default="coef_or.csv")
    args = ap.parse_args()

    model = load_model(args.model)
    # Expect a Pipeline(preprocess, clf) OR just LogisticRegression
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf") or model.named_steps.get("model")
        pre = model.named_steps.get("preprocess") or model.named_steps.get("prep")
        if pre is None:
            raise SystemExit("Can't find 'preprocess' step in pipeline")
        if hasattr(pre, "get_feature_names_out"):
            # pass a dummy list if needed
            try:
                names = pre.get_feature_names_out()
            except TypeError:
                names = pre.get_feature_names_out(None)
        else:
            raise SystemExit("preprocess.get_feature_names_out() not available. Upgrade sklearn or compute names manually.")
    else:
        clf, pre, names = model, None, None

    if hasattr(clf, "coef_"):
        beta = clf.coef_.ravel()  # binary LR
        intercept = float(clf.intercept_)
    else:
        raise SystemExit("Classifier does not have coef_/intercept_ (is it LogisticRegression?)")

    if pre is not None and names is not None:
        feat_names = [strip_prefix(n) for n in names]
    else:
        feat_names = [f"x{i}" for i in range(len(beta))]

    # build table
    rows = []
    rows.append(("Intercept", intercept, np.exp(intercept)))
    for n, b in zip(feat_names, beta):
        rows.append((n, float(b), float(np.exp(b))))

    # save csv
    import csv
    with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "beta", "odds_ratio"])
        w.writerows(rows)

    print(f"Saved {args.save_csv}. Top 10:")
    for r in rows[:10]:
        print(r)

if __name__ == "__main__":
    main()
