from pathlib import Path
import json

def test_reports_exist():
    for p in ["reports/metrics.csv", "reports/business_valid.csv", "reports/business_test.csv", "reports/summary.json"]:
        assert Path(p).exists(), f"missing {p}"

def test_summary_schema():
    js = json.loads(Path("reports/summary.json").read_text(encoding="utf-8"))
    assert "best_policy" in js and "K_percent" in js["best_policy"]
    assert "metrics" in js and "valid" in js["metrics"] and "test" in js["metrics"]
