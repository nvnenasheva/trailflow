#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROI Simulator for no-show prevention.

Features
- Single-action: скан по k, бюджетам, uplift, стоимости и p (base_p/pi).
- Multi-action: подбор лучшего действия из CSV на сетке бюджетов.
- Отчёты в CSV + (опц.) PNG-графики.
- Использует функции из roi.py (пер-таргет ENB, выбор k*, best action).

Usage examples:

1) Single-action, авто-k, скан по бюджету:
python roi_simulator.py --mode single --n 200 --L 350 --pi 0.5 --uplift 0.35 --cost 6 \
    --budgets 0:4000:250 --optimize_k --out_csv report_single.csv --plots

2) Single-action, ручной скан по k + uplift:
python roi_simulator.py --mode single --n 200 --L 350 --base_p 0.25 --cost 6 \
    --k_grid 0:1:0.05 --uplift_grid 0.1:0.6:0.05 --out_csv report_k_uplift.csv --plots

3) Multi-action из CSV + бюджеты:
python roi_simulator.py --mode multi --n 200 --L 350 --base_p 0.25 \
    --actions_csv actions.csv --budgets 0:4000:250 --optimize_k \
    --out_csv report_multi.csv --plots
"""

import argparse
import math
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# локальный импорт
from roi import (
    per_target_net_benefit,
    expected_net_benefit,
    optimal_k,
    read_actions_csv,
    pick_best_action,
    Action,
)

# ---------- utils ----------

def frange(spec: str) -> List[float]:
    """
    Parse range spec like 'start:stop:step' inclusive of stop if fits grid.
    Examples: '0:1:0.1' -> [0.0, 0.1, ..., 1.0]
              '0:4000:250' -> [0, 250, ..., 4000]
    """
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) != 3:
        raise ValueError("Range must be 'start:stop:step'")
    start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
    if step <= 0:
        raise ValueError("step must be > 0")
    vals = []
    x = start
    # guard for floating rounding
    while x <= stop + 1e-12:
        vals.append(round(x, 10))
        x += step
    return vals


def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# ---------- single-action simulation ----------

def simulate_single(
    n: int,
    L: float,
    # probabilities
    base_p: Optional[float],
    pi: Optional[float],
    # action params
    cost: float,
    uplift: Optional[float],
    # grids
    k_grid: Optional[List[float]],
    budgets: Optional[List[float]],
    uplift_grid: Optional[List[float]],
    cost_grid: Optional[List[float]],
    p_grid: Optional[List[float]],
    # controls
    optimize_k: bool,
) -> pd.DataFrame:
    """
    Returns tidy DataFrame with rows: (mode='single', n, p_eff, uplift, cost, k, budget, pt_nb, ENB, targeted, spend, ROIx)
    """
    rows = []

    # choose effective p supplier
    def p_sources() -> Iterable[float]:
        if p_grid is not None:
            return p_grid
        if pi is not None:
            return [pi]
        if base_p is not None:
            return [base_p]
        raise ValueError("Provide --pi or --base_p, or use --p_grid")

    u_vals = uplift_grid if uplift_grid is not None else [uplift] if uplift is not None else None
    if u_vals is None:
        raise ValueError("Provide --uplift or --uplift_grid")

    c_vals = cost_grid if cost_grid is not None else [cost]

    # two modes: scan by budgets (optimize_k) OR manual k_grid
    if budgets is not None:
        for p_eff in p_sources():
            for u in u_vals:
                for c in c_vals:
                    pt_nb = per_target_net_benefit(p_eff, u, L, c)
                    for B in budgets:
                        k_used = optimal_k(n, pt_nb, c, B) if optimize_k else min(1.0, B / (n * c)) if c > 0 else 1.0
                        enb = expected_net_benefit(n, k_used, p_eff, u, L, c)
                        targeted = int(round(n * k_used))
                        spend = targeted * c
                        roix = (enb + spend) / spend if spend > 0 else math.inf
                        rows.append(
                            dict(mode="single", n=n, L=L, p_eff=p_eff, uplift=u, cost=c,
                                 k=k_used, budget=B, pt_nb=pt_nb, ENB=enb,
                                 targeted=targeted, spend=spend, ROIx=roix)
                        )
    elif k_grid is not None:
        for p_eff in p_sources():
            for u in u_vals:
                for c in c_vals:
                    for k in k_grid:
                        pt_nb = per_target_net_benefit(p_eff, u, L, c)
                        enb = expected_net_benefit(n, k, p_eff, u, L, c)
                        targeted = int(round(n * k))
                        spend = targeted * c
                        roix = (enb + spend) / spend if spend > 0 else math.inf
                        rows.append(
                            dict(mode="single", n=n, L=L, p_eff=p_eff, uplift=u, cost=c,
                                 k=k, budget=np.nan, pt_nb=pt_nb, ENB=enb,
                                 targeted=targeted, spend=spend, ROIx=roix)
                        )
    else:
        raise ValueError("Provide either --budgets or --k_grid for single mode")

    return pd.DataFrame(rows)


# ---------- multi-action simulation ----------

def simulate_multi(
    n: int,
    L: float,
    base_p: Optional[float],
    pi: Optional[float],
    actions: List[Action],
    budgets: List[float],
    optimize_k: bool,
) -> pd.DataFrame:
    """
    Rows: (mode='multi', n, L, budget, best_action, p_eff, uplift, cost, k, ENB, targeted, spend, ROIx)
    """
    rows = []
    for B in budgets:
        best_action, p_eff, pt_nb, k_used, enb_total = pick_best_action(
            actions=actions,
            n=n,
            loss_per_no_show=L,
            cli_base_p=base_p,
            cli_pi=pi,
            k=None,
            budget=B,
            auto_k=optimize_k,
        )
        targeted = int(round(n * k_used))
        spend = targeted * best_action.cost
        roix = (enb_total + spend) / spend if spend > 0 else math.inf

        rows.append(
            dict(mode="multi", n=n, L=L, budget=B, best_action=best_action.name,
                 p_eff=p_eff, uplift=best_action.uplift, cost=best_action.cost,
                 k=k_used, pt_nb=pt_nb, ENB=enb_total, targeted=targeted, spend=spend, ROIx=roix)
        )
    return pd.DataFrame(rows)


# ---------- plotting ----------

def plot_single_budget(df: pd.DataFrame, out_prefix: str):
    # ENB vs Budget (one curve per uplift/cost combo)
    pivot_cols = ["p_eff", "uplift", "cost"]
    for key, dsub in df.groupby(pivot_cols):
        plt.figure()
        dsub_sorted = dsub.sort_values("budget")
        plt.plot(dsub_sorted["budget"], dsub_sorted["ENB"])
        plt.xlabel("Budget (€)")
        plt.ylabel("ENB (€ / period)")
        plt.title(f"ENB vs Budget | p={key[0]:.2f}, uplift={key[1]:.2f}, cost={key[2]:.2f}")
        fname = f"{out_prefix}_enb_budget_p{key[0]:.2f}_u{key[1]:.2f}_c{key[2]:.2f}.png".replace(".", "_")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

def plot_single_k(df: pd.DataFrame, out_prefix: str):
    # ENB vs k (one curve per uplift/cost/p combo)
    pivot_cols = ["p_eff", "uplift", "cost"]
    for key, dsub in df.groupby(pivot_cols):
        plt.figure()
        dsub_sorted = dsub.sort_values("k")
        plt.plot(dsub_sorted["k"], dsub_sorted["ENB"])
        plt.xlabel("k (share targeted)")
        plt.ylabel("ENB (€ / period)")
        plt.title(f"ENB vs k | p={key[0]:.2f}, uplift={key[1]:.2f}, cost={key[2]:.2f}")
        fname = f"{out_prefix}_enb_k_p{key[0]:.2f}_u{key[1]:.2f}_c{key[2]:.2f}.png".replace(".", "_")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

def plot_multi_budget(df: pd.DataFrame, out_prefix: str):
    # ENB vs Budget with action labels
    plt.figure()
    dsub = df.sort_values("budget")
    plt.plot(dsub["budget"], dsub["ENB"])
    plt.xlabel("Budget (€)")
    plt.ylabel("ENB (€ / period)")
    plt.title("Best Action ENB vs Budget")
    fname = f"{out_prefix}_multi_enb_budget.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

    # Best action vs Budget (step-like)
    plt.figure()
    # encode actions as integers for simple line; also save legend mapping
    actions = {a: i for i, a in enumerate(dsub["best_action"].unique(), start=1)}
    y = dsub["best_action"].map(actions)
    plt.plot(dsub["budget"], y)
    plt.yticks(list(actions.values()), list(actions.keys()))
    plt.xlabel("Budget (€)")
    plt.ylabel("Best Action")
    plt.title("Best Action by Budget")
    fname = f"{out_prefix}_multi_best_action_by_budget.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="ROI Simulator")
    ap.add_argument("--mode", choices=["single", "multi"], required=True)

    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--L", type=float, required=True, help="Loss per no-show (€/visit)")

    # probabilities
    ap.add_argument("--base_p", type=float, help="Baseline no-show prob 0..1")
    ap.add_argument("--pi", type=float, help="Precision in targeted 0..1")
    ap.add_argument("--p_grid", type=str, help="Range 'start:stop:step' for p")

    # action
    ap.add_argument("--uplift", type=float, help="Relative risk reduction 0..1")
    ap.add_argument("--cost", type=float, help="Intervention cost €")

    # grids
    ap.add_argument("--k_grid", type=str, help="Range for k, e.g. '0:1:0.05'")
    ap.add_argument("--budgets", type=str, help="Range for budgets €, e.g. '0:4000:250'")
    ap.add_argument("--uplift_grid", type=str, help="Range for uplift, e.g. '0.1:0.6:0.05'")
    ap.add_argument("--cost_grid", type=str, help="Range for cost, e.g. '1:10:1'")

    # multi
    ap.add_argument("--actions_csv", type=str, help="CSV: name,uplift,cost[,pi][,base_p]")

    # controls
    ap.add_argument("--optimize_k", action="store_true", help="Use optimal k with budget cap")
    ap.add_argument("--out_csv", type=str, default="roi_report.csv")
    ap.add_argument("--plots", action="store_true", help="Save PNG charts")
    args = ap.parse_args()

    # parse grids
    k_grid = frange(args.k_grid) if args.k_grid else None
    budgets = frange(args.budgets) if args.budgets else None
    u_grid = frange(args.uplift_grid) if args.uplift_grid else None
    c_grid = frange(args.cost_grid) if args.cost_grid else None
    p_grid = frange(args.p_grid) if args.p_grid else None

    if args.n <= 0: raise ValueError("n must be > 0")
    if args.L < 0: raise ValueError("L must be ≥ 0")
    for name, v in [("base_p", args.base_p), ("pi", args.pi)]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0,1]")

    if args.mode == "single":
        if (args.uplift is None and u_grid is None) or (args.cost is None and c_grid is None):
            raise ValueError("Single mode needs --uplift/--uplift_grid and --cost/--cost_grid")

        df = simulate_single(
            n=args.n, L=args.L,
            base_p=args.base_p, pi=args.pi,
            cost=args.cost if args.cost is not None else 0.0,
            uplift=args.uplift,
            k_grid=k_grid, budgets=budgets,
            uplift_grid=u_grid, cost_grid=c_grid, p_grid=p_grid,
            optimize_k=args.optimize_k,
        )

        ensure_dir(args.out_csv)
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv} ({len(df)} rows)")

        if args.plots:
            prefix = os.path.splitext(args.out_csv)[0]
            if budgets is not None:
                plot_single_budget(df, prefix)
            if k_grid is not None:
                plot_single_k(df, prefix)
            print("Saved plots.")

    else:
        if not args.actions_csv:
            raise ValueError("Multi mode requires --actions_csv")
        if budgets is None:
            raise ValueError("Multi mode requires --budgets")
        actions = read_actions_csv(args.actions_csv)
        df = simulate_multi(
            n=args.n, L=args.L,
            base_p=args.base_p, pi=args.pi,
            actions=actions,
            budgets=budgets,
            optimize_k=args.optimize_k,
        )
        ensure_dir(args.out_csv)
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv} ({len(df)} rows)")

        if args.plots:
            prefix = os.path.splitext(args.out_csv)[0]
            plot_multi_budget(df, prefix)
            print("Saved plots.")


if __name__ == "__main__":
    main()
