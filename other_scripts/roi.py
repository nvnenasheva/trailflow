#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROI / ENB calculator for no-show prevention.

Core model:
  ENB = n * k * (p_eff * uplift * L - cost_intervention)

Where:
  n              — число визитов в рассматриваемом периоде
  k (0..1)       — доля визитов, на которые применяем действие (top-K)
  p_eff (0..1)   — базовая вероятность no-show среди отобранных (точность отбора, π)
                   Если π не задана, используется base_p (средняя по популяции)
  uplift (0..1)  — относительное снижение риска no-show от действия
  L              — потери от no-show (€/визит)
  cost_intervention — стоимость действия (€/визит)

Дополнительно:
  - Поддержка бюджета B (€): n*k*cost_intervention <= B
  - Автовыбор k: если пер-таргет ENB > 0 → k = min(1, B/(n*cost)), иначе 0
  - Порог окупаемости по uplift: uplift* > cost / (p_eff * L)
  - Порог окупаемости по π:      π*      > cost / (uplift * L)
  - Режим сравнения нескольких действий из CSV (name,uplift,cost[,pi][,base_p])
"""

import argparse
import csv
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class Action:
    name: str
    uplift: float                # 0..1
    cost: float                  # €/visit
    pi: Optional[float] = None   # precision among selected (0..1)
    base_p: Optional[float] = None  # fallback if pi isn't provided

    def effective_p(self, cli_base_p: Optional[float], cli_pi: Optional[float]) -> float:
        """
        Priority: self.pi -> cli_pi -> self.base_p -> cli_base_p
        """
        for v in (self.pi, cli_pi, self.base_p, cli_base_p):
            if v is not None:
                return v
        raise ValueError("No probability provided: specify --pi or --base_p (or columns pi/base_p in CSV).")


# ---------- Core math ----------

def per_target_net_benefit(p_eff: float, uplift: float, loss_per_no_show: float, cost_intervention: float) -> float:
    """
    Net benefit for ONE targeted visit.
    """
    return p_eff * uplift * loss_per_no_show - cost_intervention


def expected_net_benefit(
    n: int,
    k: float,
    p_eff: float,
    uplift: float,
    loss_per_no_show: float,
    cost_intervention: float,
) -> float:
    """
    ENB = n * k * (p_eff * uplift * L - c)
    """
    return n * k * per_target_net_benefit(p_eff, uplift, loss_per_no_show, cost_intervention)


def optimal_k(
    n: int,
    pt_nb: float,                # per-target net benefit
    cost_intervention: float,
    budget: Optional[float] = None
) -> float:
    """
    If per-target NB <= 0 → k=0.
    Else k is capped by budget (if provided), otherwise 1.
    """
    if n <= 0:
        return 0.0
    if pt_nb <= 0:
        return 0.0
    k_cap = 1.0
    if budget is not None:
        if cost_intervention <= 0:
            # zero or negative cost → treat as uncapped by budget (but still ≤1)
            k_cap = 1.0
        else:
            k_cap = min(1.0, max(0.0, budget / (n * cost_intervention)))
    return k_cap


def clamp01(x: float, name: str) -> float:
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"{name} must be in [0,1], got {x}")
    return x


# ---------- CSV utilities ----------

def read_actions_csv(path: str) -> List[Action]:
    actions: List[Action] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"name", "uplift", "cost"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

        for i, row in enumerate(reader, start=2):  # header is line 1
            try:
                name = (row.get("name") or "").strip()
                uplift = float(row["uplift"])
                cost = float(row["cost"])
                pi = row.get("pi")
                base_p = row.get("base_p")
                pi_val = float(pi) if pi not in (None, "",) else None
                base_p_val = float(base_p) if base_p not in (None, "",) else None

                if not name:
                    raise ValueError("empty name")
                clamp01(uplift, "uplift")
                if pi_val is not None:
                    clamp01(pi_val, "pi")
                if base_p_val is not None:
                    clamp01(base_p_val, "base_p")
                if cost < 0:
                    raise ValueError("cost must be ≥ 0")

                actions.append(Action(name=name, uplift=uplift, cost=cost, pi=pi_val, base_p=base_p_val))
            except Exception as e:
                raise ValueError(f"CSV parse error at line {i}: {e}")
    return actions


def pick_best_action(
    actions: List[Action],
    n: int,
    loss_per_no_show: float,
    cli_base_p: Optional[float],
    cli_pi: Optional[float],
    k: Optional[float],
    budget: Optional[float],
    auto_k: bool,
) -> Tuple[Action, float, float, float, float]:
    """
    Returns: (best_action, p_eff, pt_nb, k_used, enb_total)
    """
    if not actions:
        raise ValueError("No actions provided.")

    best: Tuple[Optional[Action], float, float, float, float] = (None, 0, -math.inf, 0, -math.inf)
    for a in actions:
        p_eff = a.effective_p(cli_base_p, cli_pi)
        pt_nb = per_target_net_benefit(p_eff, a.uplift, loss_per_no_show, a.cost)
        k_used = optimal_k(n, pt_nb, a.cost, budget) if auto_k or k is None else clamp01(k, "k")
        enb_total = expected_net_benefit(n, k_used, p_eff, a.uplift, loss_per_no_show, a.cost)
        if enb_total > best[4]:
            best = (a, p_eff, pt_nb, k_used, enb_total)

    return best  # type: ignore


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Expected Net Benefit (ENB) calculator")
    ap.add_argument("--n", type=int, required=True, help="Number of visits in the period")

    # Either provide k or ask to optimize it (optionally with budget)
    ap.add_argument("--k", type=float, help="Share of targeted visits in [0,1]")
    ap.add_argument("--optimize_k", action="store_true", help="Choose k automatically (uses budget if provided)")
    ap.add_argument("--budget", type=float, help="Budget in € to cap n*k*cost")

    # Probabilities
    ap.add_argument("--base_p", type=float, help="Baseline no-show prob in population (0..1)")
    ap.add_argument("--pi", type=float, help="Precision among selected (no-show prob in targeted group, 0..1)")

    # Loss & single-action params
    ap.add_argument("--L", type=float, required=True, help="Loss per no-show (€/visit), i.e., cost_per_no_show")
    ap.add_argument("--uplift", type=float, help="Uplift (relative risk reduction 0..1) for single-action mode")
    ap.add_argument("--cost", type=float, help="Intervention cost (€/visit) for single-action mode")

    # Multi-action mode
    ap.add_argument("--actions_csv", type=str, help="CSV with columns: name,uplift,cost[,pi][,base_p]")

    args = ap.parse_args()

    # Validate core scalars
    if args.n < 0:
        raise ValueError("n must be ≥ 0")
    if args.budget is not None and args.budget < 0:
        raise ValueError("budget must be ≥ 0")
    if args.L < 0:
        raise ValueError("L (loss per no-show) must be ≥ 0")
    if args.base_p is not None:
        clamp01(args.base_p, "base_p")
    if args.pi is not None:
        clamp01(args.pi, "pi")
    if args.k is not None:
        clamp01(args.k, "k")

    # Decide mode
    multi_mode = args.actions_csv is not None

    if multi_mode:
        actions = read_actions_csv(args.actions_csv)
        best_action, p_eff, pt_nb, k_used, enb_total = pick_best_action(
            actions=actions,
            n=args.n,
            loss_per_no_show=args.L,
            cli_base_p=args.base_p,
            cli_pi=args.pi,
            k=args.k,
            budget=args.budget,
            auto_k=args.optimize_k,
        )

        # Thresholds for the chosen action
        uplift_star = None
        pi_star = None
        if p_eff > 0 and args.L > 0:
            uplift_star = best_action.cost / (p_eff * args.L)
        if best_action.uplift > 0 and args.L > 0:
            pi_star = best_action.cost / (best_action.uplift * args.L)

        targeted = int(round(args.n * k_used))
        total_cost = targeted * best_action.cost
        roi_ratio = (enb_total + total_cost) / total_cost if total_cost > 0 else float("inf")

        print(f"Mode: MULTI-ACTION (best of {len(actions)})")
        print(f"Best action: {best_action.name}")
        print(f"Inputs: n={args.n}, k*={k_used:.4f} (targeted ≈ {targeted}), budget={args.budget or 0:.2f}€")
        print(f"Assumptions: p_eff={p_eff:.4f}, uplift={best_action.uplift:.4f}, L={args.L:.2f}€, cost={best_action.cost:.2f}€")
        print(f"Per-target NB: {pt_nb:.2f}€")
        print(f"ENB (€/period): {enb_total:.2f}")
        print(f"Spend (€/period): {total_cost:.2f} → ROI (benefit/cost): {roi_ratio:.2f}x")
        if uplift_star is not None:
            print(f"Break-even uplift*: {uplift_star:.4f}")
        if pi_star is not None:
            print(f"Break-even π*:       {pi_star:.4f}")

    else:
        # Single-action mode requires uplift and cost
        if args.uplift is None or args.cost is None:
            raise ValueError("In single-action mode, provide both --uplift and --cost.")
        clamp01(args.uplift, "uplift")
        if args.cost < 0:
            raise ValueError("cost must be ≥ 0")

        # Effective p: prefer π if provided, otherwise base_p
        if args.pi is not None:
            p_eff = args.pi
        elif args.base_p is not None:
            p_eff = args.base_p
        else:
            raise ValueError("Provide either --pi or --base_p.")

        pt_nb = per_target_net_benefit(p_eff, args.uplift, args.L, args.cost)

        if args.optimize_k or args.k is None:
            k_used = optimal_k(args.n, pt_nb, args.cost, args.budget)
        else:
            k_used = args.k

        enb_total = expected_net_benefit(args.n, k_used, p_eff, args.uplift, args.L, args.cost)

        targeted = int(round(args.n * k_used))
        total_cost = targeted * args.cost
        roi_ratio = (enb_total + total_cost) / total_cost if total_cost > 0 else float("inf")

        # Break-even thresholds
        uplift_star = args.cost / (p_eff * args.L) if (p_eff > 0 and args.L > 0) else None
        pi_star = args.cost / (args.uplift * args.L) if (args.uplift > 0 and args.L > 0) else None

        print("Mode: SINGLE-ACTION")
        print(f"Inputs: n={args.n}, k*={k_used:.4f} (targeted ≈ {targeted}), budget={args.budget or 0:.2f}€")
        print(f"Assumptions: p_eff={p_eff:.4f}, uplift={args.uplift:.4f}, L={args.L:.2f}€, cost={args.cost:.2f}€")
        print(f"Per-target NB: {pt_nb:.2f}€")
        print(f"ENB (€/period): {enb_total:.2f}")
        print(f"Spend (€/period): {total_cost:.2f} → ROI (benefit/cost): {roi_ratio:.2f}x")
        if uplift_star is not None:
            print(f"Break-even uplift*: {uplift_star:.4f}")
        if pi_star is not None:
            print(f"Break-even π*:       {pi_star:.4f}")


if __name__ == "__main__":
    main()
