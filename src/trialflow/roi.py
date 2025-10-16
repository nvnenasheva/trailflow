import argparse


def expected_net_benefit(
    n: int,  # план визитов
    k: float,  # доля таргетируемых (0..1)
    base_p: float,  # базовая доля no-show (0..1)
    uplift: float,  # относительное снижение no-show в таргетируемой группе (0..1)
    cost_per_no_show: float,
    cost_intervention: float,
) -> float:
    """
    ENB = prevented_no_shows * cost_per_no_show - targeted * cost_intervention
    prevented_no_shows = n * k * base_p * uplift
    targeted = n * k
    """
    prevented = n * k * base_p * uplift
    return prevented * cost_per_no_show - (n * k) * cost_intervention


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=float, required=True)
    ap.add_argument("--base_p", type=float, required=True)
    ap.add_argument("--uplift", type=float, required=True)
    ap.add_argument("--c", type=float, required=True, help="cost_per_no_show")
    ap.add_argument("--cost", type=float, required=True, help="cost_intervention")
    args = ap.parse_args()
    enb = expected_net_benefit(
        args.n, args.k, args.base_p, args.uplift, args.c, args.cost
    )
    print(f"ENB (€/период): {enb:.2f}")
