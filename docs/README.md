# Trailflow

## What is TrailFlow?

TrailFlow is a lightweight, end-to-end system for reducing **missed/late visits** in clinical studies. 

It combines a **calibrated risk model** (predicts no-show probability per visit) with a **business policy layer** that recommends **low-cost actions** (SMS, call, reschedule, voucher) under **budget and operational constraints**. The project ships with a FastAPI **/score** endpoint and a small web UI (Single Visit, Batch, ROI/ENB) for triage and planning. It’s **model-agnostic** (baseline: calibrated logistic regression), **transparent** (clear formulas, versioned decisions), and easy to integrate via **JSON/CSV** -- ideal for pilots, demos, and production hand-offs.

## Why this matters

TrailFlow is useful anywhere missed visits create **regulatory risk, extra cost, or timeline slippage**. It turns calibrated no-show risk into **actionable, budget-aware plans**, so teams spend outreach money where it pays back.

**Who benefits**

* **Sponsors/CROs** - fewer protocol deviations and reworks, more predictable enrollment and **faster data-lock**; ENB/ROI makes budget impact explicit.
* **Sites & clinics** - steadier schedules, fewer last-minute gaps, better staff utilization; simple guidance on **whom to contact and how**.
* **Patient ops/call centers** - focus effort on the **highest-risk** visits, with recommended low-cost actions (SMS/call/reschedule) and clear logs.
* **Telemedicine/home-health partners** - identify candidates for televisits or transport vouchers when they are **most likely** to help.
* **IT/integrators** - stateless **API** and CSV/JSON batch; auditable decisions (probability, uplift, ENB) with versioned models and policies.

**Why it’s valuable**

* Moves from “optimize AUC” to **optimize euros**:
  ${ENB}_{i,a}= L * p_i * uplift_a - cost_a$ --> plan that maximizes total ENB under **budget and caps**.
* **Calibrated probabilities** → trustworthy money estimates and safer decisions.
* **Transparent & governable**: clear formulas, thresholds, and logs suitable for QA and compliance.
* **Small lift to adopt**: works with a single visit or a batch; start with one action (e.g., SMS) and scale to multiple actions when ready.

Use it for T-48/T-24 pre-visit triage, budget-limited outreach campaigns, and pilots that need a measurable **ROI/ENB**.

## Overview

TrailFlow has **two layers** that work together:

1. **Risk estimation.**
   A calibrated model predicts the **no-show probability** for each upcoming visit. You can score a **single visit** (for quick checks) or a **batch** (JSON/CSV) to triage a list and mark the Top-K% highest-risk cases. Because we calibrate the model, the probability (p_i) approximates the true frequency — which lets us convert risk into money.

2. **Business impact & decision policy.**
   Given the loss per missed visit (L), the **cost** and expected **uplift** (risk reduction) of an action, we compute **Expected Net Benefit**: ${ENB}_{i,a}= L * p_i * uplift_a - cost_a$
   We then choose **whom to target** and, optionally, **which action** (SMS/call/voucher/reschedule) to apply so that period ENB is **maximized** under your **budget** and operational caps (contact limits, quiet hours). Outputs include a transparent target plan plus **Spend**, **Benefit**, **ENB**, and **ROI**.

**Why it matters:** we don’t optimize AUC in isolation — we optimize **euro impact**. Accurate, calibrated probabilities drive better targeting; the ENB layer turns those probabilities into concrete savings and a plan you can execute.


## Probability of "no-show" (effective probability)

### Single visit

Score **one** upcoming visit and see its calibrated no-show probability. Useful for quick checks, debugging features, and seeing whether this visit would be selected under a Top-K policy.

**How it works**

1. **Fill the feature fields** (same schema as in Batch):
   `age`, `weekday (0..6)`, `lead_time_days`, `is_first_visit`, `site_id`, `visit_type`,
   `sms_received`, `scholarship`, `hypertension`, `diabetes`, `alcoholism`, `handicap`, `gender (M/F/U)`.
2. (Optional) Set **Top-K% within batch** - used only to show whether this single row would fall into Top-K if it were part of a batch.
3. Click **Score**. The table displays:

   * **Rank** - always 1 for a single row
   * **Probability (no-show)** - calibrated model probability
   * **Bar** - visual risk meter
   * **Target?** - "Top-K" if the probability exceeds the current Top-K threshold
4. Use **Download CSV** to export the scored visit. **Show raw JSON** reveals the exact request/response payload.

**Status & metadata**

* `rows` - always 1; `selected` - 0/1 depending on Top-K;
* `threshold` - probability cutoff implied by the chosen Top-K%;
* `signature/time /model` - request fingerprint and model version;
* **X-API-Key** - include if the API is protected.

**Why this is useful**
Single-visit mode provides a fast, inspectable way to validate inputs, explain one prediction to stakeholders, and check whether a specific visit would be targeted under your current Top-K policy.

**Usage example**

<img width="971" height="761" alt="image" src="https://github.com/user-attachments/assets/bd114090-ee4d-44ab-8ba8-dfefb865a971" />

### Group of visits - Batch (JSON/CSV)

Score a batch of upcoming visits and auto-select the highest-risk cases.

**How it works**

1. **Paste JSON array** of visits or **upload JSON/CSV**.
   For CSV, include a header with feature names (comma or semicolon separated), e.g.:
   `age,weekday,lead_time_days,is_first_visit,site_id,visit_type,sms_received,scholarship,hipertension,diabetes,alcoholism,handicap,gender`
2. Set **Top-K% within batch** (e.g., `30`). The UI will mark the top-K% visits by predicted no-show probability as **Target**.
3. Click **Score**. The table shows for each row:

   * **Rank** - position by risk (descending)
   * **Probability (no-show)** - calibrated model probability
   * **Bar** - visual risk meter
   * **Target?** - whether the row is in the current Top-K
4. Use **Download CSV** to export the scored batch.

**Status & metadata**

* `rows` - total items; `selected` - how many fell into Top-K;
* `threshold` - minimum probability to enter Top-K for this batch;
* `signature/time/model` - request fingerprint and model version;
* **X-API-Key** - add if the API is protected.

**Why this is useful**

Batch mode lets you quickly triage a list of visits, focus outreach on the riskiest cases, and export targets for operations.

**Usage example**

<img width="969" height="894" alt="image" src="https://github.com/user-attachments/assets/36d3d4b2-d951-4f6b-bd88-63ee9c2db331" />

## Business impact

We translate no-show risk into **money** and pick actions that maximize value.

**What we compute**

* For each visit (i) and action (a):

  * Expected **benefit**: $ {Benefit}_{i,a} = L * p_i * uplift_a$
  * **Net benefit** per action: ${ENB}_{i,a} = {Benefit}_{i,a} - cost_a$
    
**Period totals for the selected plan**
Benefit = Σ Benefit_{i,a}
Spend   = Σ cost_a
ENB     = Benefit − Spend
ROI     = Benefit / Spend   (defined if Spend > 0)


**What we optimize**

* Choose **whom to target** and **which action** to apply so that **ENB is maximized** subject to your **budget** and operational caps (e.g., contact limits, quiet hours).
* Two solvers are provided: a simple **Greedy** baseline and an optimal **ILP** formulation (details below).

**Assumptions**

* Probabilities (p_i) are **calibrated** (we monitor ECE).
* Action **uplifts** come from prior evidence and are refined over time (A/B tests or bandits).
* Costs are marginal, per targeted visit.

**Outputs**

* A transparent **target plan** (visit → action) plus **Spend**, **Benefit**, **ENB**, **ROI**, and model/policy versioning.


### Input values (ROI / ENB)

There are parameters required for calculations:

* number of visits in period (**n**)

How many visits are considered in the period? Unit: pcs. (example: 200).

* loss per show (**L**)

Average economic damage from one no-show: disrupted procedures, postponements, additional examinations, etc. Unit: €. (Example: €350).

* no-show probability in the targeted group (**π (pi)**)

The no-show probability for those we plan to target. Range: 0…1. Select \
If π is specified in the calculator, it takes precedence and is used in calculations as the "effective probability" (see p_eff). (Example: 0.50).

* baseline no-show probability (fallback) (**base_p**)

The no-show base frequency in the population/period. \
Used as a fallback when π is not available (or when counting on one's fingers without model probabilities). Range: 0…1. (Not specified in the example).

* relative risk reduction (0..1) (**uplift**)
    
Relative reduction in the risk of no-shows from the chosen intervention. \
If the risk before the intervention is p, then after it is `p * (1 − uplift)`. (Example: 0.35 = 35% risk reduction.)

* intervention cost (€) (**cost**)
    
Cost of one intervention (SMS, call, voucher, etc.). Unit: € per 1 target (example: €6).

* total per Period (€) (**budget**)
    
Total budget for the period (example: €1,200)

* share targeted (0..1) (**k**)

The share of the group of people targeted by the intervention: target_cnt = round(n k). \
If the Optimize k flag is enabled, k is calculated automatically (see below). (In the example, the "auto" field).

* Optimize k (respect budget)

If enabled, the system takes the optimal share k*, taking into account the budget and the rationale for the intervention.

### Output values

To calculate the **Expected Net Benefit (ENB)**, we have two options:

1) **Single Action**
2) **Find Best Action** # TODO: add more opportunities

### Single action

Estimate how many visits to target with one intervention (e.g., SMS) and what the expected business impact is under a budget.

Clicking on the "Compute ENB", you should get something like this:
<img width="967" height="207" alt="image" src="https://github.com/user-attachments/assets/8ad3454c-77d3-4e97-a6d0-40a1e81d93b9" />


#### Per-target economics: ${NB_{per} = L * p * uplift - cost}$

#### Choosing how many to target:
* If $NB_{per}$ ≤ 0 --> don’t target anyone: $k* = 0$.
* Otherwise, respect the budget: $k* = min(1, budget / (cost * n))$.

If you’re using per-visit probabilities, pick the "top-K visits" by $p_i$ first; see the note below.

#### Period totals
* Targeted visits: `targeted = round(n · k*)`
* Spend: `Spend = targeted · cost `
* Benefit saved: `Benefit = L · p · uplift · targeted`
* Expected Net Benefit: `ENB = Benefit − Spend`
* ROI: `ROI = Benefit / Spend (defined if Spend>0)`

#### Why this matters
This converts calibrated risk into money. A positive $NB_{per}$ means the action pays for itself on average; ENB and ROI show total impact under the budget you actually have.

> [!NOTE]
> Per-visit probabilities.
> If you have individual `p_i`, sort visits by `p_i` (or by `ENB_{per(i) = L * p_i * uplift` - cost when cost is constant) and take the top-K until the budget runs out.
> This yields higher ENB than using a flat cohort average.

#### Pseudo code for calculations:
```bash
p_eff = π if π is not None else base_p_or_mean_selected
NB_per = L * p_eff * uplift - cost

if optimize_k:
    k_star = 0 if NB_per <= 0 else min(1, budget / (cost * n))
else:
    k_star = clamp(k, 0, 1)
    # budget restriction
    k_star = min(k_star, budget / (cost * n))

targeted = round(n * k_star)
Spend = targeted * cost
Benefit = L * p_eff * uplift * targeted
ENB = Benefit - Spend
ROI = Benefit / Spend if Spend > 0 else float("inf")
```

### Find Best Action

Among multiple candidate actions (e.g., sms, call, taxi), choose the intervention that maximizes ENB under a budget and operational constraints.

#### Inputs (per action a)
* `cost_a` - cost per targeted visit.
* `uplift_a` - expected relative risk reduction for that action.
* (optional) caps/constraints: max contacts per day, quiet hours, per-site limits, etc.

>[!NOTE]
>You can use data from the given example:
```bash
[
  {"name":"SMS","uplift":0.25,"cost":0.05,"pi":0.40},
  {"name":"Call","uplift":0.45,"cost":6.0,"pi":0.50},
  {"name":"Televisit","uplift":0.60,"cost":12.0,"pi":0.50},
  {"name":"TaxiVoucher","uplift":0.30,"cost":8.0,"pi":0.55}
]
```

#### Per-visit, per-action economics
$$ ENB_{i,a} = L * p_i * uplift_a - cost_a $$

#### Two practical solvers

* **Greedy (simple, fast - great baseline)**

1) For each visit `i`, pick the single action `a*` with the highest positive $ENB_{i,a}$ (or choose to do nothing if all of the ENB for this visit ≤ 0). 
2) Sort all selected pairs (visit i, action a*): \
  a) If different actions have different costs, sort by ENB per 1 euro (i.e., ENB / cost); \
  b) If the cost is the same for all actions (e.g., only SMS), sort by ENB. 
3) Take as many as fit the budget and your limits.
4) **Output**: selected visit–action pairs, Spend, $Benefit = sum(L * p_i * uplift_{a*})$, ENB, ROI.

* **Exact ILP (optimal - use when actions/costs vary a lot)**
  
Idea is get the best plan with having restricted budget and rules/limits. \
Our task here is for each visit, pick at most one action or nothing. To achieve this, we should maximize the sum of the benefits. \
`Maximize Σ ENB_{i,a} * x_{i,a} subject to Σ cost_a * x_{i,a}` <= budget and your operational constraints. Where x_{i,a} ∈ {0,1} -take we action for the visit i or not. \
Solve with an integer linear solver (e.g., OR-Tools). Guarantees the best plan given the inputs.

> [!TIP]
> This method is useful when the prices and effects differ significantly or there are too many restrictions: greedy could make mistakes, but ILP - not.

> [!NOTE]
> Different actions have different costs and effects. Choosing the right action for the right visit increases total benefit without increasing spend.
> The greedy version is a one-file MVP; the ILP version is production-grade when budgets and constraints are tight.
