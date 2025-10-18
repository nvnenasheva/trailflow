# TrialFlow — минимальная схема

```mermaid
flowchart LR
  A[UI (static/index.html)] -->|POST /score?k=K| B[FastAPI /score<br/>src/trialflow/api.py]
  B --> C[score_visits()<br/>src/trialflow/scoring.py]
  C --> D[predict_proba_df()<br/>src/trialflow/model_io.py]
  D --> E[models/*.pkl]

  A -->|ROI / ENB считаем на клиенте| A

  classDef ui fill:#0ea5e9,color:#022336,stroke:#0ea5e9
  classDef api fill:#a78bfa,color:#111,stroke:#a78bfa
  classDef core fill:#fcd34d,color:#111,stroke:#fcd34d
  class A ui
  class B api
  class C core
  class D core
  class E core