# Data & Privacy
- Data source: Synthetic datasets (e.g., Synthea) and/or public no-show datasets.
- Prohibited: Any real medical records, personal identifiers (name, address, phone number, email, insurance ID, etc.).
- Minimization: Retain only essential fields (age_band, visit_type, weekday, lead_time_days, site_id (pseudo), label_no_show).
- Storage: Large files — via DVC-remote; local samples — in data/sample/; do not version sensitive files.
- Licenses: Specify the dataset license and link in the README.
- DPIA-lite: Risk checklist in `reports/dpia_checklist.md` (not legal advice).