# ECA Streamlit App

This app provides deterministic and probabilistic Engineering Critical Assessment (ECA) workflows.

## How to use

1. Ensure your Python modules (`models.py`, `fracture_models.py`, `fatigue.py`, `plotting.py`, `sensitivity.py`, `eca_deterministic.py`, `eca_probabilistic.py`, etc.) are **in the same folder** as this `eca_app` directory (or are importable on `PYTHONPATH`).
2. Install dependencies and run:

```bash
cd eca_app
pip install -r requirements.txt
streamlit run app.py
```

3. In the UI, configure inputs and click **Run** on each page.

> Note: The probabilistic page uses `run_pfm` from your modules and computes a Spearman-based sensitivity.
