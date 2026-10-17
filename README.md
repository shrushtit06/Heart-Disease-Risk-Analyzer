# ❤️ Heart Disease Risk Dashboard

Interactive Streamlit dashboard that predicts heart-disease risk and shows how
each attribute affects it (Decision Tree + Naive Bayes).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
(If pip and Streamlit use different Pythons: `python -m pip install -r requirements.txt` then `python -m streamlit run app.py`.)

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (5 tabs) |
| `preprocessing.py` | Loading, cleaning (zeros in Cholesterol/RestingBP → median), one-hot encoding |
| `model.py` | Decision Tree + calibrated Naive Bayes, metrics, predictions, what-if, "why this risk" effects, explanations |
| `analysis.py` | Plotly charts |
| `theme.py` | Shared colours, risk levels, pretty names |
| `.streamlit/config.toml` | Light theme (change `base` to `"dark"` for a dark look) |
| `data/heart.csv` | Dataset (918 patients, 11 attributes + `HeartDisease`) |

## Tabs
1. **Risk prediction** – big result card with plain-English reasons, a gauge per model, "why this risk" vs a typical patient, what-if explorer, CSV download.
2. **Explore & drivers** – distributions, disease rate per category, scatter plots, permutation importance, correlations.
3. **Model performance** – accuracy / precision / recall / F1 / AUC, ROC curve, confusion matrix.
4. **Learn how it works** – Decision Tree, Naive Bayes, and how to read the risk score.
5. **Data** – cleaned table and attribute dictionary.

The sidebar has quick presets (Healthy, High risk, Reset) and grouped inputs.

> Educational project only. Not a medical diagnosis.
