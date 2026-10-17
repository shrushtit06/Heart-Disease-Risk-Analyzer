"""Model training, evaluation, prediction and explanation helpers."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from preprocessing import CATEGORICAL, CONTINUOUS, TARGET, encode

RANDOM_STATE = 42


@dataclass
class Bundle:
    models: dict
    columns: list
    metrics: dict
    X_test_raw: pd.DataFrame
    y_test: pd.Series
    ranges: dict          # min/max of continuous attributes (for what-if sweeps)
    categories: dict      # allowed values of categorical attributes
    nb_plain: object = None   # un-calibrated Naive Bayes, used only for explanations
    class_stats: dict = None  # per-class mean/std of continuous attributes
    typical: dict = None      # median (numeric) / most common (categorical) value


def train(df: pd.DataFrame) -> Bundle:
    X_raw = df.drop(columns=[TARGET])
    y = df[TARGET]
    Xr_tr, Xr_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    X_tr = encode(Xr_tr)
    columns = list(X_tr.columns)
    X_te = encode(Xr_te, columns)

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=15, random_state=RANDOM_STATE),
        # Raw Gaussian NB is badly over-confident (0% / 100%), so calibrate it
        "Naive Bayes": CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5),
    }
    metrics = {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        pred = m.predict(X_te)
        proba = m.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        metrics[name] = {
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred),
            "Recall": recall_score(y_te, pred),
            "F1": f1_score(y_te, pred),
            "ROC AUC": roc_auc_score(y_te, proba),
            "cm": confusion_matrix(y_te, pred),
            "fpr": fpr,
            "tpr": tpr,
        }

    ranges = {c: (float(df[c].min()), float(df[c].max())) for c in CONTINUOUS}
    categories = {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL + ["FastingBS"]}
    nb_plain = GaussianNB().fit(X_tr, y_tr)
    class_stats = {c: {k: (float(Xr_tr.loc[y_tr == k, c].mean()),
                           float(Xr_tr.loc[y_tr == k, c].std()))
                       for k in (0, 1)} for c in CONTINUOUS}
    typical = {}
    for c in X_raw.columns:
        if c in CONTINUOUS:
            typical[c] = float(X_raw[c].median())
        else:
            typical[c] = X_raw[c].mode().iloc[0]
            if hasattr(typical[c], "item"):
                typical[c] = typical[c].item()
    return Bundle(models, columns, metrics, Xr_te, y_te, ranges, categories,
                  nb_plain, class_stats, typical)


def _row(patient: dict) -> pd.DataFrame:
    return pd.DataFrame([patient])


def predict(bundle: Bundle, patient: dict) -> dict:
    """Probability of heart disease from each model."""
    X = encode(_row(patient), bundle.columns)
    return {n: float(m.predict_proba(X)[0, 1]) for n, m in bundle.models.items()}


def sweep(bundle: Bundle, patient: dict, attr: str, values) -> pd.DataFrame:
    """Re-score the patient while varying one attribute (what-if analysis)."""
    rows = []
    for v in values:
        p = dict(patient)
        p[attr] = v
        rows.append(p)
    X = encode(pd.DataFrame(rows), bundle.columns)
    out = pd.DataFrame({attr: list(values)})
    for n, m in bundle.models.items():
        out[n] = m.predict_proba(X)[:, 1]
    return out


def attribute_importance(bundle: Bundle, name: str, n_repeats: int = 15) -> pd.DataFrame:
    """Permutation importance per ORIGINAL attribute (drop in ROC AUC when shuffled)."""
    model = bundle.models[name]
    Xr, y = bundle.X_test_raw, bundle.y_test
    base = roc_auc_score(y, model.predict_proba(encode(Xr, bundle.columns))[:, 1])
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for attr in Xr.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = Xr.copy()
            Xp[attr] = rng.permutation(Xp[attr].values)
            auc = roc_auc_score(y, model.predict_proba(encode(Xp, bundle.columns))[:, 1])
            drops.append(base - auc)
        rows.append({"Attribute": attr, "Importance": float(np.mean(drops))})
    return (pd.DataFrame(rows).sort_values("Importance", ascending=False)
            .reset_index(drop=True))


def tree_path(bundle: Bundle, patient: dict) -> list:
    """Human-readable decision rules the tree followed for this patient."""
    tree = bundle.models["Decision Tree"]
    X = encode(_row(patient), bundle.columns)
    t = tree.tree_
    node_ids = tree.decision_path(X).indices
    steps = []
    for nid in node_ids:
        if t.children_left[nid] == t.children_right[nid]:  # leaf
            continue
        col = bundle.columns[t.feature[nid]]
        thr = t.threshold[nid]
        val = X.iloc[0, t.feature[nid]]
        went_left = val <= thr
        dummy_of = next((c for c in CATEGORICAL if col.startswith(c + "_")), None)
        if dummy_of:
            cat = col[len(dummy_of) + 1:]
            steps.append(f"{dummy_of} {'≠' if went_left else '='} {cat}")
        else:
            steps.append(f"{col} {'≤' if went_left else '>'} {thr:.2f}")
    return steps


# ───────────────────────── explanation helpers ─────────────────────────────
def _dummy_of(col):
    return next((c for c in CATEGORICAL if col.startswith(c + "_")), None)


def tree_path_nodes(bundle: Bundle, patient: dict) -> list:
    tree = bundle.models["Decision Tree"]
    return list(tree.decision_path(encode(_row(patient), bundle.columns)).indices)


def tree_layout(bundle: Bundle) -> dict:
    """x/y position, label and leaf risk for every node (for drawing the tree)."""
    t = bundle.models["Decision Tree"].tree_
    nodes = {}
    counter = [0]

    def rec(nid, depth):
        l, r = t.children_left[nid], t.children_right[nid]
        v = t.value[nid][0]
        info = {"y": -depth, "samples": int(t.n_node_samples[nid]),
                "prob": float(v[1] / v.sum()), "leaf": l == r, "children": []}
        if l == r:
            info["x"] = float(counter[0])
            counter[0] += 1
        else:
            rec(l, depth + 1)
            rec(r, depth + 1)
            info["x"] = (nodes[l]["x"] + nodes[r]["x"]) / 2
            info["children"] = [int(l), int(r)]
            col = bundle.columns[t.feature[nid]]
            d = _dummy_of(col)
            if d:
                info["text"] = f"{d}<br>= {col[len(d) + 1:]} ?"
                info["labels"] = ("No", "Yes")      # left = value 0, right = value 1
            else:
                info["text"] = f"{col}<br>≤ {t.threshold[nid]:.1f} ?"
                info["labels"] = ("Yes", "No")
        nodes[int(nid)] = info

    rec(0, 0)
    return nodes


def nb_contributions(bundle: Bundle, patient: dict):
    """Per-attribute log-likelihood ratio (disease vs. no disease) from plain NB."""
    nb = bundle.nb_plain
    x = encode(_row(patient), bundle.columns).iloc[0].values

    def loglik(c):
        return (-0.5 * np.log(2 * np.pi * nb.var_[c])
                - 0.5 * (x - nb.theta_[c]) ** 2 / nb.var_[c])

    llr = loglik(1) - loglik(0)
    agg = {}
    for col, v in zip(bundle.columns, llr):
        agg[_dummy_of(col) or col] = agg.get(_dummy_of(col) or col, 0.0) + float(v)
    df = pd.DataFrame({"Attribute": list(agg), "Evidence": list(agg.values())})
    prior = float(np.log(nb.class_prior_[1] / nb.class_prior_[0]))
    return df, prior, [float(p) for p in nb.class_prior_]


def local_effects(bundle: Bundle, patient: dict) -> pd.DataFrame:
    """How much each attribute moves this patient's risk compared with a typical value.

    Effect = risk(patient) - risk(patient with that ONE attribute set to the
    typical value), averaged over both models. Positive = raises risk.
    """
    base = np.mean(list(predict(bundle, patient).values()))
    rows = []
    for attr, typ in bundle.typical.items():
        p = dict(patient)
        p[attr] = typ
        rows.append(p)
    X = encode(pd.DataFrame(rows), bundle.columns)
    alt = np.mean([m.predict_proba(X)[:, 1] for m in bundle.models.values()], axis=0)
    return pd.DataFrame({"Attribute": list(bundle.typical), "Effect": base - alt})
