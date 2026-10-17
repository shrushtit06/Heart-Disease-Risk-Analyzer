"""Heart Disease Risk Dashboard  ·  run with:  streamlit run app.py"""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis as an
import model as ml
import theme as th
from preprocessing import (CATEGORICAL, CONTINUOUS, DESCRIPTIONS, NUMERIC,
                           load_clean)

st.set_page_config(page_title="Heart Disease Risk Dashboard", page_icon="❤️",
                   layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2.2rem;}
.banner {background: linear-gradient(135deg,#0F766E 0%,#115e59 55%,#0e7490 100%);
  color:white; padding:22px 28px; border-radius:16px; margin-bottom:18px;
  display:flex; align-items:center; gap:18px;}
.banner .icon {font-size:44px; line-height:1;}
.banner h1 {color:white; margin:0; padding:0; font-size:1.9rem;}
.banner p {color:rgba(255,255,255,0.85); margin:4px 0 0 0; font-size:0.95rem;}
[data-testid="stMetric"] {background:#ffffff; border:1px solid rgba(15,23,42,0.10);
  border-radius:12px; padding:12px 16px; box-shadow:0 1px 3px rgba(15,23,42,0.06);}
.hero {background:#ffffff; border:1px solid rgba(15,23,42,0.10); border-radius:16px;
  padding:20px 24px; box-shadow:0 2px 8px rgba(15,23,42,0.07); height:100%;}
.hero .label {font-size:0.85rem; letter-spacing:.04em; text-transform:uppercase; opacity:.65;}
.hero .num {font-size:3.6rem; font-weight:800; line-height:1.05; margin:4px 0;}
.hero .badge {display:inline-block; padding:3px 14px; border-radius:999px;
  color:white; font-weight:700; font-size:0.9rem;}
.hero p {margin:12px 0 0 0; font-size:0.95rem; line-height:1.45;}
.stTabs [data-baseweb="tab"] {font-weight:600;}
</style>""", unsafe_allow_html=True)


def show(fig):
    """st.plotly_chart across Streamlit versions (use_container_width was replaced)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def get_data():
    return load_clean()


@st.cache_resource(show_spinner=False)
def get_bundle(_df):
    return ml.train(_df)


@st.cache_data
def get_importance(model_name):
    return ml.attribute_importance(bundle, model_name)


df = get_data()
with st.spinner("Training the Decision Tree and Naive Bayes models…"):
    bundle = get_bundle(df)


def gauge(p, title):
    level, color = th.risk_level(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".0f"},
        title={"text": f"{title}<br><span style='font-size:0.8em;color:{color}'>{level} risk</span>"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 30], "color": "rgba(42,157,143,0.20)"},
                         {"range": [30, 60], "color": "rgba(244,162,97,0.22)"},
                         {"range": [60, 100], "color": "rgba(230,57,70,0.20)"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=80, b=10))
    return fig


# ───────────────────────── Sidebar: presets + grouped inputs ───────────────
DEFAULTS = dict(age=54, sex="M", cp="ASY", bp=130, chol=220, fbs=False, ecg="Normal",
                maxhr=138, angina="N", oldpeak=0.6, slope="Flat")
PRESETS = {
    "healthy": dict(age=38, sex="F", cp="ATA", bp=118, chol=190, fbs=False, ecg="Normal",
                    maxhr=172, angina="N", oldpeak=0.0, slope="Up"),
    "highrisk": dict(age=63, sex="M", cp="ASY", bp=150, chol=260, fbs=True, ecg="ST",
                     maxhr=105, angina="Y", oldpeak=2.5, slope="Flat"),
    "reset": DEFAULTS,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    st.session_state.update(PRESETS[name])


with st.sidebar:
    st.header("🧑‍⚕️ Patient profile")
    st.caption("Quick examples")
    b1, b2, b3 = st.columns(3)
    b1.button("🟢 Healthy", on_click=apply_preset, args=("healthy",))
    b2.button("🔴 High risk", on_click=apply_preset, args=("highrisk",))
    b3.button("↺ Reset", on_click=apply_preset, args=("reset",))

    with st.expander("👤 Basic info", expanded=True):
        st.slider("Age (years)", 20, 90, key="age")
        st.radio("Sex", ["M", "F"], horizontal=True, key="sex")
    with st.expander("🫀 Symptoms", expanded=True):
        st.selectbox("Chest pain type", ["ASY", "ATA", "NAP", "TA"], key="cp",
                     help=DESCRIPTIONS["ChestPainType"])
        st.radio("Chest pain during exercise (angina)", ["N", "Y"], horizontal=True,
                 key="angina", help=DESCRIPTIONS["ExerciseAngina"])
    with st.expander("🧪 Test results", expanded=True):
        st.slider("Resting blood pressure (mm Hg)", 80, 200, key="bp")
        st.slider("Cholesterol (mg/dl)", 100, 600, key="chol")
        st.checkbox("Fasting blood sugar above 120 mg/dl", key="fbs")
        st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], key="ecg",
                     help=DESCRIPTIONS["RestingECG"])
        st.slider("Max heart rate (bpm)", 60, 210, key="maxhr")
        st.slider("Oldpeak (ST depression)", -2.0, 6.5, step=0.1, key="oldpeak",
                  help=DESCRIPTIONS["Oldpeak"])
        st.selectbox("ST slope", ["Up", "Flat", "Down"], key="slope",
                     help=DESCRIPTIONS["ST_Slope"])

S = st.session_state
patient = {"Age": S.age, "Sex": S.sex, "ChestPainType": S.cp, "RestingBP": S.bp,
           "Cholesterol": S.chol, "FastingBS": int(S.fbs), "RestingECG": S.ecg,
           "MaxHR": S.maxhr, "ExerciseAngina": S.angina, "Oldpeak": float(S.oldpeak),
           "ST_Slope": S.slope}

st.markdown("""
<div class="banner"><div class="icon">❤️</div><div>
<h1>Heart Disease Risk Dashboard</h1>
<p>Enter a patient profile on the left to see the estimated risk, why the models say so,
and how every attribute relates to heart disease. Educational project, not a medical diagnosis.</p>
</div></div>""", unsafe_allow_html=True)

tab_pred, tab_explore, tab_perf, tab_learn, tab_data = st.tabs(
    ["🩺 Risk prediction", "📊 Explore & drivers", "⚙️ Model performance",
     "🎓 Learn how it works", "🗂️ Data"])

# ───────────────────────────── Tab 1: prediction ───────────────────────────
with tab_pred:
    probs = ml.predict(bundle, patient)
    avg = float(np.mean(list(probs.values())))
    level, color = th.risk_level(avg)
    effects = ml.local_effects(bundle, patient)

    up = effects[effects.Effect > 0.03].sort_values("Effect", ascending=False).head(2)
    down = effects[effects.Effect < -0.03].sort_values("Effect").head(2)
    name = lambda a: f"{th.PRETTY[a]} ({th.fmt_value(a, patient[a])})"
    parts = []
    if len(up):
        parts.append("**Raising risk:** " + ", ".join(name(a) for a in up.Attribute) + ".")
    if len(down):
        parts.append("**Lowering risk:** " + ", ".join(name(a) for a in down.Attribute) + ".")
    reasons = " ".join(parts) or "No single attribute stands out for this profile."
    gap = abs(probs["Decision Tree"] - probs["Naive Bayes"])
    if gap > 0.25:
        reasons += " ⚠️ The two models disagree noticeably, so treat this estimate as uncertain."

    reasons_html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", reasons)

    c_card, c_dt, c_nb = st.columns([1.5, 1, 1])
    with c_card:
        st.markdown(f"""
<div class="hero" style="border-left:8px solid {color}">
<div class="label">Estimated heart disease risk</div>
<div class="num" style="color:{color}">{avg:.0%}</div>
<span class="badge" style="background:{color}">{level} risk</span>
<p>{reasons_html}</p>
</div>""", unsafe_allow_html=True)
    with c_dt:
        show(gauge(probs["Decision Tree"], "Decision Tree"))
    with c_nb:
        show(gauge(probs["Naive Bayes"], "Naive Bayes"))

    summary = pd.DataFrame([{**patient,
                             "Risk_DecisionTree": round(probs["Decision Tree"], 3),
                             "Risk_NaiveBayes": round(probs["Naive Bayes"], 3),
                             "Risk_Average": round(avg, 3), "Level": level}])
    st.download_button("⬇️ Download this result (CSV)", summary.to_csv(index=False),
                       "heart_risk_result.csv", "text/csv")

    st.subheader("Why this risk? Compared with a typical patient")
    left, right = st.columns([1.1, 1])
    with left:
        show(an.fig_effects(effects))
    with right:
        tbl = []
        for _, r in effects.iterrows():
            a, pts = r["Attribute"], r["Effect"] * 100
            arrow = "▲" if pts > 1 else "▼" if pts < -1 else "≈"
            tbl.append({"Attribute": th.PRETTY[a],
                        "This patient": th.fmt_value(a, patient[a]),
                        "Typical": th.fmt_value(a, bundle.typical[a]),
                        "Effect on risk": f"{arrow} {pts:+.0f} pts"})
        st.dataframe(pd.DataFrame(tbl), hide_index=True, height=440)
        st.caption("Typical = median (numbers) or most common value (categories) in the data. "
                   "Effect = how much this attribute alone changes the average risk.")

    st.subheader("🔬 What-if explorer")
    st.write("Change **one** attribute and keep everything else as set in the sidebar.")
    attr = st.selectbox("Attribute to vary", list(patient.keys()),
                        format_func=lambda a: th.PRETTY[a])
    series = [("Decision Tree", th.TREE_COLOR), ("Naive Bayes", th.NB_COLOR)]
    fig = go.Figure()
    if attr in CONTINUOUS:
        lo, hi = bundle.ranges[attr]
        sw = ml.sweep(bundle, patient, attr, np.linspace(lo, hi, 60))
        for nm, c in series:
            fig.add_scatter(x=sw[attr], y=sw[nm] * 100, name=nm, mode="lines",
                            line=dict(color=c, width=3))
        fig.add_vline(x=patient[attr], line_dash="dash", line_color="gray",
                      annotation_text="current", annotation_position="top")
    else:
        cats = bundle.categories[attr]
        sw = ml.sweep(bundle, patient, attr, cats)
        for nm, c in series:
            fig.add_bar(x=[str(v) for v in sw[attr]], y=sw[nm] * 100, name=nm, marker_color=c)
        fig.update_layout(barmode="group")
    fig.update_yaxes(title="Predicted risk (%)", range=[0, 100])
    fig.update_xaxes(title=th.PRETTY[attr])
    an.style(fig, 400)
    show(fig)
    st.caption(DESCRIPTIONS[attr])

    with st.expander("🌳 Rules the Decision Tree followed for this patient"):
        steps = ml.tree_path(bundle, patient)
        st.write(" → ".join(f"`{s}`" for s in steps) if steps else "–")

# ───────────────────────── Tab 2: explore & drivers ────────────────────────
with tab_explore:
    k = an.kpis(df)
    cols = st.columns(len(k))
    for col, (label, value) in zip(cols, k.items()):
        col.metric(label, value)

    left, right = st.columns([1, 2])
    with left:
        show(an.fig_target(df))
    with right:
        show(an.fig_age_sex(df))

    st.subheader("How does one attribute relate to heart disease?")
    sel = st.selectbox("Choose an attribute", NUMERIC + CATEGORICAL, key="explore_attr",
                       format_func=lambda a: th.PRETTY[a])
    st.caption(DESCRIPTIONS[sel])
    if sel in CONTINUOUS:
        show(an.fig_numeric(df, sel))
    else:
        show(an.fig_category_rate(df, sel))

    st.subheader("Two attributes together")
    s1, s2 = st.columns(2)
    x = s1.selectbox("X axis", CONTINUOUS, index=0, format_func=lambda a: th.PRETTY[a])
    y = s2.selectbox("Y axis", CONTINUOUS, index=3, format_func=lambda a: th.PRETTY[a])
    show(an.fig_scatter(df, x, y))

    st.divider()
    st.subheader("🧠 What drives risk overall?")
    st.write("**Permutation importance**: how much a model's ROC AUC drops when one "
             "attribute's values are shuffled. Bigger drop = the model relies on it more.")
    cols = st.columns(2)
    for col, nm, c in zip(cols, ["Decision Tree", "Naive Bayes"],
                          [th.TREE_COLOR, th.NB_COLOR]):
        imp = get_importance(nm).sort_values("Importance")
        fig = go.Figure(go.Bar(x=imp["Importance"], y=imp["Attribute"].map(th.PRETTY),
                               orientation="h", marker_color=c))
        fig.update_layout(title=nm)
        fig.update_xaxes(title="Drop in ROC AUC")
        an.style(fig, 460, legend=False)
        with col:
            show(fig)
    show(an.fig_target_correlation(df))
    with st.expander("Full correlation heatmap"):
        show(an.fig_heatmap(df))

# ───────────────────────── Tab 3: model performance ────────────────────────
with tab_perf:
    st.caption("Metrics on a 20% hold-out test set (stratified split).")
    table = pd.DataFrame({n: {k: v for k, v in m.items()
                              if k in ("Accuracy", "Precision", "Recall", "F1", "ROC AUC")}
                          for n, m in bundle.metrics.items()}).T
    st.dataframe(table.style.format("{:.3f}"))

    a, b = st.columns(2)
    roc = go.Figure()
    for nm, c in [("Decision Tree", th.TREE_COLOR), ("Naive Bayes", th.NB_COLOR)]:
        m = bundle.metrics[nm]
        roc.add_scatter(x=m["fpr"], y=m["tpr"], mode="lines",
                        name=f"{nm} (AUC {m['ROC AUC']:.2f})", line=dict(color=c, width=3))
    roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                    line=dict(color="gray", dash="dot"))
    roc.update_layout(title="ROC curve", xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    an.style(roc, 400)
    with a:
        show(roc)
    with b:
        pick = st.radio("Confusion matrix for", list(bundle.models), horizontal=True)
        cm = bundle.metrics[pick]["cm"]
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred: No", "Pred: Yes"],
                                   y=["Actual: No", "Actual: Yes"], text=cm,
                                   texttemplate="%{text}", colorscale="Teal", showscale=False))
        fig.update_yaxes(autorange="reversed")
        an.style(fig, 360, legend=False)
        show(fig)

# ───────────────────────── Tab 4: learn ────────────────────────────────────
with tab_learn:
    page = st.radio("Topic", ["🌳 Decision Tree", "📈 Naive Bayes", "📏 Understanding the risk score"],
                    horizontal=True, label_visibility="collapsed")

    if page.startswith("🌳"):
        st.header("🌳 How the Decision Tree works")
        st.markdown("""
A Decision Tree is a flowchart of **yes/no questions**. Starting at the top, a patient
answers one question at a time (e.g. *"Is the ST slope Up?"*) and follows the matching branch
until they land in a **leaf**.

**How it learns.** From the training patients, the algorithm tries every possible question and
keeps the one that separates *disease* from *no disease* best, measured by **Gini impurity**
(0 = a perfectly pure group, 0.5 = a 50/50 mix). It then repeats inside each branch.
To keep the tree simple and avoid memorising the data, it is limited to **4 levels** and
every leaf must contain **at least 15 patients**.

**How it produces a risk %.** The risk is simply the **share of training patients in that leaf who had
heart disease**. A leaf showing 85% means 85 of 100 similar training patients were diagnosed.
""")
        show(an.fig_tree(bundle, patient))
        st.caption("Red path = the route the current sidebar patient takes. "
                   "Leaf colour/number = % of training patients in that leaf with heart disease. "
                   "Hover over any node for patient counts.")
        t_nodes = ml.tree_layout(bundle)
        leaf = t_nodes[ml.tree_path_nodes(bundle, patient)[-1]]
        steps = ml.tree_path(bundle, patient)
        st.subheader("What it means for the current patient")
        c1, c2, c3 = st.columns(3)
        c1.metric("Questions answered", len(steps))
        c2.metric("Similar training patients", leaf["samples"])
        c3.metric("Tree risk", f"{leaf['prob']:.0%}")
        st.write("Rules followed: " + " → ".join(f"`{s}`" for s in steps))
        st.info(f"This patient lands in a leaf with **{leaf['samples']}** training patients, "
                f"**{leaf['prob']:.0%}** of whom had heart disease. That is the tree's risk score.")
        st.subheader("Strengths and limits")
        st.markdown("""
- ✅ Easy to read and explain; you can follow every decision.
- ✅ Handles numbers and categories without scaling.
- ⚠️ Risk only moves in **steps** (one value per leaf), so small changes in an attribute often do nothing,
  then suddenly jump when a threshold is crossed (check the what-if curve).
- ⚠️ Small changes in the data can produce a different tree.
""")
        with st.expander("Tree as text"):
            from sklearn.tree import export_text
            st.code(export_text(bundle.models["Decision Tree"], feature_names=bundle.columns))

    elif page.startswith("📈"):
        st.header("📈 How Naive Bayes works")
        prior_no, prior_yes = bundle.nb_plain.class_prior_
        st.markdown(f"""
Naive Bayes uses **Bayes' theorem**: start with how common the disease is, then update that belief with
the evidence from every attribute.

1. **Prior.** In the training data **{prior_yes:.0%}** of patients have heart disease and {prior_no:.0%} do not.
2. **Likelihood per attribute.** For each group (disease / no disease) the model learns what typical values look like,
   e.g. the average age and spread. For the current patient it asks: *"How likely is this value
   in the disease group compared with the no-disease group?"*
3. **Combine.** It assumes the attributes are **independent** (that is the "naive" part) so the evidence can just be
   multiplied. In log form this is simply **adding** each attribute's evidence to the prior.
4. **Convert to a probability.** The final score is turned into a % between 0 and 100.
""")
        st.latex(r"P(\text{disease}\mid x) \;\propto\; P(\text{disease}) \times \prod_i P(x_i \mid \text{disease})")
        contrib, _, _ = ml.nb_contributions(bundle, patient)
        st.subheader("Evidence for the current patient")
        show(an.fig_nb_contrib(contrib.assign(Attribute=contrib["Attribute"].map(th.PRETTY))))
        pos = contrib.sort_values("Evidence", ascending=False)
        st.info(f"**{th.PRETTY[pos.iloc[0]['Attribute']]}** pushes the most towards heart disease, while "
                f"**{th.PRETTY[pos.iloc[-1]['Attribute']]}** pushes the most away from it.")
        st.caption("The bars are raw Naive Bayes evidence. The risk % on the prediction tab is these scores "
                   "**calibrated**, so the two are linked but not identical.")
        st.subheader("What the model expects in each group")
        nb_attr = st.selectbox("Numeric attribute", CONTINUOUS, key="nb_attr",
                               format_func=lambda a: th.PRETTY[a])
        show(an.fig_nb_density(bundle, nb_attr, patient[nb_attr]))
        st.caption("Naive Bayes fits a bell curve for each group. Where the current patient's value (dashed line) "
                   "sits higher under one curve, that attribute favours that group.")

    else:
        st.header("📏 Understanding the risk score")
        st.markdown("""
**What the % means.** It is the model's estimated chance that a patient with this profile has
heart disease, learned from 918 patients in the dataset. Plain Naive Bayes tends to be
over-confident (jumping to 0% or 100%), so its scores are **calibrated** to behave like real
probabilities.

| Risk % | Level | Meaning |
|---|---|---|
| below 30% | 🟢 Low | Most similar patients in the data did **not** have heart disease |
| 30% to 60% | 🟠 Moderate | Mixed picture, so attributes point both ways |
| 60% and above | 🔴 High | Most similar patients in the data **did** have heart disease |

**Two models, one answer.** The Decision Tree gives step-like scores, Naive Bayes gives smooth ones.
The headline number is their average. When they disagree strongly, treat the result as uncertain.

**"Why this risk?"** For each attribute the dashboard swaps that one value for a typical one
and re-scores the patient. The difference is that attribute's effect, shown in percentage points.

**Important limits.**
- The dataset is small (918 rows), so scores are estimates, not certainties.
- A "Low" score does not rule out disease and a "High" score is not a diagnosis.
- Only the attributes in this dataset are considered (no family history, smoking, imaging, and so on).
- Please see a doctor for any real health concern.
""")
        p = ml.predict(bundle, patient)
        lt, _ = th.risk_level(p["Decision Tree"])
        ln, _ = th.risk_level(p["Naive Bayes"])
        st.success(f"Current patient: Decision Tree **{p['Decision Tree']:.0%} ({lt})**, "
                   f"Naive Bayes **{p['Naive Bayes']:.0%} ({ln})**.")

# ───────────────────────────── Tab 5: data ─────────────────────────────────
with tab_data:
    st.write(f"{len(df)} rows · {df.shape[1]} columns. Cholesterol / RestingBP "
             "values of 0 (not recorded) were replaced by the column median.")
    st.dataframe(df)
    st.download_button("Download cleaned CSV", df.to_csv(index=False),
                       "heart_cleaned.csv", "text/csv")
    with st.expander("Attribute dictionary"):
        st.table(pd.Series(DESCRIPTIONS, name="Meaning"))
