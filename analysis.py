"""Plotly figures that show how each attribute relates to heart disease."""
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

import model as ml
import theme
from preprocessing import CATEGORICAL, TARGET, encode

COLORS = {0: theme.LOW, 1: theme.HIGH}
NAMES = {0: "No disease", 1: "Heart disease"}


def style(fig, height=380, legend=True):
    """Theme-neutral layout (works in light and dark mode) with no overlapping text."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=95 if legend else 65, b=10),
        title=dict(x=0, xanchor="left", y=0.97, yanchor="top"),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom",
                    title_text=""),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


_style = style


def kpis(df: pd.DataFrame) -> dict:
    return {
        "Patients": len(df),
        "With heart disease": f"{df[TARGET].mean():.0%}",
        "Average age": f"{df['Age'].mean():.1f}",
        "Avg max HR": f"{df['MaxHR'].mean():.0f} bpm",
    }


def fig_target(df):
    counts = df[TARGET].map(NAMES).value_counts()
    fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                 color=counts.index,
                 color_discrete_map={NAMES[0]: COLORS[0], NAMES[1]: COLORS[1]},
                 title="Outcome split")
    return _style(fig, 320)


def fig_numeric(df, col):
    d = df.assign(Outcome=df[TARGET].map(NAMES))
    fig = px.histogram(d, x=col, color="Outcome", nbins=30, barmode="overlay",
                       opacity=0.65, marginal="box",
                       color_discrete_map={NAMES[0]: COLORS[0], NAMES[1]: COLORS[1]},
                       title=f"{col} distribution by outcome")
    return _style(fig, 430)


def fig_category_rate(df, col):
    g = (df.groupby(col)[TARGET].agg(["mean", "count"]).reset_index()
         .sort_values("mean", ascending=False))
    g["Rate"] = g["mean"] * 100
    fig = px.bar(g, x=col, y="Rate", text=g.apply(
        lambda r: f"{r['Rate']:.0f}%  (n={int(r['count'])})", axis=1),
        color="Rate", color_continuous_scale=["#2a9d8f", "#f4a261", "#e63946"],
        range_color=[0, 100], title=f"Heart disease rate by {col}")
    fig.update_yaxes(title="% with heart disease", range=[0, 110])
    fig.update_coloraxes(showscale=False)
    fig.add_hline(y=df[TARGET].mean() * 100, line_dash="dot", line_color="gray",
                  annotation_text="overall rate")
    return _style(fig, 400, legend=False)


def fig_target_correlation(df):
    X = encode(df)
    corr = X.corrwith(df[TARGET]).sort_values()
    fig = go.Figure(go.Bar(
        x=corr.values, y=corr.index, orientation="h",
        marker_color=[COLORS[1] if v > 0 else COLORS[0] for v in corr.values]))
    fig.update_layout(title="Correlation with heart disease (red = raises risk, green = lowers)")
    fig.update_xaxes(title="Pearson correlation", range=[-1, 1])
    return _style(fig, 560, legend=False)


def fig_heatmap(df):
    corr = encode(df).assign(**{TARGET: df[TARGET]}).corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    aspect="auto", title="Full correlation heatmap")
    return _style(fig, 700, legend=False)


def fig_scatter(df, x, y):
    d = df.assign(Outcome=df[TARGET].map(NAMES))
    fig = px.scatter(d, x=x, y=y, color="Outcome", opacity=0.7,
                     color_discrete_map={NAMES[0]: COLORS[0], NAMES[1]: COLORS[1]},
                     title=f"{y} vs {x}")
    return _style(fig, 450)


def fig_age_sex(df):
    d = df.assign(AgeGroup=pd.cut(df["Age"], [0, 40, 50, 60, 70, 100],
                                  labels=["<40", "40-49", "50-59", "60-69", "70+"]))
    g = d.groupby(["AgeGroup", "Sex"], observed=True)[TARGET].mean().reset_index()
    g[TARGET] *= 100
    fig = px.line(g, x="AgeGroup", y=TARGET, color="Sex", markers=True,
                  title="Heart disease rate by age group and sex")
    fig.update_yaxes(title="% with heart disease", range=[0, 100])
    return _style(fig, 400)


def fig_tree(bundle, patient):
    """Draw the trained Decision Tree and highlight the path of the current patient."""
    nodes = ml.tree_layout(bundle)
    path = set(ml.tree_path_nodes(bundle, patient))
    ex, ey, px_, py_ = [], [], [], []
    ann = []
    for nid, n in nodes.items():
        for side, ch in enumerate(n["children"]):
            c = nodes[ch]
            on = nid in path and ch in path
            xs, ys = (px_, py_) if on else (ex, ey)
            xs += [n["x"], c["x"], None]
            ys += [n["y"], c["y"], None]
            ann.append(dict(x=(n["x"] + c["x"]) / 2, y=(n["y"] + c["y"]) / 2,
                            text=n["labels"][side], showarrow=False,
                            font=dict(size=10, color="#e63946" if on else "gray")))
    fig = go.Figure()
    fig.add_scatter(x=ex, y=ey, mode="lines", hoverinfo="skip", showlegend=False,
                    line=dict(color="rgba(150,150,150,0.55)", width=1.5))
    fig.add_scatter(x=px_, y=py_, mode="lines", hoverinfo="skip",
                    name="Path of the current patient",
                    line=dict(color="#e63946", width=4))
    internal = [(i, n) for i, n in nodes.items() if not n["leaf"]]
    fig.add_scatter(
        x=[n["x"] for _, n in internal], y=[n["y"] for _, n in internal],
        mode="markers+text", text=[n["text"] for _, n in internal],
        textposition="top center", textfont=dict(size=10), showlegend=False,
        marker=dict(size=12, color=["#e63946" if i in path else "#8d99ae" for i, _ in internal]),
        hovertext=[f"{n['samples']} training patients here<br>{n['prob']:.0%} had heart disease"
                   for _, n in internal], hoverinfo="text")
    leaves = [(i, n) for i, n in nodes.items() if n["leaf"]]
    fig.add_scatter(
        x=[n["x"] for _, n in leaves], y=[n["y"] for _, n in leaves],
        mode="markers+text", text=[f"{n['prob']:.0%}" for _, n in leaves],
        textposition="middle center", textfont=dict(size=10, color="white"),
        showlegend=False,
        marker=dict(size=38, color=[n["prob"] for _, n in leaves], cmin=0, cmax=1,
                    colorscale=[[0, "#2a9d8f"], [0.5, "#f4a261"], [1, "#e63946"]],
                    line=dict(width=[4 if i in path else 0 for i, _ in leaves],
                              color="white"),
                    colorbar=dict(title="Risk", thickness=12, len=0.6)),
        hovertext=[f"Leaf: {n['samples']} training patients<br>{n['prob']:.0%} had heart disease"
                   for _, n in leaves], hoverinfo="text")
    depth = -min(n["y"] for n in nodes.values())
    fig.update_layout(annotations=ann, title="The trained Decision Tree")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[-depth - 0.5, 0.7])
    return style(fig, 640)


def fig_nb_contrib(contrib):
    d = contrib.sort_values("Evidence")
    fig = go.Figure(go.Bar(
        x=d["Evidence"], y=d["Attribute"], orientation="h",
        marker_color=[COLORS[1] if v > 0 else COLORS[0] for v in d["Evidence"]]))
    fig.update_layout(title="Evidence from each attribute for this patient "
                            "(red → towards disease, green → away)")
    fig.update_xaxes(title="Log-likelihood ratio", zeroline=True)
    return style(fig, 460, legend=False)


def fig_nb_density(bundle, attr, value):
    lo, hi = bundle.ranges[attr]
    xs = np.linspace(lo, hi, 200)
    fig = go.Figure()
    for k in (0, 1):
        mu, sd = bundle.class_stats[attr][k]
        pdf = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        fig.add_scatter(x=xs, y=pdf, mode="lines", name=NAMES[k], fill="tozeroy",
                        line=dict(color=COLORS[k], width=3), opacity=0.6)
    fig.add_vline(x=value, line_dash="dash", line_color="gray",
                  annotation_text="current patient", annotation_position="top")
    fig.update_layout(title=f"What Naive Bayes 'expects' for {attr} in each group")
    fig.update_xaxes(title=attr)
    fig.update_yaxes(title="Likelihood")
    return style(fig, 420)


def fig_effects(effects):
    d = effects.sort_values("Effect")
    fig = go.Figure(go.Bar(
        x=d["Effect"] * 100, y=d["Attribute"].map(theme.PRETTY), orientation="h",
        marker_color=[COLORS[1] if v > 0 else COLORS[0] for v in d["Effect"]],
        hovertemplate="%{y}: %{x:+.0f} points<extra></extra>"))
    fig.update_layout(title="Why this risk? Change vs. a typical value "
                            "(red raises, green lowers)")
    fig.update_xaxes(title="Percentage points of risk", zeroline=True)
    return style(fig, 440, legend=False)
