import bw2data as bd
import bw2calc as bc
import pickle
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# Local files
from .utils import update_fig_axes, COLOR_PSI_BLUE, COLOR_DARKGRAY_HEX, get_consumption_activity

MC_DIR = Path(__file__).parent.parent.resolve() / "data" / "monte-carlo" / "sampling-modules"
MC_DIR.mkdir(parents=True, exist_ok=True)


def compute_consumption_lcia(project, iterations, seed=42, datapackages=None):
    bd.projects.set_current(project)
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    activity = get_consumption_activity()

    fu, data_objs, _ = bd.prepare_lca_inputs({activity: 1}, method=method, remapping=False)

    if datapackages is not None:
        if type(datapackages) is list:
            data_objs += datapackages
        else:
            data_objs.append(datapackages)

    lca = bc.LCA(
        demand=fu,
        data_objs=data_objs,
        use_arrays=True,
        use_distributions=True,
        seed_override=seed,
    )
    lca.lci()
    lca.lcia()

    scores = [lca.score for _ in zip(range(iterations), lca)]

    return scores


def compute_scores(project, option, iterations, seed=42, datapackage=None):
    fp = MC_DIR / f"{option}-{seed}-{iterations}.pickle"
    if fp.exists():
        with open(fp, "rb") as f:
            scores = pickle.load(f)
    else:
        scores = compute_consumption_lcia(project, iterations, seed, datapackage)
        with open(fp, "wb") as f:
            pickle.dump(scores, f)
    return scores


def plot_sampling_modules(Y0, YS, offset=0):

    Y0 = np.array(Y0) + offset
    YS = np.array(YS) + offset

    start, end = 0, 50
    axis_text = r"$\text{LCIA scores}$"

    color1 = COLOR_PSI_BLUE
    color2 = COLOR_DARKGRAY_HEX

    fig = make_subplots(rows=1, cols=2, shared_xaxes=False)

    x = np.arange(start, end)
    fig.add_trace(go.Scatter(x=x, y=Y0[start:end]-YS[start:end],
                             mode="markers", marker=dict(color=color1, symbol="diamond-tall", size=8), showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=Y0, y=YS, mode="markers", marker=dict(color=color2, line=dict(width=1, color="black")),
                             showlegend=False, opacity=0.65),
                  row=1, col=2)

    fig = update_fig_axes(fig)

    if offset > 0:
        tickvals = [1700, 2000, 2300]
        Ymin, Ymax = 1600, 2400
    else:
        tickvals = [1000, 1300, 1600]
        Ymin, Ymax = 900, 1700

    fig.update_layout(
        width=500,
        height=160,
        xaxis1=dict(domain=[0.0, 0.58]),
        xaxis2=dict(domain=[0.76, 1.0], tickmode="array", tickvals=tickvals),
        yaxis2=dict(tickmode="array", tickvals=tickvals),
        margin=dict(l=20, r=20, t=10, b=20),
    )

    # Ymin = min(np.hstack([Y0, YS]))
    # Ymax = max(np.hstack([Y0, YS]))

    fig.update_xaxes(title_text=r"$\text{Sample number}$", title_standoff=5, row=1, col=1)
    fig.update_yaxes(range=[-100, 100], title_text=r"$\Delta \text{ LCIA scores}$", title_standoff=5, row=1, col=1)
    fig.update_xaxes(range=[Ymin, Ymax], title_text=axis_text, title_standoff=5, row=1, col=2)
    fig.update_yaxes(range=[Ymin, Ymax], title_text=axis_text, title_standoff=5, row=1, col=2)

    return fig
