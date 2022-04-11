import bw2data as bd
import bw_processing as bwp
from fs.zipfs import ZipFS
from scipy.stats import lognorm, dirichlet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from gsa_framework.utils import write_pickle, read_pickle

from akula.electricity.create_datapackages import DATA_DIR
from akula.markets import get_dirichlet_scales
from akula.utils import get_activities_from_indices

bd.projects.set_current("GSA for archetypes")
write_figs = Path("write_files") / "paper3" / "electricity_markets_dirichlet_html"
write_figs.mkdir(parents=True, exist_ok=True,)

dp = bwp.load_datapackage(ZipFS(str(DATA_DIR / "entso-timeseries.zip")))
indices = dp.get_resource('timeseries ENTSO electricity values.indices')[0]
data = dp.get_resource('timeseries ENTSO electricity values.data')[0]

electricity_markets = get_activities_from_indices(indices)

# Fit distributions
cols = sorted(list(set(indices['col'])))

i = 0

electricity_markets_filtered = {}

for market, exchanges in electricity_markets.items():
    if i % 10 == 0:
        print(f"{i}/{len(electricity_markets)} market")
    if len(exchanges) == 1 and "transformation" in exchanges[0].input['name'] and exchanges[0]['amount'] == 1:
        continue
    col = market.id
    exchanges_filtered = []
    for exc in exchanges:
        row = exc.input.id
        where = np.where(indices == np.array((row, col), dtype=bwp.INDICES_DTYPE))[0]
        assert len(where) == 1
        ind = where[0]

        data_pos = data[ind, :]
        data_pos = data_pos[data_pos > 0]
        assert sum(data_pos > 1) == 0
        data_pos = data_pos[data_pos < 1]
        # if "import from" in exc.input['name'] and len(data_pos) == 0:
        #     continue
        if len(data_pos) == 0:
            continue
        shape, loc, scale = lognorm.fit(data_pos, floc=0)
        exc["uncertainty type"] = 2
        exc["loc"] = np.log(scale)
        exc['scale'] = shape
        exchanges_filtered.append(exc)
    if len(exchanges_filtered) > 0:
        electricity_markets_filtered[market] = exchanges_filtered
    i += 1
    # if i == 11:
    #     break

electricity_markets = electricity_markets_filtered

# write_pickle(electricity_markets, "electricity_markets.pickle")
# electricity_markets = read_pickle("electricity_markets.pickle")
# key1 = list(electricity_markets)[0]
# key2 = list(electricity_markets)[3]
# electricity_markets = {key1: electricity_markets[key1]}
dirichlet_scales = get_dirichlet_scales(electricity_markets, True, True, False)

# Make plots
num_bins = 500
opacity = 0.9
color_gray_hex = "#b2bcc0"
color_darkgray_hex = "#485063"
color_black_hex = "#212931"
color_pink_rgb = "rgb(148, 52, 110)"
color_blue_rgb = "rgb(29,105,150)"
color_orange_rgb = "rgb(217,95,2)"
color_red_hex = "#ff2c54"

num_samples = data.shape[1]

j = 0

for market, exchanges in electricity_markets.items():

    col = market.id

    exc_names = [f"{exc.input['name'][:40]}, {exc.input['location']}" for exc in exchanges]

    ncols = 1
    nrows = len(exchanges)

    fig = make_subplots(
        cols=ncols,
        rows=nrows,
        subplot_titles=exc_names,
        vertical_spacing=0.5 / len(exchanges),
    )
    showlegend = True

    Y_dir = dirichlet.rvs(
        np.array([exc["amount"] for exc in exchanges]) * dirichlet_scales[j],
        size=num_samples,
    )

    for i, exc in enumerate(exchanges):
        row = exc.input.id
        where = np.where(indices == np.array((row, col), dtype=bwp.INDICES_DTYPE))[0]
        assert len(where) == 1
        ind = where[0]

        Y = data[ind, :]
        min_samples = min(Y)
        max_samples = max(Y)

        bin_min = min_samples
        bin_max = max_samples

        bins_ = np.linspace(bin_min, bin_max, num_bins+1, endpoint=True)
        Y_samples, _ = np.histogram(Y, bins=bins_, density=True)

        midbins = (bins_[1:]+bins_[:-1])/2
        # shape = exc['shape']
        loc = exc['loc']
        scale = exc['scale']
        Y_distr = lognorm.pdf(midbins, s=scale, scale=np.exp(loc))

        Y_dirichlet, _ = np.histogram(Y_dir[:, i], bins=bins_, density=True)

        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_samples,
                # name=r"$\text{Real data samples}$",
                name="Real data samples",
                legendgroup="real data",
                showlegend=showlegend,
                opacity=opacity,
                line=dict(color=color_darkgray_hex, width=1, shape="hvh"),
                fill="tozeroy",
            ),
            col=1,
            row=i+1,
        )
        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_distr,
                line=dict(color=color_red_hex),
                # name=r"$\text{Fitted lognormal distributions}$",
                name="Fitted lognormal distributions",
                legendgroup="lognormal",
                showlegend=showlegend,
            ),
            col=1,
            row=i+1,
        )
        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_dirichlet,
                # name=r"$\text{Derived Dirichlet samples}$",
                name="Derived Dirichlet samples",
                legendgroup="dirichlet",
                showlegend=showlegend,
                opacity=opacity,
                line=dict(color=color_blue_rgb, width=1, shape="hvh"),
                fill="tozeroy",
            ),
            col=1,
            row=i+1,
        )
        fig.add_trace(
            go.Scatter(
                x=[exc['amount']],
                y=[0],
                line=dict(color=color_red_hex),
                mode="markers",
                # name=r"$\text{Exc amount (log median)}$",
                name="Exc amount (log median)",
                legendgroup="static",
                showlegend=showlegend
            ),
            col=1,
            row=i+1,
        )

        fig.update_xaxes(title=f"{exc.input['unit']}")

        showlegend = False

    fig.add_annotation(
        text=f"{market['name'][:40]}, {market['location']}",
        showarrow=False,
        arrowhead=1, font_size=16,
        xref="paper", yref="paper",
        x=0.5, y=1+1/nrows/2,
        xanchor="center", yanchor="middle",
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=color_gray_hex,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=color_black_hex,
        showline=True,
        linewidth=1,
        linecolor=color_gray_hex,
    )

    # fig.update_yaxes(title_text=r"$\text{Frequency}$")
    fig.update_yaxes(title_text="Frequency")
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=color_gray_hex,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=color_black_hex,
        showline=True,
        linewidth=1,
        linecolor=color_gray_hex,
    )

    fig.update_layout(
        width=800*ncols,
        height=220*nrows,
        legend=dict(
            yanchor="top",
            y=1.00,  # -0.7
            xanchor="left",
            x=1.2,
            orientation='v',
            font=dict(size=13),
            bordercolor=color_darkgray_hex,
            borderwidth=1,
        ),
        margin=dict(t=100+3*len(exchanges), b=10, l=100, r=0),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )

    fig.write_html(write_figs / f"{j}_{market['name'][:40]}_{market['location']}.html")

    # fig.show()

    j += 1

print("")
