import numpy as np
import bw2data as bd
import bw_processing as bwp
from pathlib import Path
from fs.zipfs import ZipFS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import lognorm

# Local files
from akula.markets import DATA_DIR

color_gray_hex = "#b2bcc0"
color_darkgray_hex = "#485063"
color_black_hex = "#212931"
color_pink_rgb = "rgb(148, 52, 110)"
color_blue_rgb = "rgb(29,105,150)"
color_orange_rgb = "rgb(217,95,2)"
color_red_hex = "#ff2c54"
color_psi_brown = "#85543a"
color_psi_green = "#82911a"
color_psi_blue = "#003b6e"
color_psi_yellow = "#fdca00"
color_psi_purple = "#7c204e"
color_psi_dgreen = "#197418"
opacity = 0.9
num_bins = 300

colors = {
    0: color_psi_blue,
    2: color_psi_purple,
    1: color_psi_green,
}

iterations_zoomed = 80

if __name__ == "__main__":

    bd.projects.set_current("GSA for archetypes")
    iterations = 20000
    write_figs = Path("/Users/akim/PycharmProjects/akula/dev/write_files/paper3")

    dp = bwp.load_datapackage(ZipFS(DATA_DIR / "implicit-markets.zip"))
    tindices = dp.get_resource("implicit-markets.indices")[0]
    tdata = dp.get_resource("implicit-markets.data")[0]

    act_id = 11475
    activity = bd.get_activity(act_id)
    exchanges = [exc for exc in activity.exchanges() if exc['type'] != "production"]

    fig = make_subplots(
        rows=2,
        cols=4,
        vertical_spacing=0.11,
        horizontal_spacing=0.09,
        shared_xaxes=False,
        row_heights=[0.25, 0.75],
        column_widths=[0.2, 0.2, 0.2, 0.4]
    )

    showlegend2 = True
    for i, exchange in enumerate(exchanges):
        showlegend = True
        assert exchange['uncertainty type'] == 2
        loc = exchange['loc']
        scale = exchange['scale']
        min_distr = lognorm.ppf(0.01, s=scale, scale=np.exp(loc))
        max_distr = lognorm.ppf(0.99, s=scale, scale=np.exp(loc))

        ind = np.where(tindices == np.array((exchange.input.id, act_id), dtype=bwp.INDICES_DTYPE))[0][0]
        Y = tdata[ind, :iterations]
        min_samples = min(Y)
        max_samples = max(Y)

        bin_min = min(min_distr, min_samples)
        bin_max = max(max_distr, max_samples)

        bins_ = np.linspace(bin_min, bin_max, num_bins+1, endpoint=True)
        Y_samples, _ = np.histogram(Y, bins=bins_, density=True)

        midbins = (bins_[1:]+bins_[:-1])/2
        Y_distr = lognorm.pdf(midbins, s=scale, scale=np.exp(loc))

        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_samples,
                name=r"$\text{Dirichlet samples for exchange " + str(i+1) + " }$",
                showlegend=showlegend,
                opacity=opacity,
                line=dict(color=colors[i], width=1, shape="hvh"),
                fill="tozeroy",
            ),
            row=1,
            col=i+1,
        )
        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_distr,
                line=dict(color=color_red_hex),
                name=r"$\text{Defined lognormal distributions}$",
                showlegend=showlegend2,
                legendrank=1,
            ),
            row=1,
            col=i+1,
        )
        showlegend = False
        showlegend2 = False
        fig.add_trace(
            go.Scatter(
                x=tdata[ind, :iterations_zoomed],
                y=np.arange(1, iterations_zoomed+1),
                mode='markers',
                showlegend=showlegend,
                marker=dict(
                    color=color_darkgray_hex,
                    size=3,
                    line=dict(
                        width=0
                    )
                ),
            ),
            row=2,
            col=i+1,
        )
        for j in range(iterations_zoomed):
            fig.add_shape(
                type="line",
                x0=0, y0=j+1,
                x1=Y[j], y1=j+1,
                row=2,
                col=i+1,
                line=dict(color=colors[i], width=1)
            )
        fig.add_bar(
            x=tdata[ind, :iterations_zoomed],
            y=np.arange(1, iterations_zoomed+1),
            row=2,
            col=4,
            showlegend=False,
            orientation='h',
            marker_color=colors[i],
        )
        fig.update_xaxes(title_text=r"$\text{Production volume share}$", row=1, col=i+1, title_standoff=0.1)
    fig.update_xaxes(title_text=r"$\text{Production volume share}$", row=2, title_standoff=0.1)

    fig.update_layout(barmode="relative")

    fig.update_xaxes(range=(0.55, 1.05), col=1)
    fig.update_xaxes(range=(-0.05, 0.45), col=2)
    fig.update_xaxes(range=(-0.05, 0.45), col=3)
    fig.update_yaxes(range=(-1, 44), row=1)
    fig.update_yaxes(range=(-2, iterations_zoomed+2), row=2)

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

    fig.update_yaxes(title_text=r"$\text{Frequency}$", col=1, row=1)
    fig.update_yaxes(title_text=r"$\text{Sample number}$", col=1, row=2)
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
        width=1000,
        height=450,
        legend=dict(
            yanchor="top",
            y=1.0,  # -0.7
            xanchor="left",
            x=0.71,
            orientation='v',
            font=dict(size=13),
            bordercolor=color_darkgray_hex,
            borderwidth=1,
        ),
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )

    # fig.write_image(write_figs / f"{act_id}_implicit_market.eps")
    fig.show()

    print("")
