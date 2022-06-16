import bw2data as bd
from pathlib import Path
import numpy as np
from gsa_framework.utils import read_pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# from gsa_framework.visualization.plotting import plot_correlation_Y1_Y2

from akula.markets import DATA_DIR


# option = "implicit-markets"
# option = "liquid-fuels-kilogram"
# option = "ecoinvent-parameterization"
option = "entso-timeseries"

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
color_psi_red = "#d04729"
color_psi_lblue = "#69769e"
opacity = 0.9
num_bins = 100


def plot_correlation_Y1_Y2(
        Y1,
        Y2,
        start=0,
        end=50,
        trace_name1="Y1",
        trace_name2="Y2",
        trace_name3="Scatter plot",
        color1="#636EFA",
        color2="#EF553B",
        color3="#A95C9A",
        xaxes1_title_text=None,
        yaxes1_title_text="Values",
        xaxes2_title_text="Values",
        yaxes2_title_text="Values",
        showtitle=True,
):
    """Function that plots subset of datapoints of ``Y1`` and ``Y2``, used by Validation class."""
    x = np.arange(start, end)
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=Y1[start:end] - Y2[start:end],
            name=trace_name1,
            mode="markers",
            marker=dict(color=color1, symbol='diamond-tall', size=8),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=Y1,
            y=Y2,
            name=trace_name3,
            mode="markers",
            marker=dict(
                color=color3,
                line=dict(
                    width=1,
                    # color="#782e69",
                    color="black",
                ),
            ),
            showlegend=True,
            opacity=0.65,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        width=800,
        height=220,
        legend=dict(x=0.4, y=1.0),  # on top
        xaxis1=dict(domain=[0.0, 0.63]),
        xaxis2=dict(domain=[0.78, 1.0]),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    if xaxes1_title_text is None:
        text = "Subset of {0}/{1} datapoints".format(end - start, Y1.shape[0])
        xaxes1_title_text = text
    fig.update_xaxes(
        title_text=xaxes1_title_text,
        row=1,
        col=1,
    )
    Ymin = min(np.hstack([Y1, Y2]))
    Ymax = max(np.hstack([Y1, Y2]))
    fig.update_yaxes(range=[Ymin, Ymax], title_text=yaxes1_title_text, row=1, col=1)

    fig.update_xaxes(
        range=[Ymin, Ymax],
        title_text=xaxes2_title_text,
        # color=color1,
        row=1,
        col=2,
    )
    fig.update_yaxes(
        range=[Ymin, Ymax],
        title_text=yaxes2_title_text,
        # color=color2,
        row=1,
        col=2,
    )
    if showtitle:
        from scipy.stats import spearmanr

        pearson_coef = np.corrcoef([Y1, Y2])[0, -1]
        spearman_coef, _ = spearmanr(Y1, Y2)
        fig.update_layout(
            title=dict(
                text="Pearson = {:4.3f}, Spearman = {:4.3f}".format(
                    pearson_coef, spearman_coef
                ),
                font=dict(
                    size=14,
                ),
            )
        )

    return fig


if __name__ == "__main__":

    project = "GSA for archetypes"
    bd.projects.set_current(project)
    fp_monte_carlo = Path("write_files") / project.lower().replace(" ", "_") / "monte_carlo"
    fp_monte_carlo.mkdir(parents=True, exist_ok=True)
    write_figs = Path("/Users/akim/PycharmProjects/akula/dev/write_files/paper3")

    fp_option = DATA_DIR / f"{option}.zip"

    iterations = 500
    seed = 11111000
    fp_monte_carlo_base = fp_monte_carlo / f"base.{iterations}.{seed}.pickle"
    fp_monte_carlo_option = fp_monte_carlo / f"{option}.{iterations}.{seed}.pickle"

    Ybase = read_pickle(fp_monte_carlo_base)
    Yoption = read_pickle(fp_monte_carlo_option)

    # Plot histograms
    ex_offset = 1756.551 - 1135.223913835494
    Y1 = np.array(Ybase) + ex_offset
    Y2 = np.array(Yoption) + ex_offset

    mask1 = np.logical_and(Y1 > np.percentile(Y1, 5), Y1 < np.percentile(Y1, 95))
    mask2 = np.logical_and(Y2 > np.percentile(Y2, 5), Y2 < np.percentile(Y2, 95))
    mask = np.logical_and(mask1, mask2)

    Y1 = Y1[mask]
    Y2 = Y2[mask]

    trace_name1 = r"$\text{Difference between independent and correlated sampling}$"
    if option == "liquid-fuels-kilogram":
        use_option = "carbon balancing"
    elif option == "entso-timeseries":
        use_option = "ENTSO electricity mixes"
    elif option == "implicit-markets":
        use_option = "uncertainties in markets"
    elif option == "ecoinvent-parameterization":
        use_option = "parameterization"
    # else:
    #     use_option = option

    trace_name2 = r'$\text{Sampling with ' + f'{use_option.replace("-", " ")}' + '}$'
    trace_name3 = r'$\text{Scatter plot data points}$'

    lcia_text = r"$\text{LCIA scores, [kg CO}_2\text{-eq]}$"
    lcia_text2 = r"$\Delta \text{ LCIA scores, [kg CO}_2\text{-eq]}$"

    fig = plot_correlation_Y1_Y2(
        Y1,
        Y2,
        start=0,
        end=50,
        trace_name1=trace_name1,
        trace_name2=trace_name2,
        trace_name3=trace_name3,
        yaxes1_title_text=lcia_text2,
        xaxes2_title_text=lcia_text,
        yaxes2_title_text=lcia_text,
        showtitle=False,
        color1=color_psi_lblue,
        color2=color_psi_red,
        color3=color_darkgray_hex,
    )
    fig.update_xaxes(title_text=r"$\text{Sample number}$", col=1,)

    fig.update_xaxes(
        title_standoff=9,
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
    fig.update_yaxes(
        title_standoff=7,
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
    fig.update_xaxes(range=(1680, 2020), col=2)
    fig.update_yaxes(range=(-160, 160), col=1)
    fig.update_yaxes(range=(1680, 2020), col=2)
    fig.update_layout(
        width=700,
        height=190,
        legend=dict(
            yanchor="bottom",
            y=1.1,
            xanchor="left",
            x=0,
            orientation='v',
            font=dict(size=13),
            bordercolor=color_darkgray_hex,
            borderwidth=1,
            itemsizing="constant",
        ),
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )
    fig.write_image(write_figs / f"mc.{option}.{iterations}.{seed}.eps")
    # fig.show()
