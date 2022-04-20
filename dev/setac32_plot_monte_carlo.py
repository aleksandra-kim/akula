import bw2data as bd
from pathlib import Path
import numpy as np
from gsa_framework.utils import read_pickle
from gsa_framework.visualization.plotting import plot_correlation_Y1_Y2

from akula.markets import DATA_DIR


# option = "implicit-markets"
# option = "liquid-fuels-kilogram"
option = "ecoinvent-parameterization"
# option = "entso-timeseries"

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
    ex_offset = 660
    Y1 = np.array(Ybase) + ex_offset
    Y2 = np.array(Yoption) + ex_offset

    mask1 = np.logical_and(Y1 > np.percentile(Y1, 5), Y1 < np.percentile(Y1, 95))
    mask2 = np.logical_and(Y2 > np.percentile(Y2, 5), Y2 < np.percentile(Y2, 95))
    mask = np.logical_and(mask1, mask2)

    Y1 = Y1[mask]
    Y2 = Y2[mask]

    trace_name1 = r"$\text{Independent sampling}$"
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

    lcia_text = r"$\text{LCIA scores, [kg CO}_2\text{-eq]}$"

    fig = plot_correlation_Y1_Y2(
        Y1,
        Y2,
        start=0,
        end=50,
        trace_name1=trace_name1,
        trace_name2=trace_name2,
        yaxes1_title_text=lcia_text,
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
        range=(1700, 2050),
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
    fig.update_xaxes(range=(1700, 2050), col=2)
    fig.update_layout(
        width=700,
        height=220,
        legend=dict(
            yanchor="bottom",
            y=1.1,
            xanchor="left",
            x=0,
            orientation='v',
            font=dict(size=13),
            bordercolor=color_darkgray_hex,
            borderwidth=1,
        ),
        margin=dict(t=60, b=10, l=10, r=10),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )
    fig.write_image(write_figs / f"mc.{option}.{iterations}.{seed}.pdf")
    # fig.show()
