import numpy as np
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from fs.zipfs import ZipFS
from pathlib import Path
from gsa_framework.utils import read_pickle, write_pickle
import plotly.graph_objects as go
import plotly.figure_factory as ff
from copy import deepcopy


def get_one(db_name, **kwargs):
    possibles = [
        act
        for act in bd.Database(db_name)
        if all(act.get(key) == value for key, value in kwargs.items())
    ]
    if len(possibles) == 1:
        return possibles[0]
    else:
        raise ValueError(
            f"Couldn't get exactly one activity in database `{db_name}` for arguments {kwargs}"
        )


if __name__ == "__main__":

    option = "uncertain"

    project = 'GSA for archetypes'
    bd.projects.set_current(project)

    co = bd.Database('swiss consumption 1.0')
    fu = [act for act in co if "ch hh average consumption aggregated, years 151617" == act['name']][0]
    write_dir = Path("write_files") / project.lower().replace(" ", "_") \
        / fu['name'].lower().replace(" ", "_").replace(",", "")

    if option == "certain":
        method = ("IPCC 2013", "climate change", "GWP 100a")
    elif option == "uncertain":
        method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    ch_low = get_one("ecoinvent 3.8 cutoff", name="market for electricity, low voltage", location="CH")
    fu_mapped, pkgs, _ = bd.prepare_lca_inputs(
        demand={ch_low: 1}, method=method, remapping=False,
    )

    dp_timeseries = bwp.load_datapackage(ZipFS("../akula/data/entso-timeseries.zip"))
    dp_average = bwp.load_datapackage(ZipFS("../akula/data/entso-average.zip"))

    iterations = 2000

    # Static values, Ecoinvent
    # lca = bc.LCA(
    #     demand=fu_mapped,
    #     data_objs=pkgs,
    #     use_arrays=False,
    #     use_distributions=False
    # )
    # lca.lci()
    # lca.lcia()
    # static_ecoinvent = deepcopy(lca.score)
    static_ecoinvent = 0.048644
    print(static_ecoinvent)

    # Static values, Entso
    lca = bc.LCA(
        demand=fu_mapped,
        data_objs=pkgs + [dp_average],
        use_arrays=False,
        use_distributions=False
    )
    lca.lci()
    lca.lcia()
    static_entso = deepcopy(lca.score)
    print(static_entso)

    # MC simulations, Ecoinvent
    lca = bc.LCA(
        demand=fu_mapped,
        data_objs=(
            pkgs
        ),
        use_arrays=True,
        use_distributions=True
    )
    lca.lci()
    lca.lcia()

    fp_scores_ecoinvent = write_dir / f"mc.ch_low.{iterations}.ecoinvent.{option}.pickle"
    if fp_scores_ecoinvent.exists():
        scores_ecoinvent = read_pickle(fp_scores_ecoinvent)
    else:
        scores_ecoinvent = [lca.score for _ in zip(range(iterations), lca)]
        write_pickle(scores_ecoinvent, fp_scores_ecoinvent)
    scores_ecoinvent = np.array(scores_ecoinvent)

    # MC simulations, Entso-E
    lca = bc.LCA(
        demand=fu_mapped,
        data_objs=(
                pkgs + [dp_timeseries]
        ),
        use_arrays=True,
        use_distributions=True
    )
    lca.lci()
    lca.lcia()

    fp_scores_entso = write_dir / f"mc.ch_low.{iterations}.entso.{option}.pickle"
    if fp_scores_entso.exists():
        scores_entso = read_pickle(fp_scores_entso)
    else:
        scores_entso = [lca.score for _ in zip(range(iterations), lca)]
        write_pickle(scores_entso, fp_scores_entso)
    scores_entso = np.array(scores_entso)


    # Make plots
    ############

    num_bins = 100
    opacity = 0.85
    color_gray_hex = "#b2bcc0"
    color_darkgray_hex = "#485063"
    color_darkgray_rgb = f"rgb(72, 80, 99, {opacity})"
    color_black_hex = "#212931"
    color_pink_rgb = f"rgba(148, 52, 110, {opacity})"
    color_bright_pink_rgb = "#e75480" #"#ff44cc"
    color_psi_lpurple = "#914967"

    Y_ecoinvent = scores_ecoinvent
    Y_entso = scores_entso
    # Y_ecoinvent = Y_ecoinvent[Y_ecoinvent < 0.6]
    # Y_entso = Y_entso[Y_entso < 0.6]

    # bin_min = min(scores_entso.min(), scores_ecoinvent.min())
    # bin_max = max(scores_entso.max(), scores_ecoinvent.max())
    #
    # bins_ = np.linspace(bin_min, bin_max, num_bins+1, endpoint=True)
    # Y_ecoinvent, _ = np.histogram(scores_ecoinvent, bins=bins_, density=True)
    # Y_entso, _ = np.histogram(scores_entso, bins=bins_, density=True)

    group_labels = [r'$\text{Ecoivent}$', r'$\text{ENTSO-E}$']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
        hist_data=[Y_ecoinvent, Y_entso],
        group_labels=group_labels,
        bin_size=.005,
        colors=[color_darkgray_rgb, color_psi_lpurple],
    )

    fig.add_trace(
        go.Scatter(
            x=[static_ecoinvent],
            y=[0],
            mode="markers",
            marker=dict(size=14, symbol="x", color=color_black_hex, opacity=1),
            name=r"$\text{Static LCIA score from ecoinvent}$",
            legendrank=3,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[static_entso],
            y=[0],
            mode="markers",
            marker=dict(
                size=10,
                symbol="diamond-tall",
                color=color_bright_pink_rgb,
                # line=dict(
                #     color=color_pink_rgb,
                #     width=2,
                # )
            ),
            name=r"$\text{Static LCIA score from ENTSO-E}$",
            legendrank=4
        )
    )

    fig.update_xaxes(
        title_text=r"$\text{LCIA scores, [kg CO}_2\text{-eq.]}$",
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

    fig.update_yaxes(title_text=r"$\text{Probability}$")
    fig.layout.yaxis2.update({"title": ""})
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
        width=800,
        height=260,
        legend=dict(
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.6,
            orientation='v',
            font=dict(size=13),
            bordercolor=color_darkgray_hex,
            borderwidth=1,
        ),
        margin=dict(t=30, b=0, l=30, r=0),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        legend_traceorder="reversed"
    )

    fig.show()

    fig.write_image(write_dir / f"_figure_1_ch_low_voltage_{option}.eps")





    # fig = go.Figure()
    #
    # bin_min = min(scores_entso.min(), scores_ecoinvent.min())
    # bin_max = max(scores_entso.max(), scores_ecoinvent.max())
    #
    # bins_ = np.linspace(bin_min, bin_max, num_bins+1, endpoint=True)
    # Y_entso, _ = np.histogram(scores_entso, bins=bins_, density=True)
    # Y_ecoinvent, _ = np.histogram(scores_ecoinvent, bins=bins_, density=True)
    #
    # midbins = (bins_[1:]+bins_[:-1])/2
    #
    # # Ecoivent
    # fig.add_trace(
    #     go.Scatter(
    #         x=midbins,
    #         y=Y_ecoinvent,
    #         name=r"$\text{Ecoinvent samples}$",
    #         showlegend=True,
    #         opacity=opacity,
    #         line=dict(color=color_darkgray_hex, width=1, shape="hvh"),
    #         fill="tozeroy",
    #     ),
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=[np.mean(Y_ecoinvent)],
    #         y=[0],
    #         mode="markers",
    #         marker=dict(size=5, symbol="x", color=color_darkgray_hex),
    #     )
    # )
    #
    # # Entso
    # fig.add_trace(
    #     go.Scatter(
    #         x=midbins,
    #         y=Y_entso,
    #         name=r"$\text{ENTSO-E samples}$",
    #         showlegend=True,
    #         opacity=opacity,
    #         line=dict(color=color_pink_rgb, width=1, shape="hvh"),
    #         fill="tozeroy",
    #     ),
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=[np.mean(Y_entso)],
    #         y=[0],
    #         mode="markers",
    #         marker=dict(size=5, symbol="x", color=color_pink_rgb),
    #     )
    # )
    #
    # fig.show()



