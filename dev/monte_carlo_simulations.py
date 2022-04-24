import bw2calc as bc
import bw2data as bd
import bw_processing as bwp
from pathlib import Path
import numpy as np
from fs.zipfs import ZipFS
from gsa_framework.utils import read_pickle, write_pickle
from gsa_framework.visualization.plotting import plot_histogram_Y1_Y2, plot_correlation_Y1_Y2

from akula.markets import DATA_DIR


option = "implicit-markets"
# option = "generic-markets"
# option = "liquid-fuels-kilogram"
# option = "ecoinvent-parameterization"
# option = "entso-timeseries"

if __name__ == "__main__":
    project = "GSA for archetypes"
    bd.projects.set_current(project)
    fp_monte_carlo = Path("write_files") / project.lower().replace(" ", "_") / "monte_carlo"
    fp_monte_carlo.mkdir(parents=True, exist_ok=True)
    write_figs = Path("/Users/akim/PycharmProjects/akula/dev/write_files/paper3")

    fp_option = DATA_DIR / f"{option}.zip"

    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    me = bd.Method(method)
    bs = bd.Database("biosphere3")
    ei = bd.Database("ecoinvent 3.8 cutoff")
    re = bd.Database("swiss residual electricity mix")
    co_name = "swiss consumption 1.0"
    co = bd.Database(co_name)
    list_ = [me, bs, ei, co, re]
    dps = [
        bwp.load_datapackage(ZipFS(db.filepath_processed()))
        for db in list_
    ]

    hh_average = [act for act in co if "ch hh average consumption aggregated, years 151617" == act['name']]
    assert len(hh_average) == 1
    demand_act = hh_average[0]
    demand = {demand_act: 1}
    demand_id = {demand_act.id: 1}

    lca = bc.LCA(demand, method)
    lca.lci()
    lca.lcia()
    print(lca.score)

    iterations = 500
    seed = 11111000
    dict_for_lca = dict(
        use_distributions=True,
        use_arrays=True,
        seed_override=seed,
    )
    fp_monte_carlo_base = fp_monte_carlo / f"base.{iterations}.{seed}.pickle"
    fp_monte_carlo_option = fp_monte_carlo / f"{option}.{iterations}.{seed}.pickle"

    if fp_monte_carlo_base.exists():
        scores = read_pickle(fp_monte_carlo_base)
    else:
        lca = bc.LCA(
            demand_id,
            data_objs=dps,
            **dict_for_lca,
        )
        lca.lci()
        lca.lcia()
        scores = [lca.score for _, _ in zip(lca, range(iterations))]
        write_pickle(scores, fp_monte_carlo_base)

    if fp_monte_carlo_option.exists():
        scores_option = read_pickle(fp_monte_carlo_option)
    else:
        lca_option = bc.LCA(
            demand_id,
            data_objs=dps + [fp_option],
            **dict_for_lca,
        )
        lca_option.lci()
        lca_option.lcia()
        scores_option = [lca_option.score for _, _ in zip(lca_option, range(iterations))]
        write_pickle(scores_option, fp_monte_carlo_option)

    # Plot histograms
    Y1 = np.array(scores)
    Y2 = np.array(scores_option)
    trace_name1 = "Without uncertainties in {}".format(option.replace("_", " "))
    trace_name2 = "With uncertainties in {}".format(option.replace("_", " "))
    lcia_text = "LCIA scores, {}".format(me.metadata['unit'])

    # fig = plot_histogram_Y1_Y2(
    #     Y1,
    #     Y2,
    #     trace_name1=trace_name1,
    #     trace_name2=trace_name2,
    #     xaxes_title_text=lcia_text,
    # )
    # fig.update_layout(
    #     width=800,
    #     height=600,
    # )
    # fig.show()

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
    )
    fig.update_layout(
        height=300,
        legend=dict(
            yanchor="top",
            y=-0.7,
            xanchor="center",
            x=0.5,
            orientation='v',
            font=dict(size=13),
            borderwidth=1,
        ),
        # margin=dict(t=100, b=10, l=10, r=0),
    )
    # fig.write_image(write_figs / f"mc.{option}.{iterations}.{seed}.pdf")
    fig.show()
