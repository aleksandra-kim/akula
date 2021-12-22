import bw2calc as bc
import bw2data as bd
import bw_processing as bwp

from pathlib import Path
import numpy as np
from fs.zipfs import ZipFS

from akula.virtual_markets import DATA_DIR
from gsa_framework.utils import read_pickle, write_pickle

project = "GSA for archetypes"
bd.projects.set_current(project)
fp_virtual_markets = DATA_DIR / "virtual-markets.zip"
fp_monte_carlo = Path("write_files") / project.lower().replace(" ", "_") / "monte_carlo"
fp_monte_carlo.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":

    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    me = bd.Method(method)
    bs = bd.Database("biosphere3")
    ei = bd.Database("ecoinvent 3.8 cutoff")
    co_name = "swiss consumption 1.0"
    co = bd.Database(co_name)
    list_ = [me, bs, ei, co]
    dps = [
        bwp.load_datapackage(ZipFS(db.filepath_processed()))
        for db in list_
    ]

    hh_average = [act for act in co if "ch hh average consumption aggregated" == act['name']]
    assert len(hh_average) == 1
    demand_act = hh_average[0]
    demand = {demand_act: 1}
    demand_id = {demand_act.id: 1}

    iterations = 2000
    seed = 11111000
    dict_for_lca = dict(
        use_distributions=True,
        use_arrays=True,
        seed_override=seed,
    )
    fp_monte_carlo_virtual_markets_no = fp_monte_carlo / "{}.{}.{}.pickle".format(
        "virtual_markets_no", iterations, seed
    )
    fp_monte_carlo_virtual_markets_yes = fp_monte_carlo / "{}.{}.{}.pickle".format(
        "virtual_markets_yes", iterations, seed
    )

    if fp_monte_carlo_virtual_markets_no.exists():
        scores = read_pickle(fp_monte_carlo_virtual_markets_no)
    else:
        lca = bc.LCA(
            demand_id,
            data_objs=dps,
            **dict_for_lca,
        )
        lca.lci()
        lca.lcia()
        scores = [lca.score for _, _ in zip(lca, range(iterations))]
        write_pickle(scores, fp_monte_carlo_virtual_markets_no)

    if fp_monte_carlo_virtual_markets_yes.exists():
        scores_vm = read_pickle(fp_monte_carlo_virtual_markets_yes)
    else:
        lca_vm = bc.LCA(
            demand_id,
            data_objs=dps + [fp_virtual_markets],
            **dict_for_lca,
        )
        lca_vm.lci()
        lca_vm.lcia()
        scores_vm = [lca_vm.score for _, _ in zip(lca_vm, range(iterations))]
        write_pickle(scores_vm, fp_monte_carlo_virtual_markets_yes)

    # Plot histograms
    from gsa_framework.visualization.plotting import plot_histogram_Y1_Y2, plot_correlation_Y1_Y2
    Y1 = np.array(scores)
    Y2 = np.array(scores_vm)
    trace_name1 = "Without uncertainties in virtual markets"
    trace_name2 = "With uncertainties in virtual markets"
    lcia_text = "LCIA scores, {}".format(me.metadata['unit'])
    fig = plot_histogram_Y1_Y2(
        Y1,
        Y2,
        trace_name1=trace_name1,
        trace_name2=trace_name2,
        xaxes_title_text=lcia_text,
    )
    fig.update_layout(
        width=800,
        height=600,
    )
    fig.show()

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
    fig.show()

