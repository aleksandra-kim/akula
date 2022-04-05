import numpy as np
from pathlib import Path
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from fs.zipfs import ZipFS
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from akula.parameterized_exchanges import DATA_DIR
# from akula.electricity.create_datapackages import create_average_entso_datapackages
from akula.markets import get_dirichlet_scale
from gsa_framework.utils import read_pickle
from scipy.stats import lognorm, dirichlet


if __name__ == "__main__":

    project = "GSA for archetypes"
    bd.projects.set_current(project)

    co = bd.Database('swiss consumption 1.0')
    # fu = [act for act in co if "ch hh average consumption aggregated, years 151617" == act['name']][0]
    # demand = {fu: 1}
    # method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    #
    # fu_mapped, packages, _ = bd.prepare_lca_inputs(demand=demand, method=method, remapping=False)
    #
    # fp_flocal_sa = DATA_DIR / "local-sa-1e+01-liquid-fuels-kilogram.zip"
    # dp = bwp.load_datapackage(ZipFS(str(fp_flocal_sa)))
    #
    # tindices = dp.get_resource('local-sa-liquid-fuels-kilogram-tech.indices')[0]
    # tdata = dp.get_resource('local-sa-liquid-fuels-kilogram-tech.data')[0]
    # bindices = dp.get_resource('local-sa-liquid-fuels-kilogram-bio.indices')[0]
    # bdata = dp.get_resource('local-sa-liquid-fuels-kilogram-bio.data')[0]
    #
    # lca = bc.LCA(demand=fu_mapped, data_objs=packages, use_distributions=False)
    # lca.lci()
    # lca.lcia()
    # print("\n--> data_objs=packages, use_distributions=False")
    # print([(lca.score, next(lca)) for _ in range(5)])
    #
    # lca = bc.LCA(demand=fu_mapped, data_objs=packages+[dp], use_distributions=False, use_arrays=True)
    # lca.lci()
    # lca.lcia()
    # print("\n--> data_objs=packages+[dp], use_distributions=False")
    # print([(lca.score, next(lca)) for _ in range(5)])

    markets_type = "implicit"
    fp_markets = DATA_DIR / f"{markets_type}-markets.pickle"
    markets = read_pickle(fp_markets)
    act = list(markets)[0]
    exc_inf = markets[act][0:2] + markets[act][-1:]
    exc_noninf = markets[act][2:-1]

    amounts_inf = np.array([exc['amount'] for exc in exc_inf])
    amounts_exchanges_inf = {amounts_inf[i]: exc_inf[i] for i in range(len(amounts_inf))}
    scale_inf = get_dirichlet_scale(
        amounts_exchanges_inf, fit_variance=True, based_on_contributions=False, use_threshold=False,
    )

    amounts_noninf = np.array([exc['amount'] for exc in exc_noninf])
    amounts_exchanges_noninf = {amounts_noninf[i]: exc_noninf[i] for i in range(len(amounts_noninf))}
    scale_noninf = get_dirichlet_scale(
        amounts_exchanges_noninf, fit_variance=True, based_on_contributions=False, use_threshold=False,
    )

    num_samples = 10000
    alpha_inf = scale_inf * np.array(list(amounts_exchanges_inf.keys()))
    data_inf = dirichlet.rvs(alpha_inf, size=num_samples)
    data_inf = data_inf * amounts_inf.sum()

    alpha_noninf = scale_noninf * np.array(list(amounts_exchanges_noninf.keys()))
    data_noninf = dirichlet.rvs(alpha_noninf, size=num_samples)
    data_noninf = data_noninf * amounts_noninf.sum()

    dict_ = [(exc_inf[i], data_inf[:, i]) for i in range(len(exc_inf))] + \
            [(exc_noninf[i], data_noninf[:, i]) for i in range(len(exc_noninf))]

    amounts_original = np.array([exc['amount'] for exc in markets[act]])
    amounts_exchanges_original = {amounts_original[i]: markets[act][i] for i in range(len(amounts_original))}
    scale_original = get_dirichlet_scale(
        amounts_exchanges_original, fit_variance=True, based_on_contributions=False, use_threshold=True
    )
    alpha_original = scale_original * np.array(list(amounts_exchanges_original.keys()))
    data_original = dirichlet.rvs(alpha_original, size=num_samples)
    data_original = data_original * amounts_original.sum()

    nrows = len(amounts_exchanges_inf) + len(amounts_exchanges_noninf)

    fig = make_subplots(
        rows=nrows,
        cols=1,
    )

    num_bins = 200
    showlegend = True
    order = [0, 1, -1,  2, 3]
    for i, element in enumerate(dict_):
        exc = element[0]

        Y = element[1]
        bins_ = np.linspace(min(Y), max(Y), num_bins+1, endpoint=True)
        bins_ = np.linspace(0, 1, num_bins+1, endpoint=True)
        Y_samples, _ = np.histogram(Y, bins=bins_, density=True)

        Y1 = data_original[:, order[i]]
        # bins_ = np.linspace(min(Y1), max(Y1), num_bins+1, endpoint=True)
        Y_samples_original, _ = np.histogram(Y1, bins=bins_, density=True)

        # Given distribution
        assert exc['uncertainty type'] == 2
        loc = exc['loc']
        scale = exc['scale']
        midbins = (bins_[1:]+bins_[:-1])/2
        Y_distr = lognorm.pdf(midbins, s=scale, scale=np.exp(loc))
        distance = np.sqrt(sum(Y_distr-Y_samples)**2)/max(Y_distr)

        # fig.add_trace(
        #     go.Scatter(
        #         x=midbins,
        #         y=Y_samples,
        #         line_color='blue',
        #         name='Dirichlet samples',
        #         showlegend=showlegend,
        #     ),
        #     row=i+1,
        #     col=1,
        # )
        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_samples_original,
                line_color='green',
                name='Dirichlet samples one scale for all',
                showlegend=showlegend,
            ),
            row=i+1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_distr,
                line_color='red',
                name='Defined distribution',
                showlegend=showlegend,
            ),
            row=i+1,
            col=1,
        )
        showlegend = False
    fig.update_layout(
        width=600,
        height=250*nrows,
        legend=dict(
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0.01,
            orientation='h',
        )
    )
    fig.show()

    print("")



