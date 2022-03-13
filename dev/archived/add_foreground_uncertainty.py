import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
import numpy as np
import pandas as pd
from pathlib import Path
from consumption_model_ch.utils import get_habe_filepath
from gsa_framework.utils import read_pickle, write_pickle
import plotly.graph_objects as go
from consumption_model_ch.consumption_fus import get_households_numpeople_archetype

from akula.sensitivity_analysis import get_mask

HABE_CODES_DICT = {
    'anzahldesktopcomputer11': 'cg_nodesktoppcs',
    'anzahllaptopcomputer11': 'cg_nolaptops',
    'anzahldrucker11': 'cg_noprinters',
}


# Foreground uncertainty weighted by number of people per household
def get_household_data_weighted(
    indices,
    year_habe,
    consumption_db_name="swiss consumption 1.0",
    archetype_label=None,
    fp_habe_clustering=None,
):
    # 1. Get some metadata from the consumption database
    co = bd.Database(consumption_db_name)
    dir_habe = co.metadata['dir_habe']
    files = ['Ausgaben', 'Mengen', 'Konsumgueter']

    # 2. Extract total demand from HABE
    dfs = {}
    for name in files:
        path = get_habe_filepath(dir_habe, year_habe, name)
        df = pd.read_csv(path, sep='\t')
        # if name == 'Personen':
        #     num_personen = dict(Counter(df["HaushaltID"]))
        #     num_personen = [{'haushaltid': k, "n_personen": v} for k, v in num_personen.items()]
        #     dfs[name] = pd.DataFrame.from_dict(num_personen)
        # else:
        df.columns = [col.lower() for col in df.columns]
        dfs[name] = df

    codes_co_db = sorted([act['code'] for act in co])
    columns_a = dfs['Ausgaben'].columns.values
    columns_m = [columns_a[0]]
    codes_m = []
    for code_a in columns_a[1:]:
        code_m = code_a.replace('a', 'm')
        if code_m in codes_co_db:
            columns_m.append(code_m)
            codes_m.append(code_m)
        else:
            columns_m.append(code_a)
    dfs['Ausgaben'].columns = columns_m
    columns_k = [HABE_CODES_DICT.get(col, col) for col in dfs['Konsumgueter'].columns]
    dfs['Konsumgueter'].columns = columns_k
    # Replace ausgaben data with mengen data
    for code_m in codes_m:
        dfs['Ausgaben'][code_m] = dfs['Mengen'][code_m].values
    weighted_ausgaben = pd.concat(
        [
            # dfs['Personen'].set_index('haushaltid'),
            dfs['Ausgaben'].set_index('haushaltid'),
            dfs['Konsumgueter'].set_index('haushaltid'),
        ],
        join='inner',
        axis=1,
    )
    # people_per_household = sum(weighted_ausgaben['n_personen'].values) / len(weighted_ausgaben)
    # co.metadata.update({f'people per household, years {year_habe}': people_per_household})
    # weighted_ausgaben = weighted_ausgaben.div(weighted_ausgaben['n_personen'], axis=0)

    if archetype_label is not None:
        if fp_habe_clustering is not None:
            df_hh_ppl_archetype = get_households_numpeople_archetype(dir_habe, fp_habe_clustering, year_habe)

            # df_habe_clustering = pd.read_csv(fp_habe_clustering)
            # df_habe_clustering.columns = [col.lower() for col in df_habe_clustering.columns]
            weighted_ausgaben = pd.concat(
                [
                    df_hh_ppl_archetype,
                    weighted_ausgaben,
                ],
                join='inner',
                axis=1
            )
            # ppl_per_archetype = sum(weighted_ausgaben['n_people'].values) / len(weighted_ausgaben)
            weighted_ausgaben = weighted_ausgaben[weighted_ausgaben['cluster_label_name'] == archetype_label]
            archetype_act = [
                act for act in co if f"archetype {archetype_label} consumption, years {year_habe}" in act['name']
            ]
            assert len(archetype_act) == 1
            archetype_act = archetype_act[0]
            # ppl_per_archetype = sum(weighted_ausgaben['n_people'].values) / len(weighted_ausgaben)
            # print(ppl_per_archetype)
            for exc in archetype_act.exchanges():
                if 'mx' in exc.input['code']:
                    weighted_ausgaben[exc.input['code']] = exc.amount
            # use_columns = sorted(list(set(weighted_ausgaben.columns) - {'cluster_label_name'}))
            weighted_ausgaben.drop(columns=['cluster_label_name'], inplace=True)
        else:
            print("Please provide the path for household clustering!")
            return
    print(archetype_label, len(weighted_ausgaben))
    weighted_ausgaben = weighted_ausgaben.div(weighted_ausgaben['n_people'], axis=0)
    weighted_ausgaben = weighted_ausgaben.reset_index()

    # all_consumption_codes = [act['code'] for act in co]
    # codes_to_ignore = [code for code in weighted_ausgaben.iloc[0].index if code not in all_consumption_codes]
    data = np.zeros((0, len(weighted_ausgaben)))
    for inds in indices:
        input_code = bd.get_activity(inds[0])['code']
        # if input_code not in codes_to_ignore:
        data_current = weighted_ausgaben[input_code].values
        data = np.vstack([data, data_current])
    return data


project = 'GSA for archetypes'
bd.projects.set_current(project)
year = '091011'
fp_archetype_clustering = "/Users/akim/Documents/LCA_files/db_andi_091011/habe_clustering.csv"
fp_monte_carlo = Path("write_files") / 'GSA for archetypes'.replace(' ', '_') / "monte_carlo" / f"archetypes_{year}"
fp_monte_carlo.mkdir(parents=True, exist_ok=True)
iterations = 400
seed = 11111000

co_name = 'swiss consumption 1.0'
co_bw = bd.Database(co_name)
bs_dp = bd.Database('biosphere3').datapackage()
ei_dp = bd.Database('ecoinvent 3.8 cutoff').datapackage()
co_dp = bd.Database('swiss consumption 1.0').datapackage()
co_indices = co_dp.get_resource("swiss_consumption_1.0_technosphere_matrix.indices")[0]
co_flip = co_dp.get_resource("swiss_consumption_1.0_technosphere_matrix.flip")[0]
method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
me_dp = bd.Method(method).datapackage()

archetype_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
    'O', 'OA', 'OB', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
]

for archetype_label in archetype_labels:

    fu = [act for act in co_bw if f"archetype {archetype_label} consumption, years {year}" in act['name']]
    assert len(fu) == 1
    fu = fu[0]
    fu_mapped, _, _ = bd.prepare_lca_inputs(demand={fu: 1}, method=method, remapping=False)

    use_indices = [(exc.input.id, fu.id) for exc in fu.exchanges() if exc['type'] != 'production']
    use_mask = get_mask(co_indices, use_indices)
    use_flip = co_flip[use_mask]
    household_data = get_household_data_weighted(
        use_indices,
        year,
        co_name,
        archetype_label,
        fp_archetype_clustering,
    )
    choice = np.sort(np.random.choice(household_data.shape[1], iterations, replace=True))
    use_data = household_data[:, choice]
    use_indices = np.array(use_indices, dtype=[('row', '<i4'), ('col', '<i4')])

    fp_monte_carlo_bg_fg = fp_monte_carlo / "{}.{}.{}.{}.pickle".format(
        archetype_label, "bg+fg", iterations, seed
    )
    fp_monte_carlo_bg = fp_monte_carlo / "{}.{}.{}.{}.pickle".format(
        archetype_label, "bg", iterations, seed
    )

    co_static = co_dp
    co_uncertain = bwp.create_datapackage(sequential=True)
    co_uncertain.add_persistent_array(
        matrix="technosphere_matrix",
        indices_array=use_indices,
        name="swiss_consumptwion_1.0_technosphere_matrix",
        data_array=use_data,
        flip_array=use_flip,
    )

    dps_bg = [me_dp, bs_dp, ei_dp, co_static]
    dps_bg_fg = [me_dp, bs_dp, ei_dp, co_static, co_uncertain]

    options = {
        "bg+fg": {
            "fp": fp_monte_carlo_bg_fg,
            "dps": dps_bg_fg,
        },
        "bg": {
            "fp": fp_monte_carlo_bg,
            "dps": dps_bg,
        },
    }

    scores = {}
    for option, data in options.items():
        print(option)
        fp = data['fp']
        dps = data['dps']
        if fp.exists():
            scores[option] = read_pickle(fp)
        else:
            if option == 'fg':
                use_distributions = False
            else:
                use_distributions = True
            dict_for_lca = dict(
                use_distributions=use_distributions,
                use_arrays=True,
                seed_override=seed,
            )
            lca_new = bc.LCA(
                fu_mapped,
                data_objs=dps,
                **dict_for_lca,
            )
            lca_new.lci()
            lca_new.lcia()
            scores_current = []
            for i in range(iterations):
                next(lca_new)
                scores_current.append(lca_new.score)
            scores[option] = scores_current
            write_pickle(scores[option], fp)

    ppl_per_hh = fu['ppl_per_household']

    bin_min = min(scores['bg+fg'])
    bin_max = max(scores['bg+fg'])
    num_bins = 100
    opacity = 0.65

    bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)
    color_bg = "rgb(29,105,150)"
    color_bg_fg = "rgb(148, 52, 110)"
    color_gray_hex = "#b2bcc0"
    color_darkgray_hex = "#485063"
    color_black_hex = "#212931"
    lca_scores_axis_title = r"$\text{LCIA scores, [kg CO}_2\text{-eq}]$"

    fig = go.Figure()

    # Background + foreground
    freq1, bins1 = np.histogram(scores['bg+fg'], bins=bins_)
    fig.add_trace(
        go.Scatter(
            x=bins1,
            y=freq1,
            name=r"$\text{Background and foreground vary}$",
            opacity=opacity,
            line=dict(color=color_bg_fg, width=1, shape="hvh"),
            showlegend=True,
            fill="tozeroy",
        ),
    )

    # Background
    freq2, bins2 = np.histogram(np.array(scores['bg'])/ppl_per_hh, bins=bins_)
    fig.add_trace(
        go.Scatter(
            x=bins2,
            y=freq2,
            name=r"$\text{Only background varies}$",
            opacity=opacity,
            line=dict(color=color_bg, width=1, shape="hvh"),
            showlegend=True,
            fill="tozeroy",
        ),
    )

    fig.update_xaxes(
        title_text=lca_scores_axis_title,
        showgrid=True,
        gridwidth=1,
        gridcolor=color_gray_hex,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=color_gray_hex,
        showline=True,
        linewidth=1,
        linecolor=color_gray_hex,
    )
    fig.update_yaxes(
        title_text=r"$\text{Frequency}$",
        range=[-10, max(freq2)+50],
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
        width=600,
        height=250,
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        legend=dict(
            x=0.7,
            y=0.90,
            orientation="v",
            xanchor="center",
            font=dict(size=12),
            bordercolor=color_darkgray_hex,
            borderwidth=1,
        ),
        margin=dict(l=65, r=0, t=60, b=0),
        title={
            'text': f"Archetype {archetype_label}",
        }
    )

    filepath_fig = fp_monte_carlo / "figures" / f"{archetype_label}.uncertainty_bg_fg.{iterations}.{seed}.pdf"
    fig.write_image(filepath_fig.as_posix())

    # start = 0
    # end = 200
    # fig = go.Figure()
    # for option, data in scores.items():
    #     data = np.array(data)*12
    #     if option == 'bg':
    #         data /= ppl_per_hh
    #     fig.add_trace(
    #         go.Scatter(
    #             x=np.arange(iterations),
    #             y=data[start:end],
    #             name=option,
    #             mode="lines+markers",
    #             showlegend=True,
    #         ),
    #     )
    #
    # fig.show()
