import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
import numpy as np
import pandas as pd
from pathlib import Path
from consumption_model_ch.utils import get_habe_filepath
from gsa_framework.utils import read_pickle, write_pickle
import plotly.graph_objects as go
from scipy.stats import norm
from consumption_model_ch.consumption_fus import get_households_numpeople_archetype

from akula.sensitivity_analysis import get_mask

HABE_CODES_DICT = {
    'anzahldesktopcomputer11': 'cg_nodesktoppcs',
    'anzahllaptopcomputer11': 'cg_nolaptops',
    'anzahldrucker11': 'cg_noprinters',
}


def compute_ci(y, conf_level=0.95):
    z = norm.ppf(0.5 + conf_level / 2)
    mean = np.mean(y)
    half_width = z*np.std(y)
    return mean, half_width


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
    files = ['Ausgaben', 'Mengen', 'Konsumgueter', "Standard"]

    # 2. Extract total demand from HABE
    dfs = {}
    for name in files:
        path = get_habe_filepath(dir_habe, year_habe, name)
        df = pd.read_csv(path, sep='\t')
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
            dfs['Ausgaben'].set_index('haushaltid'),
            dfs['Konsumgueter'].set_index('haushaltid'),
            dfs['Standard'].set_index('haushaltid'),
        ],
        join='inner',
        axis=1,
    )

    if archetype_label is not None:
        if fp_habe_clustering is not None:
            df_hh_ppl_archetype = get_households_numpeople_archetype(dir_habe, fp_habe_clustering, year_habe)
            weighted_ausgaben = pd.concat(
                [
                    df_hh_ppl_archetype,
                    weighted_ausgaben,
                ],
                join='inner',
                axis=1
            )
            weighted_ausgaben = weighted_ausgaben[weighted_ausgaben['cluster_label_name'] == archetype_label]
            archetype_act = [
                act for act in co if f"archetype {archetype_label} consumption, years {year_habe}" in act['name']
            ]
            assert len(archetype_act) == 1
            archetype_act = archetype_act[0]
            for exc in archetype_act.exchanges():
                if 'mx' in exc.input['code']:
                    weighted_ausgaben[exc.input['code']] = exc.amount
            weighted_ausgaben.drop(columns=['cluster_label_name'], inplace=True)
        else:
            print("Please provide the path for household clustering!")
            return
    print(archetype_label, len(weighted_ausgaben))
    # weighted_ausgaben = weighted_ausgaben.div(weighted_ausgaben['n_people'], axis=0)
    weighted_ausgaben = weighted_ausgaben.div(weighted_ausgaben['bruttoeinkommen08'], axis=0)
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
fp_monte_carlo = Path("write_files") / project.lower().replace(' ', '_') / "monte_carlo" / f"archetypes_{year}"
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
archetype_labels = ['ALL']

color_bg = "rgb(29,105,150)"
color_bg_fg = "rgb(148, 52, 110)"
color_gray_hex = "#b2bcc0"
color_darkgray_hex = "#485063"
color_black_hex = "#212931"
lca_scores_axis_title = r"$\text{LCIA scores, [kg CO}_2\text{-eq per CHF}]$"

for archetype_label in archetype_labels:
    co_static = co_dp
    fp_monte_carlo_bg = fp_monte_carlo / "{}.{}.{}.{}.pickle".format(
        archetype_label, "bg", iterations, seed
    )
    dps_bg = [me_dp, bs_dp, ei_dp, co_static]

    options = {
        "bg": {
            "fp": fp_monte_carlo_bg,
            "dps": dps_bg,
        },
    }
    if archetype_label == 'ALL':
        fu = [act for act in co_bw if f"ch hh average consumption aggregated, years {year}" in act['name']]
        assert len(fu) == 1
        fu = fu[0]
        fu_mapped, _, _ = bd.prepare_lca_inputs(demand={fu: 1}, method=method, remapping=False)
    else:
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
        co_uncertain = bwp.create_datapackage(sequential=True)
        co_uncertain.add_persistent_array(
            matrix="technosphere_matrix",
            indices_array=use_indices,
            name="swiss_consumptwion_1.0_technosphere_matrix",
            data_array=use_data,
            flip_array=use_flip,
        )
        fp_monte_carlo_bg_fg = fp_monte_carlo / "{}.{}.{}.{}.pickle".format(
            archetype_label + ".per_income", "bg+fg", iterations, seed
        )

        dps_bg_fg = [me_dp, bs_dp, ei_dp, co_static, co_uncertain]
        options["bg+fg"] = {
            "fp": fp_monte_carlo_bg_fg,
            "dps": dps_bg_fg,
        }

        ppl_per_hh = fu['ppl_per_household']
        income_per_hh = fu['income_per_household']

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

    if archetype_label != "ALL":
        temp = [x for x in scores['bg+fg'] if x == x]
        temp = [x for x in temp if np.percentile(temp, 5) < x < np.percentile(temp, 95)]
        bin_min = min(temp)  # min(scores['bg+fg'])
        bin_max = max(temp)  # max(scores['bg+fg'])
    else:
        bin_min = min(scores['bg'])
        bin_max = max(scores['bg'])
    num_bins = 100
    opacity = 0.65

    bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)

    fig = go.Figure()

    if archetype_label != "ALL":
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
        freq2, bins2 = np.histogram(np.array(scores['bg'])/income_per_hh, bins=bins_)
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
            range=(bin_min, bin_max),
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

        filepath_fig = fp_monte_carlo / "figures" / \
            f"{archetype_label}.per_income.uncertainty_bg_fg.{iterations}.{seed}.pdf"
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

# Confidence interval plots
stats_bg_fg, stats_bg, incomes = [], [], []
scores_archetypes = np.array([])
archetype_labels = sorted([a for a in archetype_labels if a != "ALL"])
for archetype_label in archetype_labels:
    fu = [act for act in co_bw if f"archetype {archetype_label} consumption, years {year}" in act['name']]
    assert len(fu) == 1
    fu = fu[0]
    income_per_hh = fu["income_per_household"]
    incomes.append(income_per_hh)
    fp_monte_carlo_bg_fg = fp_monte_carlo / "{}.{}.{}.{}.pickle".format(
        archetype_label + ".per_income", "bg+fg", iterations, seed
    )
    scores_bg_fg = read_pickle(fp_monte_carlo_bg_fg)
    scores_bg_fg = [x for x in scores_bg_fg if x == x]
    scores_bg_fg = [x for x in scores_bg_fg if np.percentile(scores_bg_fg, 5) < x < np.percentile(scores_bg_fg, 95)]
    stats_bg_fg.append(compute_ci(scores_bg_fg))

    fp_monte_carlo_bg = fp_monte_carlo / "{}.{}.{}.{}.pickle".format(
        archetype_label, "bg", iterations, seed
    )
    scores_bg = read_pickle(fp_monte_carlo_bg) / income_per_hh
    scores_bg = [x for x in scores_bg if x == x]
    scores_bg = [x for x in scores_bg if np.percentile(scores_bg, 5) < x < np.percentile(scores_bg, 95)]
    stats_bg.append(compute_ci(scores_bg))

    scores_archetypes = np.hstack([scores_archetypes, scores_bg_fg])

fp_monte_carlo_bg = fp_monte_carlo / "{}.{}.{}.{}.pickle".format(
    "ALL", "bg", iterations, seed
)
scores_archetypes = np.random.choice(scores_archetypes, size=iterations)
scores_bg_all = read_pickle(fp_monte_carlo_bg) / np.mean(incomes)
stats_bg_all = compute_ci(scores_bg_all)
stats_archetypes = compute_ci(scores_archetypes)
stats_bg_fg = np.array(stats_bg_fg + [stats_archetypes])
stats_bg = np.array(stats_bg + [stats_bg_all])

color_bg_fga = "rgb(148, 52, 110, 0.2)"
fig = go.Figure()

x = np.arange(len(archetype_labels) + 1)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x-0.1,
        y=stats_bg_fg[:, 0],
        error_y=dict(
            type='data',  # value of error bar given in data coordinates
            array=stats_bg_fg[:, 1],
            visible=True,
            color=color_bg_fg,
        ),
        mode='markers',
        marker=dict(color=color_bg_fg, size=8),
        name="Background and foreground vary",
    )
)
fig.add_trace(
    go.Scatter(
        x=x+0.1,
        y=stats_bg[:, 0],
        error_y=dict(
            type='data',  # value of error bar given in data coordinates
            array=stats_bg[:, 1],
            visible=True,
            color=color_bg,
        ),
        mode='markers',
        marker=dict(color=color_bg, size=8),
        name="Only background varies",
    )
)

fig.update_xaxes(title_text="Archetypes", )
fig.update_yaxes(title_text="LCIA scores, kg CO2-eq per CHF", )
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=x,
        ticktext=archetype_labels + ["ALL"],
    ),
    height=600,
    width=1400,
)
filepath_fig = fp_monte_carlo / "figures" / \
               f"_confidence_intervals.per_income.uncertainty_bg_fg.{iterations}.{seed}.html"
# fig.write_image(filepath_fig.as_posix())
fig.write_html(filepath_fig.as_posix())



# All households in one plot
############################
scores = {
    "bg+fg": scores_archetypes,
    "bg": scores_bg_all,
}
temp = [x for x in scores['bg+fg'] if x == x]
temp = [x for x in temp if np.percentile(temp, 5) < x < np.percentile(temp, 95)]
bin_min = min(temp)  # min(scores['bg+fg'])
bin_max = max(temp)  # max(scores['bg+fg'])
num_bins = 100
opacity = 0.65

bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)


# Background + foreground
fig = go.Figure()
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
freq2, bins2 = np.histogram(np.array(scores['bg']), bins=bins_)
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
    range=(bin_min, bin_max),
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
        'text': f"All households",
    }
)

filepath_fig = fp_monte_carlo / "figures" / \
    f"ALL.per_income.uncertainty_bg_fg.{iterations}.{seed}.pdf"
fig.write_image(filepath_fig.as_posix())
