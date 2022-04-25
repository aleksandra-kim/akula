import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from copy import deepcopy


from sensitivity_analysis import get_mask

color_gray_hex = "#b2bcc0"
color_darkgray_hex = "#485063"
color_black_hex = "#212931"


def update_fig_axes(fig):
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
        legend=dict(
            bordercolor=color_darkgray_hex,
            borderwidth=1,
        ),
        margin=dict(t=40, b=10, l=10, r=0),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )
    return fig


def setup_bw_project(years="151617"):
    project = "GSA for archetypes"
    bd.projects.set_current(project)

    co = bd.Database('swiss consumption 1.0')
    fu = [act for act in co if f"ch hh average consumption aggregated, years {years}" == act['name']][0]
    demand = {fu: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")

    lca = bc.LCA(demand=demand, method=method, use_distributions=False)
    lca.lci()
    lca.lcia()

    return lca


def get_activities_from_indices(indices):

    bd.projects.set_current("GSA for archetypes")
    activities = {}

    if indices is not None:

        cols = sorted(set(indices['col']))
        for col in cols:

            rows = sorted(indices[indices['col'] == col]['row'])
            act = bd.get_activity(int(col))

            exchanges = []
            for exc in act.exchanges():
                if exc.input.id in rows:
                    exchanges.append(exc)

            if len(exchanges) > 0:
                activities[act] = exchanges

    return activities


def create_static_datapackage(name, indices_tech=None, data_tech=None, flip_tech=None, indices_bio=None, data_bio=None):

    dp = bwp.create_datapackage(
        name=f"validation.{name}.static",
        seed=42,
    )

    if indices_tech is not None:
        dp.add_persistent_vector(
            matrix="technosphere_matrix",
            data_array=data_tech,
            # Resource group name that will show up in provenance
            name=f"{name}-tech",
            indices_array=indices_tech,
            flip_array=flip_tech,
        )

    if indices_bio is not None:
        dp.add_persistent_vector(
            matrix="biosphere_matrix",
            data_array=data_bio,
            # Resource group name that will show up in provenance (?)
            name=f"{name}-bio",
            indices_array=indices_bio,
        )

    return dp


def pop_indices_from_dict(indices, dict_):
    count = 0
    for key in indices:
        try:
            dict_.pop(tuple(key))
            count += 1
        except KeyError:
            pass
    # print(f"Removed {count:4d} elements from dictionary")


def get_mask_wrt_dp(indices_all, indices_dp, mask_screening):
    """Get screening mask wrt indices of a datapackage."""
    mask_dp_and_screening_wrt_all = get_mask(indices_all, indices_dp) & mask_screening
    indices_dp_and_screening = indices_all[mask_dp_and_screening_wrt_all]
    mask_screening_wrt_dp = get_mask(indices_dp, indices_dp_and_screening)
    return mask_screening_wrt_dp
