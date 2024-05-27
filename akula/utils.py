import numpy as np
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
import pickle
import stats_arrays as sa
from copy import deepcopy

COLOR_GRAY_HEX = "#b2bcc0"
COLOR_DARKGRAY_HEX = "#485063"
COLOR_DARKGRAY_HEX_OPAQUE = "rgba(72, 80, 99, 0.5)"
COLOR_BLACK_HEX = "#212931"
COLOR_PSI_BLUE = "#003b6e"
COLOR_PSI_LPURPLE = "#914967"
COLOR_PSI_LPURPLE_OPAQUE = "rgba(145, 73, 103, 0.5)"
COLOR_BRIGHT_PINK_RGB = "#e75480"
COLOR_PSI_DGREEN = "#197418"

LABELS_DICT = {
    "technosphere": "ecoinvent_3.8_cutoff_technosphere_matrix",
    "biosphere": 'ecoinvent_3.8_cutoff_biosphere_matrix',
    "characterization": 'IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data',
}


def get_consumption_activity():
    co = bd.Database('swiss consumption 1.0')
    activity = [act for act in co if f"ch hh average consumption aggregated" in act['name']]
    assert len(activity) == 1
    return activity[0]


def get_lca(project):
    bd.projects.set_current(project)
    act = get_consumption_activity()
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    lca = bc.LCA({act: 1}, method)
    return lca


def get_fu_pkgs(project):
    bd.projects.set_current(project)
    act = get_consumption_activity()
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    fu_mapped, pkgs, _ = bd.prepare_lca_inputs(demand={act: 1}, method=method, remapping=False)
    return fu_mapped, pkgs


def compute_deterministic_score(project):
    bd.projects.set_current(project)
    lca = get_lca(project)
    lca.lci()
    lca.lcia()
    return lca.score


def get_amounts_shift(lca, shift_median=True):

    dict_ = {}

    for matrix_type in ["technosphere", "biosphere", "characterization"]:

        obj = getattr(lca, f"{matrix_type}_mm")
        for group in obj.groups:
            if group.label == LABELS_DICT[matrix_type]:
                break

        params = group.package.data[2]

        # Lognormal
        lognormal_where = np.where(
            params["uncertainty_type"] == sa.LognormalUncertainty.id
        )[0]
        lognormal = params[lognormal_where]
        m = lognormal["loc"]
        s = lognormal["scale"]
        lognormal_mean = np.exp(m + s ** 2 / 2)
        lognormal_median = np.exp(m)

        # Triangular
        triangular_where = np.where(
            params["uncertainty_type"] == sa.TriangularUncertainty.id
        )[0]
        triangular = params[triangular_where]
        c = triangular["loc"]
        a = triangular["minimum"]
        b = triangular["maximum"]
        triangular_mean = (a + b + c) / 3
        triangular_median = np.empty(triangular.shape[0])
        triangular_median[:] = np.nan
        case1 = np.where(c >= (a + b) / 2)[0]
        triangular_median[case1] = a[case1] + np.sqrt(
            (b[case1] - a[case1]) * (c[case1] - a[case1]) / 2
        )
        case2 = np.where(c < (a + b) / 2)[0]
        triangular_median[case2] = b[case2] - np.sqrt(
            (b[case2] - a[case2]) * (b[case2] - c[case2]) / 2
        )

        amounts = deepcopy(group.package.data[1])
        if shift_median:
            amounts[lognormal_where] = lognormal_median
            amounts[triangular_where] = triangular_median
        else:
            amounts[lognormal_where] = lognormal_mean
            amounts[triangular_where] = triangular_mean

        dict_[matrix_type] = amounts

    return dict_


def get_lca_score_shift(project, masks_dict, shift_median=True):

    lca = get_lca(project)
    lca.lci()
    lca.lcia()

    static_score = deepcopy(lca.score)

    modified_amounts = get_amounts_shift(lca, shift_median)

    dp = bwp.create_datapackage()

    for matrix_type, mask in masks_dict.items():

        obj = getattr(lca, f"{matrix_type}_mm")

        for group in obj.groups:
            if group.label == LABELS_DICT[matrix_type]:
                break

        new_amounts = deepcopy(group.package.data[1])
        if mask is not None:
            new_amounts[mask] = modified_amounts[matrix_type][mask]
        else:
            new_amounts = modified_amounts[matrix_type]

        if matrix_type == "technosphere":
            dp.add_persistent_vector(
                matrix=f"{matrix_type}_matrix",
                data_array=new_amounts,
                # Resource group name that will show up in provenance
                name=f"{matrix_type}.temp",
                indices_array=group.package.data[0],
                flip_array=group.package.data[3],
            )
        else:
            dp.add_persistent_vector(
                matrix=f"{matrix_type}_matrix",
                data_array=new_amounts,
                # Resource group name that will show up in provenance
                name=f"{matrix_type}.temp",
                indices_array=group.package.data[0],
            )

    [
        d.update({"global_index": 1}) for d in dp.metadata['resources']
        if d['matrix'] == "characterization_matrix"
    ]  # TODO Chris, is this correct?

    lca2 = bc.LCA(
        demand=lca.demand, data_objs=lca.packages + [dp], use_distributions=False,
    )
    lca2.lci()
    lca2.lcia()

    return lca2.score - static_score


def write_pickle(data, filepath):
    """Write ``data`` to a file with .pickle extension"""
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def read_pickle(filepath):
    """Read ``data`` from a file with .pickle extension"""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def update_fig_axes(fig):
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=COLOR_GRAY_HEX,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=COLOR_BLACK_HEX,
        showline=True,
        linewidth=1,
        linecolor=COLOR_GRAY_HEX,
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=COLOR_GRAY_HEX,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=COLOR_BLACK_HEX,
        showline=True,
        linewidth=1,
        linecolor=COLOR_GRAY_HEX,
    )
    fig.update_layout(
        legend=dict(
            bordercolor=COLOR_DARKGRAY_HEX,
            borderwidth=1,
        ),
        margin=dict(t=40, b=10, l=10, r=0),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )
    return fig


# def setup_bw_project(years="151617"):
#     project = "GSA for correlations"
#     bd.projects.set_current(project)
#
#     co = bd.Database('swiss consumption 1.0')
#     fu = [act for act in co if f"ch hh average consumption aggregated, years {years}" == act['name']][0]
#     demand = {fu: 1}
#     method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
#
#     lca = bc.LCA(demand=demand, method=method, use_distributions=False)
#     lca.lci()
#     lca.lcia()
#
#     return lca


# def create_static_datapackage(name, indices_tech=None, data_tech=None, flip_tech=None, indices_bio=None, data_bio=None):
#
#     dp = bwp.create_datapackage(
#         name=f"validation.{name}.static",
#         seed=42,
#     )
#
#     if indices_tech is not None:
#         dp.add_persistent_vector(
#             matrix="technosphere_matrix",
#             data_array=data_tech,
#             # Resource group name that will show up in provenance
#             name=f"{name}-tech",
#             indices_array=indices_tech,
#             flip_array=flip_tech,
#         )
#
#     if indices_bio is not None:
#         dp.add_persistent_vector(
#             matrix="biosphere_matrix",
#             data_array=data_bio,
#             # Resource group name that will show up in provenance (?)
#             name=f"{name}-bio",
#             indices_array=indices_bio,
#         )
#
#     return dp
#
#
# def pop_indices_from_dict(indices, dict_):
#     count = 0
#     for key in indices:
#         try:
#             dict_.pop(tuple(key))
#             count += 1
#         except KeyError:
#             pass
#     # print(f"Removed {count:4d} elements from dictionary")


# def get_mask_wrt_dp(indices_all, indices_dp, mask_screening):
#     """Get screening mask wrt indices of a datapackage."""
#     mask_dp_and_screening_wrt_all = get_mask(indices_all, indices_dp) & mask_screening
#     indices_dp_and_screening = indices_all[mask_dp_and_screening_wrt_all]
#     mask_screening_wrt_dp = get_mask(indices_dp, indices_dp_and_screening)
#     return mask_screening_wrt_dp
