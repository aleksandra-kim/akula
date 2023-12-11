import bw2data as bd
import bw2calc as bc
import pickle

COLOR_GRAY_HEX = "#b2bcc0"
COLOR_DARKGRAY_HEX = "#485063"
COLOR_DARKGRAY_HEX_OPAQUE = "rgba(72, 80, 99, 0.5)"
COLOR_BLACK_HEX = "#212931"
COLOR_PSI_BLUE = "#003b6e"
COLOR_PSI_LPURPLE = "#914967"
COLOR_PSI_LPURPLE_OPAQUE = "rgba(145, 73, 103, 0.5)"
COLOR_BRIGHT_PINK_RGB = "#e75480"
COLOR_PSI_DGREEN = "#197418"


def get_consumption_activity():
    co = bd.Database('swiss consumption 1.0')
    activity = [act for act in co if f"ch hh average consumption aggregated" in act['name']]
    assert len(activity) == 1
    return activity[0]


def compute_deterministic_score(project):
    bd.projects.set_current(project)
    act = get_consumption_activity()
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    lca = bc.LCA({act: 1}, method)
    lca.lci()
    lca.lcia()
    return lca.score


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
