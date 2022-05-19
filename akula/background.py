import numpy as np
from pathlib import Path
from fs.zipfs import ZipFS
import bw_processing as bwp
from bw_processing.merging import merge_datapackages_with_mask
import bw2data as bd
import bw2calc as bc
from matrix_utils.resource_group import FakeRNG
import stats_arrays as sa
from copy import deepcopy

from sensitivity_analysis import get_mask


DATA_DIR = Path(__file__).parent.resolve() / "data"
SAMPLES = 25000
LABELS_DICT = {
    "technosphere": "ecoinvent_3.8_cutoff_technosphere_matrix",
    "biosphere": 'ecoinvent_3.8_cutoff_biosphere_matrix',
    "characterization": 'IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data',
}


def create_lca(demand, use_distributions=True, seed=42):

    bd.projects.set_current("GSA for archetypes")

    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    fu_mapped, pkgs, _ = bd.prepare_lca_inputs(demand=demand, method=method, remapping=False)

    lca = bc.LCA(demand=fu_mapped, data_objs=pkgs, use_distributions=use_distributions, seed_override=seed)
    lca.lci()
    lca.lcia()

    return lca


def create_background_datapackage(demand, matrix_type, name, indices, num_samples=SAMPLES, seed=42):

    lca = create_lca(demand, seed=seed)

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / "xgboost" / f"{name}.zip"), write=True),
        name=name,
        seed=seed,
        sequential=True,
    )

    if matrix_type == "technosphere":
        num_resources = 4
    else:
        num_resources = 3

    obj = getattr(lca, f"{matrix_type}_mm")
    indices_array = np.hstack(
        [
            group.package.data[0] for group in obj.groups
            if (not isinstance(group.rng, FakeRNG)) and (not group.empty) and (len(group.package.data) == num_resources)
        ]
    )

    if len(indices_array) != len(indices):
        mask = get_mask(indices_array, indices)
    else:
        if np.all(indices_array == indices):
            mask = np.ones(len(indices_array), dtype=bool)
        else:
            mask = get_mask(indices_array, indices)

    data = []
    np.random.seed(seed)
    for _ in range(num_samples):
        next(obj)
        idata = []
        for group in obj.groups:
            if (not isinstance(group.rng, FakeRNG)) and (not group.empty):
                idata.append(group.rng.random_data)
        data.append(np.hstack(idata)[mask])
    data_array = np.vstack(data).T

    if matrix_type == "technosphere":
        flip_array = np.hstack(
            [
                group.flip for group in obj.groups
                if (not isinstance(group.rng, FakeRNG)) and (not group.empty) and (len(group.package.data) == num_resources)
            ]
        )
        dp.add_persistent_array(
            matrix=f"{matrix_type}_matrix",
            data_array=data_array,
            # Resource group name that will show up in provenance
            name=name,
            indices_array=indices_array[mask],
            flip_array=flip_array[mask],
        )
    else:
        dp.add_persistent_array(
            matrix=f"{matrix_type}_matrix",
            data_array=data_array,
            # Resource group name that will show up in provenance
            name=name,
            indices_array=indices_array[mask],
        )
    [
        d.update({"global_index": 1}) for d in dp.metadata['resources']
        if d['matrix'] == "characterization_matrix"
    ]  # TODO Chris, is this correct?
    return dp


def generate_validation_datapackages(demand, matrix_type, indices, mask, num_samples=SAMPLES, seed=42):

    name_all = f"validation.{matrix_type}.all"
    dp_validation_all = create_background_datapackage(demand, matrix_type, name_all, indices, num_samples, seed)

    bd.projects.set_current("GSA for archetypes")
    if matrix_type == "technosphere":
        dp = bd.Database('ecoinvent 3.8 cutoff').datapackage()
        group_label = 'ecoinvent_3.8_cutoff_technosphere_matrix'
    elif matrix_type == "biosphere":
        dp = bd.Database("ecoinvent 3.8 cutoff").datapackage()
        group_label = "ecoinvent_3.8_cutoff_biosphere_matrix"
    elif matrix_type == "characterization":
        dp = bd.Method(("IPCC 2013", "climate change", "GWP 100a", "uncertain")).datapackage()
        group_label = "IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data"
    else:
        return

    indices_static = dp.get_resource(f"{group_label}.indices")[0]
    dp_static = dp.filter_by_attribute('matrix', f'{matrix_type}_matrix')

    indices_array = dp_validation_all.data[0]
    assert np.all(indices_array == indices_static)

    # dp_validation_inf = merge_datapackages_with_mask(
    #     first_dp=dp_validation_all,
    #     first_resource_group_label=name_all,
    #     second_dp=dp_static,
    #     second_resource_group_label=group_label,
    #     mask_array=mask,
    # )

    name_inf = f"validation.{matrix_type}.inf"
    dp_validation_inf = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name_inf}.zip"), write=True),
        name=name_inf,
        seed=seed,
        sequential=True,
    )

    data_array = deepcopy(dp_validation_all.data[1])
    data_array[~mask] = np.tile(dp_static.data[1][~mask], (num_samples,  1)).T

    if matrix_type == "technosphere":
        flip_array = dp_validation_all.data[2]
        dp_validation_inf.add_persistent_array(
            matrix=f"{matrix_type}_matrix",
            data_array=data_array,
            # Resource group name that will show up in provenance
            name=name_inf,
            indices_array=indices_array,
            flip_array=flip_array,
        )
    else:
        dp_validation_inf.add_persistent_array(
            matrix=f"{matrix_type}_matrix",
            data_array=data_array,
            # Resource group name that will show up in provenance
            name=name_inf,
            indices_array=indices_array,
        )

    [
        d.update({"global_index": 1}) for d in dp_validation_inf.metadata['resources']
        if d['matrix'] == "characterization_matrix"
    ]  # TODO Chris, is this correct?

    return dp_validation_all, dp_validation_inf


def get_amounts_shift(lca, shift_median=True):

    dict_ = {}

    for matrix_type in ["technosphere", "biosphere", "characterization"]:

        obj = getattr(lca, f"{matrix_type}_mm")
        for group in obj.groups:
            if group.label == LABELS_DICT[matrix_type]:
                break

        params = group.package.data[2]

        # 1. Lognormal
        lognormal_where = np.where(
            params["uncertainty_type"] == sa.LognormalUncertainty.id
        )[0]
        lognormal = params[lognormal_where]
        m = lognormal["loc"]
        s = lognormal["scale"]
        lognormal_mean = np.exp(m + s ** 2 / 2)
        lognormal_median = np.exp(m)

        # # 2. Normal
        # normal_where = np.where(
        #     params["uncertainty_type"] == sa.NormalUncertainty.id
        # )[0]
        # normal = params[normal_where]
        # m = normal["loc"]
        # normal_mean = m
        # normal_median = normal_mean
        #
        # # 2. Uniform
        # uniform_where = np.where(
        #     params["uncertainty_type"] == sa.UniformUncertainty.id
        # )[0]
        # uniform = params[uniform_where]
        # a = uniform["minimum"]
        # b = uniform["maximum"]
        # uniform_mean = (a + b) / 2
        # uniform_median = uniform_mean

        # 4. Triangular
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
            # amounts[normal_where] = normal_median
            # amounts[uniform_where] = uniform_median
            amounts[triangular_where] = triangular_median
        else:
            amounts[lognormal_where] = lognormal_mean
            # amounts[normal_where] = normal_mean
            # amounts[uniform_where] = uniform_mean
            amounts[triangular_where] = triangular_mean

        dict_[matrix_type] = amounts

    return dict_


def get_lca_score_shift(demand, masks_dict, shift_median=True):

    lca = create_lca(demand, use_distributions=False)

    static_score = deepcopy(lca.score)

    modified_amounts = get_amounts_shift(lca, shift_median)

    dp = bwp.create_datapackage()

    for matrix_type, mask in masks_dict.items():

        obj = getattr(lca, f"{matrix_type}_mm")

        for group in obj.groups:
            if group.label == LABELS_DICT[matrix_type]:
                break

        new_amounts = deepcopy(group.package.data[1])
        new_amounts[mask] = modified_amounts[matrix_type][mask]

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
