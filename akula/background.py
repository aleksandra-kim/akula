import numpy as np
from pathlib import Path
from fs.zipfs import ZipFS
import bw_processing as bwp
from bw_processing.merging import merge_datapackages_with_mask
import bw2data as bd
import bw2calc as bc
from matrix_utils.resource_group import FakeRNG

# Local files
from utils import get_activities_from_indices, create_static_datapackage

DATA_DIR = Path(__file__).parent.resolve() / "data"
SAMPLES = 25000


def create_lca():

    bd.projects.set_current("GSA for archetypes")

    co = bd.Database('swiss consumption 1.0')
    fu = [act for act in co if "ch hh average consumption aggregated, years 151617" == act['name']][0]

    demand = {fu: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    fu_mapped, pkgs, _ = bd.prepare_lca_inputs(demand=demand, method=method, remapping=False)

    lca = bc.LCA(demand=fu_mapped, data_objs=pkgs, use_distributions=True,)
    lca.lci()
    lca.lcia()

    return lca


def create_background_datapackage(type, name, indices, num_samples=SAMPLES, seed=42):

    lca = create_lca()

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}.zip"), write=True),
        name=name,
        seed=seed,
    )

    if type == "technosphere":
        num_resources = 4
    else:
        num_resources = 3

    obj = getattr(lca, f"{type}_mm")
    indices_array = np.hstack(
        [
            group.package.data[0] for group in obj.groups
            if (not isinstance(group.rng, FakeRNG)) and (not group.empty) and (len(group.package.data) == num_resources)
        ]
    )
    # mask = np.hstack(
    #     [
    #         group.rng.params["uncertainty_type"] > 1 for group in lca.technosphere_mm.groups
    #         if (not isinstance(group.rng, FakeRNG)) and (not group.empty) and (len(group.package.data) == 4)
    #     ]
    # )

    data = []
    np.random.seed(seed)
    for _ in range(num_samples):
        next(obj)
        idata = []
        for group in obj.groups:
            if (not isinstance(group.rng, FakeRNG)) and (not group.empty):
                idata.append(group.rng.random_data)
        data.append(np.hstack(idata))
    data_array = np.vstack(data).T

    assert np.all(indices_array == indices)

    if type == "technosphere":
        flip_array = np.hstack(
            [
                group.flip for group in obj.groups
                if (not isinstance(group.rng, FakeRNG)) and (not group.empty)
            ]
        )
        dp.add_persistent_array(
            matrix=f"{type}_matrix",
            data_array=data_array,
            # Resource group name that will show up in provenance
            name=name,
            indices_array=indices_array,
            flip_array=flip_array,
        )
    else:
        dp.add_persistent_array(
            matrix=f"{type}_matrix",
            data_array=data_array,
            # Resource group name that will show up in provenance
            name=name,
            indices_array=indices_array,
        )
    return dp


def generate_validation_datapackages(type, indices, mask, num_samples=SAMPLES, seed=42):

    name_all = f"validation.{type}.all"
    dp_validation_all = create_background_datapackage(type, name_all, indices, num_samples, seed)

    #
    # activities = get_activities_from_indices(indices)
    # exc_data = {
    #     (exc.input.id, exc.output.id): (exc.amount, exc['type'] != "production")
    #     for lst in activities.values() for exc in lst
    # }
    #
    # data_static = np.array([exc_data[(int(inds['row']), int(inds['col']))][0] for inds in indices])
    # flip_static = np.array([exc_data[(int(inds['row']), int(inds['col']))][1] for inds in indices], dtype=bool)
    #
    # dp_static = create_static_datapackage(
    #     "technosphere", indices, data_tech=data_static, flip_tech=flip_static,
    # )

    bd.projects.set_current("GSA for archetypes")
    if type == "technosphere":
        dp = bd.Database('ecoinvent 3.8 cutoff').datapackage()
        group_label = 'ecoinvent_3.8_cutoff_technosphere_matrix'

    elif type == "biosphere":
        dp = bd.Database("ecoinvent 3.8 cutoff").datapackage()
        group_label = "ecoinvent_3.8_cutoff_biosphere_matrix"
    elif type == "characterization":
        dp = bd.Method(("IPCC 2013", "climate change", "GWP 100a", "uncertain")).datapackage()
        group_label = "IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data"
    else:
        return

    indices_static = dp.get_resource(f"{group_label}.indices")[0]
    dp_static = dp.filter_by_attribute('matrix', f'{type}_matrix')

    indices_validation_all = dp_validation_all.data[0]
    assert np.all(indices_validation_all == indices_static)

    dp_validation_inf = merge_datapackages_with_mask(
        first_dp=dp_validation_all,
        first_resource_group_label=name_all,
        second_dp=dp_static,
        second_resource_group_label=group_label,
        mask_array=mask,
    )

    [
        d.update({"global_index": 1}) for d in dp_validation_all.metadata['resources']
        if d['matrix'] == "characterization_matrix"
    ]  # TODO Chris, is this correct?
    [
        d.update({"global_index": 1}) for d in dp_validation_inf.metadata['resources']
        if d['matrix'] == "characterization_matrix"
    ]  # TODO Chris, is this correct?

    return dp_validation_all, dp_validation_inf
