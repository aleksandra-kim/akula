import numpy as np
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from tqdm import tqdm
from fs.zipfs import ZipFS
from copy import deepcopy
from pathlib import Path
from ..utils import read_pickle, write_pickle

DATA_DIR = Path(__file__).parents[1].resolve() / "data"


def get_mask_parameters(all_indices, use_indices):
    """Creates a `mask` such that `all_indices[mask]=use_indices`."""
    all_indices = np.array(all_indices, dtype=bwp.INDICES_DTYPE)
    use_indices = np.array(use_indices, dtype=bwp.INDICES_DTYPE)
    mask = np.zeros(len(all_indices), dtype=bool)
    for inds in tqdm(use_indices):
        mask_current = all_indices == inds
        mask = mask | mask_current
    return mask


# def add_variances(list_, lcia_score):
#     for dictionary in list_:
#         values = np.vstack(list(dictionary.values()))
#         values = np.hstack([values, np.ones((len(values), 1))*lcia_score])
#         variances = np.var(values, axis=1)
#         for i, k in enumerate(dictionary.keys()):
#             dictionary[k] = {
#                 "arr": values[i, :],
#                 "var": variances[i],
#             }
#
#
# def get_variance_threshold(list_, num_parameters):
#     # Collect all variances
#     vars = np.array([value['var'] for dictionary in list_ for key, value in dictionary.items()])
#     vars = np.sort(vars)[-1::-1]
#     vars_threshold = vars[:num_parameters][-1]
#     return vars_threshold
#
#
# # Remove lowly influential
# def get_indices_high_variance(dictionary, variances_threshold):
#     return [key for key in dictionary if dictionary[key]['var'] >= variances_threshold]



class LocalSAInterface:
    def __init__(self, indices, data, distributions, mask, factor=10):
        self.indices = indices
        self.data = data
        self.distributions = distributions
        self.has_uncertainty = self.get_uncertainty_bool(self.distributions)
        self.factor = factor
        self.mask = mask  # boolean mask to select indices

        assert self.indices.shape[0] == self.data.shape[0] == self.distributions.shape[0]

        self.masked_indices = self.indices[self.mask]
        self.masked_data = self.data[self.mask]
        self.masked_has_uncertainty = self.has_uncertainty[self.mask]

        self.size = len(self.masked_indices)
        self.index = None  # To indicate we haven't consumed first value yet
        self.mask_where = np.where(self.mask)[0]

    def __next__(self):
        if self.index is None:
            self.index = 0
        else:
            self.index += 1

        if self.index < self.size:
            # 0 and 1 are `no` and `unknown` uncertainty
            while not self.masked_has_uncertainty[self.index]:
                self.index += 1
                if self.index >= self.size:
                    raise StopIteration
        else:
            raise StopIteration

        data = self.data.copy()
        # print(data[self.mask_where[self.index]])
        data[self.mask_where[self.index]] *= self.factor
        # print(data[self.mask_where[self.index]])
        return data

    @staticmethod
    def get_uncertainty_bool(distributions):
        try:
            arr = distributions['uncertainty_type'] >= 2
        except (KeyError, IndexError):
            arr = distributions > 0
        return arr

    @property
    def coordinates(self):
        return self.masked_indices[self.index]


def run_local_sa(
        matrix_type,
        fu_mapped,
        packages,
        indices_array,
        data_array,
        distributions_array,
        mask,
        flip_array=None,  # only needed for technosphere
        const_factor=10,
):

    interface = LocalSAInterface(
        indices_array,
        data_array,
        distributions_array,
        mask,
        const_factor,
    )

    dp = bwp.create_datapackage()
    dp.add_dynamic_vector(
        matrix=f"{matrix_type}_matrix",
        interface=interface,
        indices_array=indices_array,
        flip_array=flip_array,
    )
    if matrix_type == "characterization":
        [d.update({"global_index": 1}) for d in dp.metadata['resources']]  # TODO Chris, is this correct?

    lca_local_sa = bc.LCA(demand=fu_mapped, data_objs=packages + [dp])
    lca_local_sa.lci()
    lca_local_sa.lcia()

    interface.index = None  # there should be a better way to discount the first __next__
    indices_local_sa_scores = {}
    # current_chunk = {}

    count = 0
    # ichunk = 0
    try:
        while True:
            next(lca_local_sa)
            count += 1
            if count % 500 == 0:
                print(count)
                # write_pickle(current_chunk, f"{ichunk}.pickle")
                # ichunk += 1
            # print(lca_local_sa.score)
            indices_local_sa_scores[tuple(interface.coordinates)] = np.array([lca_local_sa.score])
            # current_chunk[tuple(interface.coordinates)] = np.array([lca_local_sa.score])
    except StopIteration:
        pass

    assert count <= sum(interface.mask)

    return indices_local_sa_scores


def run_local_sa_technosphere(
        func_unit_mapped,
        packages,
        tech_has_uncertainty,
        mask_tech_without_noninf,
        factors,
        directory,
        tag,
):
    ecoinvent = bd.Database('ecoinvent 3.8 cutoff').datapackage()
    tech_ecoinvent = ecoinvent.filter_by_attribute('matrix', 'technosphere_matrix')
    tech_indices = tech_ecoinvent.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.indices')[0]
    tech_data = tech_ecoinvent.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.data')[0]
    tech_flip = tech_ecoinvent.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.flip')[0]
    for i, factor in enumerate(factors):
        fp_factor = directory / f"local_sa.{tag}.factor_{factor:.0e}.pickle"
        if fp_factor.exists():
            local_sa_current = read_pickle(fp_factor)
        else:
            local_sa_current = run_local_sa(
                "technosphere",
                func_unit_mapped,
                packages,
                tech_indices,
                tech_data,
                tech_has_uncertainty,
                mask_tech_without_noninf,
                tech_flip,
                factor,
            )
            write_pickle(local_sa_current, fp_factor)
        if i == 0:
            local_sa_results = deepcopy(local_sa_current)
        else:
            local_sa_results = {
                k: np.hstack([local_sa_results[k], local_sa_current[k]]) for k in local_sa_results.keys()
            }
    return local_sa_results


def run_local_sa_from_samples(
        name,
        func_unit_mapped,
        packages,
        factor,
        indices=None,
):
    fp = DATA_DIR / f"local-sa-{factor:.0e}-{name}.zip"
    dp = bwp.load_datapackage(ZipFS(fp))

    lca_local_sa = bc.LCA(
        demand=func_unit_mapped,
        data_objs=packages + [dp],
        use_arrays=True,
        # seed_override=seed,  # do NOT specify seed with sequential, because seed will cause random selection of data
        use_distributions=False,
    )
    lca_local_sa.lci()
    lca_local_sa.lcia()

    if indices is None:
        indices = dp.get_resource(f"local-sa-{name}-tech.indices")[0]

    # indices_local_sa_scores = {tuple(indices[0]): np.array([lca_local_sa.score])}
    indices_local_sa_scores = {}

    count = 0
    try:
        while True:
            indices_local_sa_scores[tuple(indices[count])] = np.array([lca_local_sa.score])

            next(lca_local_sa)

            count += 1
            if count % 200 == 0:
                print(count)

    except (StopIteration, IndexError) as e:
        pass

    assert count == len(indices)
    return indices_local_sa_scores


def run_local_sa_from_samples_technosphere(
        name,
        func_unit_mapped,
        packages,
        factors,
        indices,
        directory,
):
    print(f"Running local SA for {name} technosphere")
    tag = name.replace("-", "_")
    for i, factor in enumerate(factors):
        fp_factor = directory / f"local_sa.{tag}.factor_{factor:.0e}.pickle"
        if fp_factor.exists():
            local_sa_current = read_pickle(fp_factor)
        else:
            local_sa_current = run_local_sa_from_samples(
                name,
                func_unit_mapped,
                packages,
                factor,
                indices,
            )
            write_pickle(local_sa_current, fp_factor)
        if i == 0:
            local_sa_results = deepcopy(local_sa_current)
        else:
            local_sa_results = {
                k: np.hstack([local_sa_results[k], local_sa_current[k]]) for k in local_sa_results.keys()
            }
    return local_sa_results
