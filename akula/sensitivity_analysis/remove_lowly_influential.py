import numpy as np
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from tqdm import tqdm
from fs.zipfs import ZipFS
from copy import deepcopy
from pathlib import Path
from ..utils import read_pickle, write_pickle, get_fu_pkgs

GSA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "sensitivity-analysis"


class LocalSensitivityAnalysisSampler:
    """
    A class that generates input samples for local sensitivity analysis by varying all uncertain inputs by a `factor`.
    """
    def __init__(self, indices, data, distributions, mask, factor=10):
        self.indices = indices
        self.data = data
        self.distributions = distributions
        self.has_uncertainty = self.get_uncertainty_bool(self.distributions)
        self.const_factor = factor
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
        data[self.mask_where[self.index]] *= self.const_factor

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
        indices,
        data,
        distributions,
        mask,
        flip=None,  # only needed for technosphere
        factor=10,
):
    """Run MC simulations by varying 1 input at a time, and return LCIA scores for each index in indices_array[mask]."""

    sampler = LocalSensitivityAnalysisSampler(
        indices,
        data,
        distributions,
        mask,
        factor,
    )

    dp = bwp.create_datapackage()
    dp.add_dynamic_vector(
        matrix=f"{matrix_type}_matrix",
        interface=sampler,
        indices_array=indices,
        flip_array=flip,
    )
    if matrix_type == "characterization":
        [d.update({"global_index": 1}) for d in dp.metadata['resources']]

    lca = bc.LCA(demand=fu_mapped, data_objs=packages + [dp])
    lca.lci()
    lca.lcia()

    sampler.index = None  # there should be a better way to discount the first __next__
    scores = {}

    count = 0
    try:
        while True:
            next(lca)
            count += 1
            if count % 500 == 0:
                print(count)
            scores[tuple(sampler.coordinates)] = np.array([lca.score])
    except StopIteration:
        pass

    assert count <= sum(sampler.mask)

    return scores


def get_scores_local_sa_technosphere(mask, project, factor, tag):
    """Wrapper function to run MC simulations by varying 1 input at a time (local SA) for TECHNOSPHERE."""

    fp = GSA_DIR / f"scores.tech.lsa.factor_{factor}.{tag}.pickle"

    if fp.exists():
        scores = read_pickle(fp)
    else:
        fu, pkgs = get_fu_pkgs(project)
        ei = bd.Database('ecoinvent 3.8 cutoff').datapackage()
        tei = ei.filter_by_attribute('matrix', 'technosphere_matrix')
        tindices = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.indices')[0]
        tdata = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.data')[0]
        tflip = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.flip')[0]
        tdistributions = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.distributions')[0]

        factors = [1 / factor, factor]

        for i, factor in enumerate(factors):

            fp_i = GSA_DIR / f"local_sa.{tag}.factor_{factor:.0e}.pickle"

            if fp_i.exists():
                scores_i = read_pickle(fp_i)
            else:
                scores_i = run_local_sa(
                    "technosphere", fu, pkgs, tindices, tdata, tdistributions, mask, tflip, factor
                )
                write_pickle(scores_i, fp_i)

            if i == 0:
                scores = deepcopy(scores_i)
            else:
                scores = {k: np.hstack([scores[k], scores_i[k]]) for k in scores.keys()}

        write_pickle(scores, fp)

    return scores


def get_scores_local_sa_biosphere(mask, project, factor):
    """Wrapper function to run MC simulations by varying 1 input at a time (local SA) for BIOSPHERE."""

    fp = GSA_DIR / f"scores.bio.lsa.factor_{factor}.pickle"

    if fp.exists():
        scores = read_pickle(fp)
    else:
        fu, pkgs = get_fu_pkgs(project)
        ei = bd.Database('ecoinvent 3.8 cutoff').datapackage()
        bei = ei.filter_by_attribute('matrix', 'biosphere_matrix')
        bindices = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.indices')[0]
        bdata = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.data')[0]
        bdistributions = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.distributions')[0]

        scores = run_local_sa(
            "biosphere", fu, pkgs, bindices, bdata, bdistributions, mask, factor=factor
        )

        write_pickle(scores, fp)
    return scores


def get_scores_local_sa_characterization(mask, project, factor):
    """Wrapper function to run MC simulations by varying 1 input at a time (local SA) for CHARACTERIZATION FACTORS."""

    fp = GSA_DIR / f"scores.cf.lsa.factor_{factor}.pickle"

    if fp.exists():
        scores = read_pickle(fp)
    else:
        fu, pkgs = get_fu_pkgs(project)
        cf = bd.Method(("IPCC 2013", "climate change", "GWP 100a", "uncertain")).datapackage()
        cindices = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.indices')[0]
        cdata = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.data')[0]
        cdistributions = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.distributions')[0]

        scores = run_local_sa(
            "characterization", fu, pkgs, cindices, cdata, cdistributions, mask, factor=factor
        )

        write_pickle(scores, fp)

    return scores


def get_all_lowinf_scores(project, factor, cutoff, max_calc):
    tag = f"cutoff_{cutoff:.0e}.maxcalc_{max_calc:.0e}"

    tmask_wo_noninf = read_pickle(GSA_DIR / f"tech.mask.without_noninf.sct.{tag}.pickle")
    bmask_wo_noninf = read_pickle(GSA_DIR / "bio.mask.without_noninf.pickle")
    cmask_wo_noninf = read_pickle(GSA_DIR / "cf.mask.without_noninf.pickle")

    tscores = get_scores_local_sa_technosphere(tmask_wo_noninf, project, factor, tag)
    bscores = get_scores_local_sa_biosphere(bmask_wo_noninf, project, factor)
    cscores = get_scores_local_sa_characterization(cmask_wo_noninf, project, factor)

    return tscores, bscores, cscores


def get_tmask_wo_lowinf(project, factor, cutoff, max_calc):

    tag = f"cutoff_{cutoff:.0e}.maxcalc_{max_calc:.0e}"

    fp = GSA_DIR / f"tech.mask.without_lowinf.lsa.factor_{factor}.{tag}.pickle"


    tmask_wo_noninf = read_pickle(GSA_DIR / f"tech.mask.without_noninf.sct.{tag}.pickle")

    if fp.exists():
        mask_wo_lowinf = read_pickle(fp)
    else:
        write_pickle(mask_wo_lowinf, fp)

    return scores


def get_bmask_wo_lowinf(project):
    return


def get_cmask_wo_lowinf(project):
    return


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











# def run_local_sa_from_samples(
#         name,
#         func_unit_mapped,
#         packages,
#         factor,
#         indices=None,
# ):
#     fp = DATA_DIR / f"local-sa-{factor:.0e}-{name}.zip"
#     dp = bwp.load_datapackage(ZipFS(fp))
#
#     lca_local_sa = bc.LCA(
#         demand=func_unit_mapped,
#         data_objs=packages + [dp],
#         use_arrays=True,
#         # seed_override=seed,  # do NOT specify seed with sequential, because seed will cause random selection of data
#         use_distributions=False,
#     )
#     lca_local_sa.lci()
#     lca_local_sa.lcia()
#
#     if indices is None:
#         indices = dp.get_resource(f"local-sa-{name}-tech.indices")[0]
#
#     # indices_local_sa_scores = {tuple(indices[0]): np.array([lca_local_sa.score])}
#     indices_local_sa_scores = {}
#
#     count = 0
#     try:
#         while True:
#             indices_local_sa_scores[tuple(indices[count])] = np.array([lca_local_sa.score])
#
#             next(lca_local_sa)
#
#             count += 1
#             if count % 200 == 0:
#                 print(count)
#
#     except (StopIteration, IndexError) as e:
#         pass
#
#     assert count == len(indices)
#     return indices_local_sa_scores
#
#
# def run_local_sa_from_samples_technosphere(
#         name,
#         func_unit_mapped,
#         packages,
#         factors,
#         indices,
#         directory,
# ):
#     print(f"Running local SA for {name} technosphere")
#     tag = name.replace("-", "_")
#     for i, factor in enumerate(factors):
#         fp_factor = directory / f"local_sa.{tag}.factor_{factor:.0e}.pickle"
#         if fp_factor.exists():
#             local_sa_current = read_pickle(fp_factor)
#         else:
#             local_sa_current = run_local_sa_from_samples(
#                 name,
#                 func_unit_mapped,
#                 packages,
#                 factor,
#                 indices,
#             )
#             write_pickle(local_sa_current, fp_factor)
#         if i == 0:
#             local_sa_results = deepcopy(local_sa_current)
#         else:
#             local_sa_results = {
#                 k: np.hstack([local_sa_results[k], local_sa_current[k]]) for k in local_sa_results.keys()
#             }
#     return local_sa_results
