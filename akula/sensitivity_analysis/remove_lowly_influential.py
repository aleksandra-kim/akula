import numpy as np
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from fs.zipfs import ZipFS
from copy import deepcopy
from pathlib import Path

from .utils import get_mask
from ..utils import read_pickle, write_pickle, get_fu_pkgs, get_lca

GSA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "sensitivity-analysis"
DATA_DIR = Path(__file__).parent.parent.parent.resolve() / "data"
DP_DIR = DATA_DIR / "datapackages"


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
        tag="",
):
    """Run MC simulations by varying 1 input at a time, and return LCIA scores for each index in indices_array[mask]."""

    scores = dict()

    if matrix_type == "technosphere":
        batch_size = 25000
    elif matrix_type == "biosphere":
        batch_size = 100000
    else:
        batch_size = len(indices)

    for start in range(0, len(indices), batch_size):

        end = min(start + batch_size, len(indices))
        fp_current = GSA_DIR / f"scores.{tag}.indices_{start:06d}_{end:06d}.pickle"

        if matrix_type == "technosphere":
            print(f"LSA scores for TECH -- indices {start:6d} - {end:6d}")
        elif matrix_type == "biosphere":
            print(f"LSA scores for  BIO -- indices {start:6d} - {end:6d}")
        else:
            print(f"LSA scores for   CF -- indices {start:6d} - {end:6d}")

        if fp_current.exists():
            scores_current = read_pickle(fp_current)

        else:
            if matrix_type == "technosphere":
                flip_current = flip[start:end]
            else:
                flip_current = flip

            sampler = LocalSensitivityAnalysisSampler(
                indices[start:end],
                data[start:end],
                distributions[start:end],
                mask[start:end],
                factor,
            )

            dp = bwp.create_datapackage()
            dp.add_dynamic_vector(
                matrix=f"{matrix_type}_matrix",
                interface=sampler,
                indices_array=indices[start:end],
                flip_array=flip_current,
            )
            if matrix_type == "characterization":
                [d.update({"global_index": 1}) for d in dp.metadata['resources']]

            lca = bc.LCA(demand=fu_mapped, data_objs=packages + [dp])
            lca.lci()
            lca.lcia()

            sampler.index = None  # there should be a better way to discount the first __next__
            scores_current = dict()
            count = 0
            try:
                while True:
                    next(lca)
                    count += 1
                    scores_current[tuple(sampler.coordinates)] = np.array([lca.score])
            except StopIteration:
                pass

            assert count <= sum(sampler.mask)

            write_pickle(scores_current, fp_current)

        scores.update(scores_current)

    return scores


def get_scores_local_sa_technosphere(mask, project, factor, tag):
    """Wrapper function to run MC simulations by varying 1 input at a time (local SA) for TECHNOSPHERE."""

    fp = GSA_DIR / f"scores.tech.lsa.{tag}.factor_{factor}.pickle"

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

        for i, current_factor in enumerate(factors):

            if current_factor == factor:
                current_tag = f"tech.lsa.{tag}.factor_{factor}_mult"
            else:
                current_tag = f"tech.lsa.{tag}.factor_{factor}_div"

            fp_i = GSA_DIR / f"scores.{current_tag}.pickle"

            if fp_i.exists():
                scores_i = read_pickle(fp_i)
            else:
                scores_i = run_local_sa(
                    "technosphere", fu, pkgs, tindices, tdata, tdistributions, mask, tflip,
                    current_factor, tag=current_tag,
                )
                write_pickle(scores_i, fp_i)

            if i == 0:
                scores = deepcopy(scores_i)
            else:
                scores = {k: np.hstack([scores[k], scores_i[k]]) for k in scores.keys()}

        write_pickle(scores, fp)

    return scores


def get_scores_local_sa_biosphere(mask, project, factor):
    """
    Wrapper function to run MC simulations by varying 1 input at a time (local SA) for BIOSPHERE.

    Since biosphere inputs are linear wrt to the LCIA score, local sensitivity analysis was performed by multiplying
    each default input value by only 1 factor.
    """

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

        tag = f"bio.lsa.factor_{factor}_mult"
        scores = run_local_sa(
            "biosphere", fu, pkgs, bindices, bdata, bdistributions, mask, factor=factor, tag=tag
        )

        write_pickle(scores, fp)
    return scores


def get_scores_local_sa_characterization(mask, project, factor):
    """
    Wrapper function to run MC simulations by varying 1 input at a time (local SA) for CHARACTERIZATION FACTORS.

    Since characterization inputs are linear wrt to the LCIA score, local sensitivity analysis was performed by
    multiplying each default input value by only 1 factor.
    """

    fp = GSA_DIR / f"scores.cf.lsa.factor_{factor}.pickle"

    if fp.exists():
        scores = read_pickle(fp)
    else:
        fu, pkgs = get_fu_pkgs(project)
        cf = bd.Method(("IPCC 2013", "climate change", "GWP 100a", "uncertain")).datapackage()
        cindices = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.indices')[0]
        cdata = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.data')[0]
        cdistributions = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.distributions')[0]

        tag = f"cf.lsa.factor_{factor}_mult"
        scores = run_local_sa(
            "characterization", fu, pkgs, cindices, cdata, cdistributions, mask, factor=factor, tag=tag
        )

        write_pickle(scores, fp)

    return scores


def get_scores_local_sa(project, factor, cutoff, max_calc):
    """Wrapper function to get all LSA scores for TECH, BIO, and CF."""

    tag = f"cutoff_{cutoff:.0e}.maxcalc_{max_calc:.0e}"

    tmask_wo_noninf = read_pickle(GSA_DIR / f"mask.tech.without_noninf.sct.{tag}.pickle")
    bmask_wo_noninf = read_pickle(GSA_DIR / "mask.bio.without_noninf.pickle")
    cmask_wo_noninf = read_pickle(GSA_DIR / "mask.cf.without_noninf.pickle")

    tscores = get_scores_local_sa_technosphere(tmask_wo_noninf, project, factor, tag)
    bscores = get_scores_local_sa_biosphere(bmask_wo_noninf, project, factor)
    cscores = get_scores_local_sa_characterization(cmask_wo_noninf, project, factor)

    lsa_scores = {"tech": tscores, "bio": bscores, "cf": cscores}

    return lsa_scores


def get_variances_of_lsa_scores(project, lsa_scores):
    """
    Compute variances of LSA scores for tech, bio, and cf dictionaries passed in the `lsa_scores`. Include deterministic
    LCIA score when computing the variances.
    """

    lca = get_lca(project)
    lca.lci()
    lca.lcia()

    variances = dict()

    for input_type, dictionary in lsa_scores.items():
        values = np.vstack(list(dictionary.values()))
        values = np.hstack([values, np.ones((len(values), 1))*lca.score])
        values_var = np.var(values, axis=1)

        variances[input_type] = dict()

        for i, k in enumerate(dictionary.keys()):
            variances[input_type][k] = {
                "lsa_scores": values[i, :],
                "variance": values_var[i],
            }

    return variances


def get_variance_threshold(variances, num_parameters):
    """Determine the threshold for the variance based on preselected number of lowly influential inputs."""
    # Collect all variances
    variances = np.array([
        value['variance'] for dictionary in variances.values() for key, value in dictionary.items()
    ])
    variances = np.sort(variances)[-1::-1]
    # Compute threshold variance
    variance_threshold = variances[:num_parameters][-1]
    return variance_threshold


def get_indices_high_variance(variances, variance_threshold):
    """Select lowly influential inputs with variances above the threshold."""
    selected = dict()
    for input_type, dictionary in variances.items():
        selected[input_type] = [key for key in dictionary if dictionary[key]['variance'] >= variance_threshold]
    return selected


def get_masks_wo_lowinf_lsa(project, factor, cutoff, max_calc, num_lowinf):
    """Wrapper function that collects all masks for TECH, BIO, and CF after removing lowly influential inputs."""

    tag = f"cutoff_{cutoff:.0e}.maxcalc_{max_calc:.0e}"

    fp_mask_tech = GSA_DIR / f"mask.tech.without_lowinf.{num_lowinf}.lsa.factor_{factor}.{tag}.pickle"
    fp_mask_bio = GSA_DIR / f"mask.bio.without_lowinf.{num_lowinf}.lsa.factor_{factor}.{tag}.pickle"
    fp_mask_cf = GSA_DIR / f"mask.cf.without_lowinf.{num_lowinf}.lsa.factor_{factor}.{tag}.pickle"
    fp_masks = [fp_mask_tech, fp_mask_bio, fp_mask_cf]

    masks_exist = [fp.exists() for fp in fp_masks]

    if all(masks_exist):
        tmask_wo_lowinf = read_pickle(fp_mask_tech)
        bmask_wo_lowinf = read_pickle(fp_mask_bio)
        cmask_wo_lowinf = read_pickle(fp_mask_cf)

    else:
        fp_inds_tech = GSA_DIR / f"indices.tech.without_lowinf.{num_lowinf}.lsa.factor_{factor}.{tag}.pickle"
        fp_inds_bio = GSA_DIR / f"indices.bio.without_lowinf.{num_lowinf}.lsa.factor_{factor}.{tag}.pickle"
        fp_inds_cf = GSA_DIR / f"indices.cf.without_lowinf.{num_lowinf}.lsa.factor_{factor}.{tag}.pickle"
        fp_inds = [fp_inds_tech, fp_inds_bio, fp_inds_cf]

        inds_exist = [fp.exists() for fp in fp_inds]

        if all(inds_exist):
            tindices_wo_lowinf = read_pickle(fp_inds_tech)
            bindices_wo_lowinf = read_pickle(fp_inds_bio)
            cindices_wo_lowinf = read_pickle(fp_inds_cf)

        else:
            lsa_scores = get_scores_local_sa(project, factor, cutoff, max_calc)
            variances = get_variances_of_lsa_scores(project, lsa_scores)
            variance_threshold = get_variance_threshold(variances, num_lowinf)
            selected = get_indices_high_variance(variances, variance_threshold)

            tindices_wo_lowinf = selected["tech"]
            bindices_wo_lowinf = selected["bio"]
            cindices_wo_lowinf = selected["cf"]

            write_pickle(tindices_wo_lowinf, fp_inds_tech)
            write_pickle(bindices_wo_lowinf, fp_inds_bio)
            write_pickle(cindices_wo_lowinf, fp_inds_cf)

        tmask_wo_lowinf = get_tmask_wo_lowinf(project, tindices_wo_lowinf)
        bmask_wo_lowinf = get_bmask_wo_lowinf(project, bindices_wo_lowinf)
        cmask_wo_lowinf = get_cmask_wo_lowinf(project, cindices_wo_lowinf)

        write_pickle(tmask_wo_lowinf, fp_mask_tech)
        write_pickle(bmask_wo_lowinf, fp_mask_bio)
        write_pickle(cmask_wo_lowinf, fp_mask_cf)

    return tmask_wo_lowinf, bmask_wo_lowinf, cmask_wo_lowinf


def get_tmask_wo_lowinf(project, tindices_wo_lowinf):
    bd.projects.set_current(project)
    ei = bd.Database("ecoinvent 3.8 cutoff").datapackage()
    tei = ei.filter_by_attribute('matrix', 'technosphere_matrix')
    tindices_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.indices')[0]
    mask = get_mask(tindices_ei, tindices_wo_lowinf)
    return mask


def get_bmask_wo_lowinf(project, bindices_wo_lowinf):
    bd.projects.set_current(project)
    ei = bd.Database("ecoinvent 3.8 cutoff").datapackage()
    bei = ei.filter_by_attribute('matrix', 'biosphere_matrix')
    bindices = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.indices')[0]
    mask = get_mask(bindices, bindices_wo_lowinf)
    return mask


def get_cmask_wo_lowinf(project, cindices_wo_lowinf):
    bd.projects.set_current(project)
    cf = bd.Method(("IPCC 2013", "climate change", "GWP 100a", "uncertain")).datapackage()
    cindices = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.indices')[0]
    mask = get_mask(cindices, cindices_wo_lowinf)
    return mask


def get_pmask_wo_lowinf(iterations, seed, pindices_wo_lowinf):
    fp = DP_DIR / f"parameterization-parameters-{seed}-{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(fp))
    pindices = dp.get_resource('ecoinvent-parameters.indices')[0]
    mask = get_mask(pindices, pindices_wo_lowinf)
    return mask


# def get_mask_parameters(all_indices, use_indices):
#     """Creates a `mask` such that `all_indices[mask]=use_indices`."""
#     all_indices = np.array(all_indices, dtype=bwp.INDICES_DTYPE)
#     use_indices = np.array(use_indices, dtype=bwp.INDICES_DTYPE)
#     mask = np.zeros(len(all_indices), dtype=bool)
#     for inds in tqdm(use_indices):
#         mask_current = all_indices == inds
#         mask = mask | mask_current
#     return mask


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
