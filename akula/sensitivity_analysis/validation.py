import numpy as np
from pathlib import Path
import bw2data as bd
import bw_processing as bwp
from fs.zipfs import ZipFS

from ..utils import read_pickle, write_pickle, get_lca_score_shift
from ..monte_carlo import compute_consumption_lcia
from ..parameterization import generate_parameterization_datapackage
from ..combustion import generate_combustion_datapackage
from ..electricity import generate_entsoe_datapackage
from ..markets import generate_markets_datapackage

DATA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "datapackages"
GSA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "sensitivity-analysis"
GSA_DIR_CORR = GSA_DIR / "correlated"
GSA_DIR_INDP = GSA_DIR / "independent"
GSA_DIR_CORR.mkdir(exist_ok=True, parents=True)
GSA_DIR_INDP.mkdir(exist_ok=True, parents=True)


def create_all_datapackages(fp_ecoinvent, project, iterations, seed=42, directory=None):
    bd.projects.set_current(project)
    _, dp_parameterization = generate_parameterization_datapackage(
        fp_ecoinvent, "parameterization", iterations, seed, directory=directory
    )
    dp_combustion = generate_combustion_datapackage("combustion", iterations, seed, directory=directory)
    dp_entsoe = generate_entsoe_datapackage("entsoe", iterations, seed, directory=directory)
    dp_markets = generate_markets_datapackage("markets", iterations, seed, directory=directory)
    datapackages = [dp_parameterization, dp_combustion, dp_entsoe, dp_markets]
    return datapackages


def run_mc_simulations_all_inputs(project, fp_ecoinvent, iterations, seed=42, correlations=True):
    """Run Monte Carlo simulations when all model inputs vary."""
    directory = GSA_DIR_CORR if correlations else GSA_DIR_INDP
    fp = directory / f"scores.all_inputs.{seed}.{iterations}.pickle"
    if fp.exists():
        scores = read_pickle(fp)
    else:
        datapackages = []
        if correlations:
            datapackages += create_all_datapackages(fp_ecoinvent, project, iterations, seed)
        scores = compute_consumption_lcia(project, iterations, seed, datapackages)
        write_pickle(scores, fp)

    masks = {"technosphere": None, "biosphere": None, "characterization": None}
    offset = get_lca_score_shift(project, masks)

    scores = np.array(scores) + offset

    return scores


def create_masked_vector_datapackage(project, tmask, bmask, cmask, tag):
    """Create datapackages that exclude masked inputs."""

    fp_datapackage = DATA_DIR / f"vector_dp.{tag}.zip"

    bd.projects.set_current(project)

    # Extract TECH datapackage values
    ei = bd.Database("ecoinvent 3.8 cutoff").datapackage()
    tei = ei.filter_by_attribute('matrix', 'technosphere_matrix')
    tindices = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.indices')[0]
    tdata = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.data')[0]
    tflip = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.flip')[0]

    # Extract BIO datapackage values
    bei = ei.filter_by_attribute('matrix', 'biosphere_matrix')
    bindices = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.indices')[0]
    bdata = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.data')[0]

    # Extract CF datapackage values
    cf = bd.Method(("IPCC 2013", "climate change", "GWP 100a", "uncertain")).datapackage()
    cindices = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.indices')[0]
    cdata = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.data')[0]

    # Create datapackages

    dp = bwp.create_datapackage(
        fs=ZipFS(str(fp_datapackage), write=True),
        sequential=True,
    )
    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=tdata[tmask],
        name=f"{tag}.tech",
        indices_array=tindices[tmask],
        flip_array=tflip[tmask],
    )
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=bdata[bmask],
        name=f"{tag}.bio",
        indices_array=bindices[bmask],
    )
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=cdata[cmask],
        name=f"{tag}.cf",
        indices_array=cindices[cmask],
    )
    [d.update({"global_index": 1}) for d in dp.metadata['resources']]

    dp.finalize_serialization()

    return dp


def create_noninf_datapackage(project, cutoff, max_calc):

    # Extract all masks without non-influential inputs
    tag = f"cutoff_{cutoff:.0e}.maxcalc_{max_calc:.0e}"
    fp_tech = GSA_DIR / f"mask.tech.without_noninf.sct.{tag}.pickle"
    fp_bio = GSA_DIR / "mask.bio.without_noninf.pickle"
    fp_cf = GSA_DIR / "mask.cf.without_noninf.pickle"
    tmask = read_pickle(fp_tech)
    bmask = read_pickle(fp_bio)
    cmask = read_pickle(fp_cf)

    tag = "without_noninf"
    dp = create_masked_vector_datapackage(project, ~tmask, ~bmask, ~cmask, tag)

    masks = {"technosphere": tmask, "biosphere": bmask, "characterization": cmask}
    offset = get_lca_score_shift(project, masks)

    return dp, offset


def create_lowinf_lsa_datapackage(project, factor, cutoff, max_calc, num_lowinf):

    # Extract all masks without non-influential inputs
    tag = f"cutoff_{cutoff:.0e}.maxcalc_{max_calc:.0e}"
    fp_tech = GSA_DIR / f"mask.tech.without_lowinf.{num_lowinf}.lsa.factor_{factor}.{tag}.pickle"
    fp_bio = GSA_DIR / f"mask.bio.without_lowinf.{num_lowinf}.lsa.factor_{factor}.{tag}.pickle"
    fp_cf = GSA_DIR / f"mask.cf.without_lowinf.{num_lowinf}.lsa.factor_{factor}.{tag}.pickle"
    tmask = read_pickle(fp_tech)
    bmask = read_pickle(fp_bio)
    cmask = read_pickle(fp_cf)

    tag = f"without_lowinf_lsa.{num_lowinf}"
    dp = create_masked_vector_datapackage(project, ~tmask, ~bmask, ~cmask, tag)

    masks = {"technosphere": tmask, "biosphere": bmask, "characterization": cmask}
    offset = get_lca_score_shift(project, masks)

    return dp, offset


def create_lowinf_xgb_datapackage(project, num_lowinf, xgb_model_tag):

    # Extract all masks without non-influential inputs
    fp_tech = GSA_DIR_CORR / f"mask.tech.without_lowinf.{num_lowinf}.xgb.model_{xgb_model_tag}.pickle"
    fp_bio = GSA_DIR_CORR / f"mask.bio.without_lowinf.{num_lowinf}.xgb.model_{xgb_model_tag}.pickle"
    fp_cf = GSA_DIR_CORR / f"mask.cf.without_lowinf.{num_lowinf}.xgb.model_{xgb_model_tag}.pickle"
    tmask = read_pickle(fp_tech)
    bmask = read_pickle(fp_bio)
    cmask = read_pickle(fp_cf)

    tag = f"without_lowinf_xgb.{num_lowinf}"
    dp = create_masked_vector_datapackage(project, ~tmask, ~bmask, ~cmask, tag)

    masks = {"technosphere": tmask, "biosphere": bmask, "characterization": cmask}
    offset = get_lca_score_shift(project, masks)

    return dp, offset


def run_mc_simulations_masked(
        project, fp_ecoinvent, datapackage_masked, iterations, seed=42, tag="", correlations=True
):
    """Run Monte Carlo simulations without non-influential inputs, but with all sampling modules."""

    directory = GSA_DIR_CORR if correlations else GSA_DIR_INDP
    fp = directory / f"scores.{tag}.{seed}.{iterations}.pickle"

    if fp.exists():
        scores = read_pickle(fp)
    else:
        datapackages = [datapackage_masked]
        if correlations:
            datapackages_sampling_modules = create_all_datapackages(fp_ecoinvent, project, iterations, seed)
            datapackages += datapackages_sampling_modules

        scores = compute_consumption_lcia(project, iterations, seed, datapackages)

        write_pickle(scores, fp)

    return scores


def run_mc_simulations_wo_noninf(
        project, fp_ecoinvent, cutoff, max_calc, iterations, seed, num_noninf=None, correlations=True
):
    datapackage_noninf, offset = create_noninf_datapackage(project, cutoff, max_calc)
    tag = "without_noninf" if num_noninf is None else f"without_noninf.{num_noninf}"
    scores = run_mc_simulations_masked(project, fp_ecoinvent, datapackage_noninf, iterations, seed, tag, correlations)
    scores = np.array(scores) + offset
    return scores


def run_mc_simulations_wo_lowinf_lsa(
        project, fp_ecoinvent, factor, cutoff, max_calc, iterations, seed, num_lowinf, correlations=True
):
    datapackage_lowinf, offset = create_lowinf_lsa_datapackage(project, factor, cutoff, max_calc, num_lowinf)
    tag = f"without_lowinf_lsa.{num_lowinf}"
    scores = run_mc_simulations_masked(project, fp_ecoinvent, datapackage_lowinf, iterations, seed, tag, correlations)
    scores = np.array(scores) + offset
    return scores


def run_mc_simulations_wo_lowinf_xgb(project, fp_ecoinvent, xgb_model_tag, iterations, seed, num_lowinf):
    datapackage_lowinf, offset = create_lowinf_xgb_datapackage(project, num_lowinf, xgb_model_tag)
    tag = f"without_lowinf_xgb.model_{xgb_model_tag}.{num_lowinf}"
    scores = run_mc_simulations_masked(project, fp_ecoinvent, datapackage_lowinf, iterations, seed, tag)
    scores = np.array(scores) + offset
    return scores
