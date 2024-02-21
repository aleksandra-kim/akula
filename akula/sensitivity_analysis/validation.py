from pathlib import Path
import bw2data as bd
import bw_processing as bwp
from fs.zipfs import ZipFS

from ..utils import read_pickle, write_pickle
from ..monte_carlo import compute_consumption_lcia
from ..parameterization import generate_parameterization_datapackage
from ..combustion import generate_combustion_datapackage
from ..electricity import generate_entsoe_datapackage
from ..markets import generate_markets_datapackage

GSA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "sensitivity-analysis"
DATA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "datapackages"


def create_all_datapackages(fp_ecoinvent, project, iterations, seed=42):
    bd.projects.set_current(project)
    _, dp_parameterization = generate_parameterization_datapackage(
        fp_ecoinvent, "parameterization", iterations, seed
    )
    dp_combustion = generate_combustion_datapackage("combustion", iterations, seed)
    dp_entsoe = generate_entsoe_datapackage("entsoe", iterations, seed)
    dp_markets = generate_markets_datapackage("markets", iterations, seed)
    datapackages = [dp_parameterization, dp_combustion, dp_entsoe, dp_markets]
    return datapackages


def run_mc_simulations_all_inputs(project, fp_ecoinvent, iterations, seed=42):
    """Run Monte Carlo simulations when all model inputs vary."""
    fp = GSA_DIR / f"scores.validation.all_inputs.{seed}.{iterations}.pickle"
    if fp.exists():
        scores = read_pickle(fp)
    else:
        datapackages = create_all_datapackages(fp_ecoinvent, project, iterations, seed)
        scores = compute_consumption_lcia(project, iterations, seed, datapackages)
        write_pickle(scores, fp)
    return scores


def create_masked_vector_datapackage(project, tmask, bmask, cmask, tag):
    """Create datapackages that exclude masked inputs."""

    fp_datapackage = DATA_DIR / f"validation.mask.{tag}.zip"

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
        name=f"validation.{tag}.tech",
        indices_array=tindices[tmask],
        flip_array=tflip[tmask],
    )
    dp.add_persistent_vector(
        matrix="biosphere_matrix",
        data_array=bdata[bmask],
        name=f"validation.{tag}.bio",
        indices_array=bindices[bmask],
    )
    dp.add_persistent_vector(
        matrix="characterization_matrix",
        data_array=cdata[cmask],
        name=f"validation.{tag}.cf",
        indices_array=cindices[cmask],
    )
    [d.update({"global_index": 1}) for d in dp.metadata['resources']]

    dp.finalize_serialization()

    return dp


def create_noninf_datapackage(project, cutoff, max_calc):

    # Extract all masks without non-influential inputs
    fp_tech = GSA_DIR / f"mask.tech.without_noninf.sct.cutoff_{cutoff:.0e}.maxcalc_{max_calc:.0e}.pickle"
    fp_bio = GSA_DIR / "mask.bio.without_noninf.pickle"
    fp_cf = GSA_DIR / "mask.cf.without_noninf.pickle"
    tmask = read_pickle(fp_tech)
    bmask = read_pickle(fp_bio)
    cmask = read_pickle(fp_cf)

    tag = "noninf"
    dp = create_masked_vector_datapackage(project, ~tmask, ~bmask, ~cmask, tag)

    return dp


def create_lowinf_datapackage(project, factor, cutoff, max_calc):

    # Extract all masks without non-influential inputs
    tag = f"cutoff_{cutoff:.0e}.maxcalc_{max_calc:.0e}"
    fp_tech = GSA_DIR / f"mask.tech.without_lowinf.lsa.factor_{factor}.{tag}.pickle"
    fp_bio = GSA_DIR / f"mask.bio.without_lowinf.lsa.factor_{factor}.{tag}.pickle"
    fp_cf = GSA_DIR / f"mask.cf.without_lowinf.lsa.factor_{factor}.{tag}.pickle"
    tmask = read_pickle(fp_tech)
    bmask = read_pickle(fp_bio)
    cmask = read_pickle(fp_cf)

    tag = "lowinf"
    dp = create_masked_vector_datapackage(project, ~tmask, ~bmask, ~cmask, tag)

    return dp


def run_mc_simulations_masked(project, fp_ecoinvent, datapackage_masked, iterations, seed=42, tag=""):
    """Run Monte Carlo simulations without non-influential inputs, but with all sampling modules."""

    fp = GSA_DIR / f"scores.validation.{tag}.{seed}.{iterations}.pickle"

    if fp.exists():
        scores = read_pickle(fp)
    else:
        datapackages_sampling_modules = create_all_datapackages(fp_ecoinvent, project, iterations, seed)
        datapackages = datapackages_sampling_modules + [datapackage_masked]
        scores = compute_consumption_lcia(project, iterations, seed, datapackages)

        write_pickle(scores, fp)

    return scores


def run_mc_simulations_wo_noninf(project, fp_ecoinvent, cutoff, max_calc, iterations, seed):
    datapackage_noninf = create_noninf_datapackage(project, cutoff, max_calc)
    tag = "wo_noninf"
    scores = run_mc_simulations_masked(project, fp_ecoinvent, datapackage_noninf, iterations, seed, tag)
    return scores


def run_mc_simulations_wo_lowinf(project, fp_ecoinvent, factor, cutoff, max_calc, iterations, seed=42):
    datapackage_lowinf = create_lowinf_datapackage(project, factor, cutoff, max_calc)
    tag = "wo_lowinf"
    scores = run_mc_simulations_masked(project, fp_ecoinvent, datapackage_lowinf, iterations, seed, tag)
    return scores
