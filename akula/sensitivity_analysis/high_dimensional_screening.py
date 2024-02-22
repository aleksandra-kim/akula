import numpy as np
from pathlib import Path
import bw_processing as bwp
import bw2data as bd
import bw2calc as bc
from fs.zipfs import ZipFS
from matrix_utils.resource_group import FakeRNG

from ..utils import read_pickle, write_pickle, get_consumption_activity
from ..sensitivity_analysis import create_all_datapackages, create_lowinf_datapackage

GSA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "sensitivity-analysis"
DATA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "datapackages"

MC_BATCH_SIZE = 10
N_BATCH_CONST = 2000


def get_lca_stochastic(project, seed):

    bd.projects.set_current(project)

    act = get_consumption_activity()
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    fu_mapped, pkgs, _ = bd.prepare_lca_inputs(demand={act: 1}, method=method, remapping=False)

    lca = bc.LCA(demand=fu_mapped, data_objs=pkgs, use_distributions=True, seed_override=seed)
    lca.lci()
    lca.lcia()

    return lca


def create_background_datapackage(project, matrix_type, mask, num_samples, seed=42):

    name = f"{matrix_type}.{seed}.{num_samples}"
    fp = DATA_DIR / f"{name}.zip"

    if fp.exists():
        dp = bwp.load_datapackage(ZipFS(str(fp)))

    else:
        lca = get_lca_stochastic(project, seed)

        dp = bwp.create_datapackage(
            fs=ZipFS(str(fp), write=True),
            name=name,
            seed=seed,
            sequential=True,
        )

        num_resources = 4 if matrix_type == "technosphere" else 3

        obj = getattr(lca, f"{matrix_type}_mm")
        indices_array = np.hstack([
            group.package.data[0] for group in obj.groups
            if (not isinstance(group.rng, FakeRNG)) and (not group.empty) and (len(group.package.data) == num_resources)
        ])

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
            flip_array = np.hstack([
                group.flip for group in obj.groups if (not isinstance(group.rng, FakeRNG))
                and (not group.empty) and (len(group.package.data) == num_resources)
            ])
            dp.add_persistent_array(
                matrix=f"{matrix_type}_matrix",
                data_array=data_array,
                name=name,
                indices_array=indices_array[mask],
                flip_array=flip_array[mask],
            )
        else:
            dp.add_persistent_array(
                matrix=f"{matrix_type}_matrix",
                data_array=data_array,
                name=name,
                indices_array=indices_array[mask],
            )

            [
                d.update({"global_index": 1}) for d in dp.metadata['resources']
                if d['matrix'] == "characterization_matrix"
            ]

            dp.finalize_serialization()

    return dp


def create_base_datapackages(project, factor, cutoff, max_calc, iterations, seed, num_lowinf):

    tag = f"cutoff_{cutoff:.0e}.maxcalc_{max_calc:.0e}"

    fp_tech = GSA_DIR / f"mask.tech.without_lowinf.{num_lowinf}.lsa.factor_{factor}.{tag}.pickle"
    fp_bio = GSA_DIR / f"mask.bio.without_lowinf.{num_lowinf}.lsa.factor_{factor}.{tag}.pickle"
    fp_cf = GSA_DIR / f"mask.cf.without_lowinf.{num_lowinf}.lsa.factor_{factor}.{tag}.pickle"

    tmask = read_pickle(fp_tech)
    bmask = read_pickle(fp_bio)
    cmask = read_pickle(fp_cf)

    tdp = create_background_datapackage(project, "technosphere", tmask, iterations, seed)
    bdp = create_background_datapackage(project, "biosphere", bmask, iterations, seed)
    cdp = create_background_datapackage(project, "characterization", cmask, iterations, seed)

    return [tdp, bdp, cdp]


def get_datapackages_screening(project, fp_ecoinvent, factor, cutoff, max_calc, iterations, seed, num_lowinf):
    """Create all datapackages for high-dimensional screening."""
    dp_base = create_base_datapackages(project, factor, cutoff, max_calc, iterations, seed, num_lowinf)
    dp_without_lowinf = create_lowinf_datapackage(project, factor, cutoff, max_calc, num_lowinf)
    dp_modules = create_all_datapackages(fp_ecoinvent, project, iterations, seed)
    dps = dp_base + [dp_without_lowinf] + dp_modules
    return dps


def compute_consumption_lcia_screening(project, fp_ecoinvent, factor, cutoff, max_calc, iterations, seed, num_lowinf):

    bd.projects.set_current(project)

    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    activity = get_consumption_activity()
    fu, _, _ = bd.prepare_lca_inputs({activity: 1}, method=method, remapping=False)

    datapackages = get_datapackages_screening(
        project, fp_ecoinvent, factor, cutoff, max_calc, iterations, seed, num_lowinf
    )

    lca = bc.LCA(
        demand=fu,
        data_objs=datapackages,
        use_arrays=True,
        use_distributions=False,
        # seed_override=seed,
    )
    lca.lci()
    lca.lcia()

    scores = [lca.score for _ in zip(range(iterations), lca)]

    return scores


def run_mc_simulations_screening(project, fp_ecoinvent, factor, cutoff, max_calc, iterations, seed, num_lowinf):
    """Run Monte Carlo simulations for high-dimensional screening."""
    fp = GSA_DIR / f"scores.without_lowinf.{num_lowinf}.{seed}.{iterations}.pickle"

    if fp.exists():
        scores = read_pickle(fp)

    else:
        scores = []

        starts = np.arange(0, iterations, MC_BATCH_SIZE)
        n_batches = len(starts)
        np.random.seed(seed)
        # N_BATCH_CONST is used to generate same random seeds for different values of `n_batches`. This way LCIA scores
        # can be reused if more MC iterations are needed.
        seeds = np.random.randint(100_000, 999_999, max(N_BATCH_CONST, n_batches)).tolist()[:n_batches]

        for i in range(n_batches):
            current_iterations = MC_BATCH_SIZE if i < n_batches - 1 else iterations - starts[i]
            print(f"MC simulations for screening -- random seed {i+1:2d} / {n_batches:2d} -- {seeds[i]}")

            scores_current = compute_consumption_lcia_screening(
                project, fp_ecoinvent, factor, cutoff, max_calc, current_iterations, seeds[i], num_lowinf
            )

            scores += scores_current

        write_pickle(scores, fp)

    return scores


def get_scores(iterations, seed, num_lowinf):
    fp = GSA_DIR / f"scores.without_lowinf.{num_lowinf}.{seed}.{iterations}.pickle"
    scores = read_pickle(fp)
    return scores


def get_input_data_parameterization(iterations, seed):
    fp = DATA_DIR / f"ecoinvent-parameters-{seed}.{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource("ecoinvent-parameters.indices")[0]
    data = dp.get_resource("ecoinvent-parameters.data")[0]
    return indices, data


def get_input_data_combustion(iterations, seed):
    fp = DATA_DIR / f"combustion-{seed}.{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource("combustion-tech.indices")[0]
    data = dp.get_resource("combustion-tech.data")[0]
    return indices, data


def get_input_data_entsoe(iterations, seed):
    fp = DATA_DIR / f"entsoe-{seed}.{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource("timeseries ENTSO electricity values.indices")[0]
    data = dp.get_resource("timeseries ENTSO electricity values.data")[0]
    return indices, data


def get_input_data_markets(iterations, seed):
    fp = DATA_DIR / f"markets-{seed}.{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource("markets.indices")[0]
    data = dp.get_resource("markets.data")[0]
    return indices, data


def get_input_data(iterations, seed):
    """Read input data from datapackages and return indices and data."""
    starts = np.arange(0, iterations, MC_BATCH_SIZE)
    n_batches = len(starts)

    np.random.seed(seed)
    seeds = np.random.randint(100_000, 999_999, max(N_BATCH_CONST, n_batches)).tolist()[:n_batches]

    data_parameterization = []
    data_combustion = []
    data_entsoe = []
    data_markets = []

    for i in range(n_batches):
        current_iterations = MC_BATCH_SIZE if i < n_batches - 1 else iterations - starts[i]

        pindices, pdata = get_input_data_parameterization(current_iterations, seeds[i])
        cindices, cdata = get_input_data_combustion(current_iterations, seeds[i])
        eindices, edata = get_input_data_entsoe(current_iterations, seeds[i])
        mindices, mdata = get_input_data_markets(current_iterations, seeds[i])

        data_parameterization.append(pdata)
        data_combustion.append(cdata)
        data_entsoe.append(edata)
        data_markets.append(mdata)

    data_parameterization = np.hstack(data_parameterization)
    data_combustion = np.hstack(data_combustion)
    data_entsoe = np.hstack(data_entsoe)
    data_markets = np.hstack(data_markets)

    indices = np.hstack([pindices, cindices, eindices, mindices], dtype=bwp.INDICES_DTYPE)
    data = np.vstack([data_parameterization, data_combustion, data_entsoe, data_markets])

    return indices, data


def train_xgboost_model():
    return
