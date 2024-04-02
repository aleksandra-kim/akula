import numpy as np
from pathlib import Path
import bw_processing as bwp
import bw2data as bd
import bw2calc as bc
from fs.zipfs import ZipFS
from matrix_utils.resource_group import FakeRNG
from xgboost import XGBRegressor
import shap
import json

from ..utils import read_pickle, write_pickle, get_consumption_activity
from ..sensitivity_analysis import create_all_datapackages, create_lowinf_datapackage

GSA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "sensitivity-analysis"
SCREENING_DIR = GSA_DIR / "high-dimensional-screening"
SCREENING_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "datapackages"

MC_BATCH_SIZE = 5_000
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
    fp = SCREENING_DIR / f"{name}.zip"

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


def create_tech_bio_cf_datapackages(project, factor, cutoff, max_calc, iterations, seed, num_lowinf):

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
    dp_base = create_tech_bio_cf_datapackages(project, factor, cutoff, max_calc, iterations, seed, num_lowinf)
    dp_without_lowinf = create_lowinf_datapackage(project, factor, cutoff, max_calc, num_lowinf)
    dp_modules = create_all_datapackages(fp_ecoinvent, project, iterations, seed, SCREENING_DIR)
    dps = dp_base + [dp_without_lowinf] + dp_modules
    return dps


def compute_consumption_lcia_screening(project, fp_ecoinvent, factor, cutoff, max_calc, iterations, seed, num_lowinf):

    bd.projects.set_current(project)

    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    activity = get_consumption_activity()
    fu, pkgs, _ = bd.prepare_lca_inputs({activity: 1}, method=method, remapping=False)

    datapackages = get_datapackages_screening(
        project, fp_ecoinvent, factor, cutoff, max_calc, iterations, seed, num_lowinf
    )

    lca = bc.LCA(
        demand=fu,
        data_objs=pkgs + datapackages,
        use_arrays=True,
        use_distributions=False,
        # seed_override=seed,
    )
    lca.lci()
    lca.lcia()

    scores = [lca.score for _ in zip(range(iterations), lca)]

    return scores


def get_seeds(iterations, seed):
    """Generate random seeds for running MC simulations in batches."""
    starts = np.arange(0, iterations, MC_BATCH_SIZE)
    n_batches = len(starts)
    np.random.seed(seed)
    # N_BATCH_CONST is used to generate same random seeds for different values of `n_batches`. This way LCIA scores
    # can be reused if more MC iterations are needed.
    seeds = np.random.randint(100_000, 999_999, max(N_BATCH_CONST, n_batches)).tolist()[:n_batches]
    return starts, n_batches, seeds


def run_mc_simulations_screening(project, fp_ecoinvent, factor, cutoff, max_calc, iterations, seed, num_lowinf):
    """Run Monte Carlo simulations for high-dimensional screening."""
    fp = SCREENING_DIR / f"scores.without_lowinf.{num_lowinf}.{seed}.{iterations}.pickle"

    if fp.exists():
        scores = read_pickle(fp)

    else:
        scores = []
        starts, n_batches, seeds = get_seeds(iterations, seed)

        for i in range(n_batches):
            current_iterations = MC_BATCH_SIZE if i < n_batches - 1 else iterations - starts[i]
            print(f"MC simulations for screening -- random seed {i+1:2d} / {n_batches:2d} -- {seeds[i]}")

            fp_current = SCREENING_DIR / f"scores.without_lowinf.{num_lowinf}.{seeds[i]}.{current_iterations}.pickle"

            if fp_current.exists():
                scores_current = read_pickle(fp_current)
            else:
                scores_current = compute_consumption_lcia_screening(
                    project, fp_ecoinvent, factor, cutoff, max_calc, current_iterations, seeds[i], num_lowinf
                )
                write_pickle(scores_current, fp_current)

            scores += scores_current

        write_pickle(scores, fp)

    return scores


def get_y_scores(iterations, seed, num_lowinf):
    fp = SCREENING_DIR / f"scores.without_lowinf.{num_lowinf}.{seed}.{iterations}.pickle"
    scores = read_pickle(fp).flatten()
    return scores


def get_x_data_technosphere(iterations, seed):
    fp = SCREENING_DIR / f"technosphere-{seed}-{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource("technosphere_matrix.indices")[0]
    data = dp.get_resource("technosphere_matrix.data")[0]
    return data, indices


def get_x_data_biosphere(iterations, seed):
    fp = SCREENING_DIR / f"biosphere-{seed}-{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource("biosphere_matrix.indices")[0]
    data = dp.get_resource("biosphere_matrix.data")[0]
    return data, indices


def get_x_data_characterization(iterations, seed):
    fp = SCREENING_DIR / f"characterization-{seed}-{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource("characterization_matrix.indices")[0]
    data = dp.get_resource("characterization_matrix.data")[0]
    return data, indices


def get_x_data_parameterization(iterations, seed):
    fp = SCREENING_DIR / f"parameterization-parameters-{seed}-{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource("ecoinvent-parameters.indices")[0]
    data = dp.get_resource("ecoinvent-parameters.data")[0]
    return data, indices


def get_x_data_combustion(iterations, seed):
    fp = SCREENING_DIR / f"combustion-{seed}-{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource("combustion-tech.indices")[0]
    data = dp.get_resource("combustion-tech.data")[0]
    return data, indices


def get_x_data_entsoe(iterations, seed):
    fp = SCREENING_DIR / f"entsoe-{seed}-{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource("entsoe.indices")[0]
    data = dp.get_resource("entsoe.data")[0]
    return data, indices


def get_x_data_markets(iterations, seed):
    fp = SCREENING_DIR / f"markets-{seed}-{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource("markets.indices")[0]
    data = dp.get_resource("markets.data")[0]
    return data, indices


def get_x_data(iterations, seed):
    """Read input data from datapackages and return indices and data."""

    starts, n_batches, seeds = get_seeds(iterations, seed)

    data_technosphere = []
    data_biosphere = []
    data_characterization = []
    data_parameterization = []
    data_combustion = []
    data_entsoe = []
    data_markets = []

    for i in range(n_batches):
        current_iterations = MC_BATCH_SIZE if i < n_batches - 1 else iterations - starts[i]

        tech_data, tech_indices = get_x_data_technosphere(current_iterations, seeds[i])
        bio_data, bio_indices = get_x_data_biosphere(current_iterations, seeds[i])
        cf_data, cf_indices = get_x_data_characterization(current_iterations, seeds[i])
        pdata, pindices = get_x_data_parameterization(current_iterations, seeds[i])
        cdata, cindices = get_x_data_combustion(current_iterations, seeds[i])
        edata, eindices = get_x_data_entsoe(current_iterations, seeds[i])
        mdata, mindices = get_x_data_markets(current_iterations, seeds[i])

        data_technosphere.append(tech_data)
        data_biosphere.append(bio_data)
        data_characterization.append(cf_data)
        data_parameterization.append(pdata)
        data_combustion.append(cdata)
        data_entsoe.append(edata)
        data_markets.append(mdata)

    data_technosphere = np.hstack(data_technosphere)
    data_biosphere = np.hstack(data_biosphere)
    data_characterization = np.hstack(data_characterization)
    data_parameterization = np.hstack(data_parameterization)
    data_combustion = np.hstack(data_combustion)
    data_entsoe = np.hstack(data_entsoe)
    data_markets = np.hstack(data_markets)

    data = np.vstack([data_technosphere, data_biosphere, data_characterization, data_parameterization])

    indices = {
        "technosphere": tech_indices,
        "biosphere": bio_indices,
        "characterization": cf_indices,
        "parameterization": pindices,
    }

    return data, indices


def train_xgboost_model(tag, iterations, seed, num_lowinf, train_test_split=0.2):
    """Train gradient boosted tree regressor."""

    fp = SCREENING_DIR / f"xgboost.{tag}.pickle"

    # Read X and Y data
    X, _ = get_x_data(iterations, seed)
    Y = get_y_scores(iterations, seed, num_lowinf)
    split = int(train_test_split * X.shape[0])
    X_train = X[:-split, :]
    X_test = X[-split:, :]
    Y_train = Y[:-split]
    Y_test = Y[-split:]
    del X, Y

    if fp.exists():
        model = read_pickle(fp)

    else:
        # Define the model
        fp_params = SCREENING_DIR / f"xgboost.{tag}.params.json"
        params = dict(
            n_estimators=10,
            max_depth=4,
            eta=0.1,
            subsample=0.2,
            colsample_bytree=0.9,
            base_score=np.mean(Y_train),
            booster='gbtree',
            #     tree_method="hist",
            #     objective='reg:linear',
            verbose_eval=100,
            early_stopping_rounds=20,
        )
        # Write params into a json file
        with open(fp_params, 'w') as f:
            json.dump(params, f)

        # Define the model and train it
        model = XGBRegressor(**params)
        eval_set = [(X_train, Y_train), (X_test, Y_test)]
        model.fit(X_train, Y_train, eval_metric=["error", "rmse"], eval_set=eval_set, verbose=True)

    # Print results
    score_train = model.score(X_train, Y_train)
    score_test = model.score(X_test, Y_test)
    print(f"{score_train:4.3f}  train score")
    print(f"{score_test:4.3f}  test score")

    return model


def compute_shap_values(tag, iterations, seed, num_inf):
    fp = SCREENING_DIR / f"xgboost.{tag}.pickle"
    model = read_pickle(fp)
    explainer = shap.TreeExplainer(model)
    X, indices = get_x_data(iterations, seed)
    shap_values = explainer.shap_values(X)
    where = np.argsort(shap_values)[:num_inf]
    print(shap_values[where])
    print(indices[where])
