import numpy as np
from pathlib import Path
import bw_processing as bwp
import bw2data as bd
import bw2calc as bc
from fs.zipfs import ZipFS
from matrix_utils.resource_group import FakeRNG
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score
import json

from .utils import get_mask
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
    dp_modules = create_all_datapackages(fp_ecoinvent, project, iterations, seed, SCREENING_DIR)
    dps = dp_base + dp_modules
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


def get_random_seeds(iterations, seed):
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
        starts, n_batches, seeds = get_random_seeds(iterations, seed)

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
    scores = np.array(read_pickle(fp))
    return scores


def get_x_data_technosphere(iterations, seed):
    name = f"technosphere.{seed}.{iterations}"
    fp = SCREENING_DIR / f"{name}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource(f"{name}.indices")[0]
    data = dp.get_resource(f"{name}.data")[0]
    return data, indices


def get_x_data_biosphere(iterations, seed):
    name = f"biosphere.{seed}.{iterations}"
    fp = SCREENING_DIR / f"{name}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource(f"{name}.indices")[0]
    data = dp.get_resource(f"{name}.data")[0]
    return data, indices


def get_x_data_characterization(iterations, seed):
    name = f"characterization.{seed}.{iterations}"
    fp = SCREENING_DIR / f"{name}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    indices = dp.get_resource(f"{name}.indices")[0]
    data = dp.get_resource(f"{name}.data")[0]
    return data, indices


def get_x_data_parameterization(iterations, seed):
    # Parameters
    fp = SCREENING_DIR / f"parameterization-parameters-{seed}-{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    param_indices = dp.get_resource("ecoinvent-parameters.indices")[0]
    param_data = dp.get_resource("ecoinvent-parameters.data")[0]
    # Biosphere and technosphere exchanges
    fp = SCREENING_DIR / f"parameterization-exchanges-{seed}-{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    tech_indices = dp.get_resource("parameterized-tech.indices")[0]
    bio_indices = dp.get_resource("parameterized-bio.indices")[0]
    return param_data, param_indices, tech_indices, bio_indices


def get_x_data_combustion(iterations, seed):
    fp = SCREENING_DIR / f"combustion-{seed}-{iterations}.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    ctech_data = dp.get_resource("combustion-tech.data")[0]
    ctech_indices = dp.get_resource("combustion-tech.indices")[0]
    cbio_indices = dp.get_resource("combustion-bio.indices")[0]
    return ctech_data, ctech_indices, cbio_indices


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

    starts, n_batches, seeds = get_random_seeds(iterations, seed)

    data_technosphere = []
    data_biosphere = []
    data_characterization = []
    data_parameterization = []
    data_combustion = []
    data_entsoe = []
    data_markets = []

    for i in range(n_batches):
        current_iterations = MC_BATCH_SIZE if i < n_batches - 1 else iterations - starts[i]

        # if i < 5:
        #     continue

        tech_data, tech_indices = get_x_data_technosphere(current_iterations, seeds[i])
        bio_data, bio_indices = get_x_data_biosphere(current_iterations, seeds[i])
        cf_data, cf_indices = get_x_data_characterization(current_iterations, seeds[i])
        pdata, pindices, ptech_indices, pbio_indices = get_x_data_parameterization(current_iterations, seeds[i])
        ctech_data, ctech_indices, cbio_indices = get_x_data_combustion(current_iterations, seeds[i])
        etech_data, etech_indices = get_x_data_entsoe(current_iterations, seeds[i])
        mtech_data, mtech_indices = get_x_data_markets(current_iterations, seeds[i])

        data_technosphere.append(tech_data)
        data_biosphere.append(bio_data)
        data_characterization.append(cf_data)
        data_parameterization.append(pdata)
        data_combustion.append(ctech_data)
        data_entsoe.append(etech_data)
        data_markets.append(mtech_data)

    data_technosphere = np.hstack(data_technosphere)
    data_biosphere = np.hstack(data_biosphere)
    data_characterization = np.hstack(data_characterization)
    data_parameterization = np.hstack(data_parameterization)
    data_combustion = np.hstack(data_combustion)
    data_entsoe = np.hstack(data_entsoe)
    data_markets = np.hstack(data_markets)

    # 1. Parameterized exchanges from technosphere and biosphere should be removed, because they are dependent inputs
    ptech_mask = get_mask(tech_indices, ptech_indices)
    pbio_mask = get_mask(bio_indices, pbio_indices)
    # 2.1 Combustion biosphere exchanges should be removed, because they are dependent inputs
    cbio_mask = get_mask(bio_indices, cbio_indices)
    # 2.2 Combustion technosphere exchanges should replace respective technosphere exchanges
    ctech_mask = get_mask(tech_indices, ctech_indices)
    # 3 Electricity data from ENTSOE should replace respective technosphere exchanges
    etech_mask = get_mask(tech_indices, etech_indices)
    # 4 Market data should replace respective technosphere exchanges
    mtech_mask = get_mask(tech_indices, mtech_indices)
    # Collect masks from all sampling modules
    tech_mask = ~(ptech_mask | ctech_mask | etech_mask | mtech_mask)
    bio_mask = ~(pbio_mask | cbio_mask)

    data_technosphere = np.vstack([data_technosphere[tech_mask, :], data_combustion, data_entsoe, data_markets])
    data_biosphere = data_biosphere[bio_mask, :]
    data = np.vstack([
        data_technosphere,
        data_biosphere,
        data_characterization,
        data_parameterization
    ])
    tech_indices = np.hstack([tech_indices[tech_mask], ctech_indices, etech_indices, mtech_indices],
                             dtype=bwp.INDICES_DTYPE)
    bio_indices = bio_indices[bio_mask]
    indices = {
        "technosphere": tech_indices,
        "biosphere": bio_indices,
        "characterization": cf_indices,
        "parameterization": pindices,
    }

    return data, indices


def get_x_data_v2(iterations, seed):
    """Read input data from datapackages and return indices and data."""

    starts, n_batches, seeds = get_random_seeds(iterations, seed)

    data_technosphere = []
    data_biosphere = []
    data_characterization = []
    data_parameterization = []
    data_combustion = []
    data_entsoe = []
    data_markets = []

    for i in range(n_batches):
        current_iterations = MC_BATCH_SIZE if i < n_batches - 1 else iterations - starts[i]

        if i < 20:
            continue

        print(i)

        tech_data, tech_indices = get_x_data_technosphere(current_iterations, seeds[i])
        bio_data, bio_indices = get_x_data_biosphere(current_iterations, seeds[i])
        cf_data, cf_indices = get_x_data_characterization(current_iterations, seeds[i])
        pdata, pindices, ptech_indices, pbio_indices = get_x_data_parameterization(current_iterations, seeds[i])
        ctech_data, ctech_indices, cbio_indices = get_x_data_combustion(current_iterations, seeds[i])
        etech_data, etech_indices = get_x_data_entsoe(current_iterations, seeds[i])
        mtech_data, mtech_indices = get_x_data_markets(current_iterations, seeds[i])

        data_technosphere.append(tech_data)
        data_biosphere.append(bio_data)
        data_characterization.append(cf_data)
        data_parameterization.append(pdata)
        data_combustion.append(ctech_data)
        data_entsoe.append(etech_data)
        data_markets.append(mtech_data)

    data_technosphere = np.hstack(data_technosphere)
    data_biosphere = np.hstack(data_biosphere)
    data_characterization = np.hstack(data_characterization)
    data_parameterization = np.hstack(data_parameterization)
    data_combustion = np.hstack(data_combustion)
    data_entsoe = np.hstack(data_entsoe)
    data_markets = np.hstack(data_markets)

    data_technosphere = np.vstack([data_technosphere, data_combustion, data_entsoe, data_markets])
    data = np.vstack([
        data_technosphere,
        data_biosphere,
        data_characterization,
        data_parameterization
    ])
    tech_indices = np.hstack([tech_indices, ctech_indices, etech_indices, mtech_indices],
                             dtype=bwp.INDICES_DTYPE)
    bio_indices = bio_indices
    indices = {
        "technosphere": tech_indices,
        "biosphere": bio_indices,
        "characterization": cf_indices,
        "parameterization": pindices,
    }

    return data, indices


def train_xgboost_model(tag, iterations, seed, num_lowinf, test_size=0.2):
    """Train gradient boosted tree regressor."""

    fp = SCREENING_DIR / f"xgboost.{tag}.pickle"

    # Read X and Y data
    X, _ = get_x_data(iterations, seed)
    X = X.T
    Y = get_y_scores(iterations, seed, num_lowinf)
    write_pickle(X, SCREENING_DIR / "X_100_125.pickle")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed, shuffle=False,
    )
    del X, Y
    dtrain = xgb.DMatrix(X_train, Y_train)
    X_dtest = xgb.DMatrix(X_test)

    if fp.exists():
        model = read_pickle(fp)

    else:
        # Define the model
        fp_params = SCREENING_DIR / f"xgboost.{tag}.params.json"
        params = dict(
            base_score=np.mean(Y_train),  # the initial prediction score of all instances, global bias
            n_estimators=600,           # number of gradient boosted trees
            max_depth=4,               # maximum tree depth for base learners
            learning_rate=0.15,        # boosting learning rate, xgb's `eta`
            verbosity=3,               # degree of verbosity, valid values are 0 (silent) - 3 (debug)
            # booster='gbtree',          # specify which booster to use: gbtree, gblinear or dart
            gamma=0,                   # minimum loss reduction to make a further partition on a leaf node of the tree
            subsample=0.3,             # subsample ratio of the training instance
            colsample_bytree=0.9,      # subsample ratio of columns when constructing each tree
            reg_alpha=0,               # L1 regularization term on weights (xgb’s alpha)
            reg_lambda=0,              # L2 regularization term on weights (xgb’s lambda)
            # importance_type="gain",    # for tree models: “gain”, “weight”, “cover”, “total_gain” or “total_cover”
            # early_stopping_rounds=30,  # validation metric needs to improve at least once in every early_stopping_rounds
            # eval_metric=["error", "rmse"],
            random_state=seed,
            # tree_method="hist",
            # objective='reg:linear',
            min_child_weight=300,
        )
        # Write params into a json file
        with open(fp_params, 'w') as f:
            json.dump(params, f)

        # Train the model
        model = xgb.train(params, dtrain, num_boost_round=params["n_estimators"])
        write_pickle(model, fp)

    # Print results
    # score_train = model.score(X_train, Y_train)
    # score_test = model.score(X_test, Y_test)
    # print(f"{score_train:4.3f}  train score")
    # print(f"{score_test:4.3f}  test score")
    y_pred = model.predict(X_dtest)
    r2 = r2_score(Y_test, y_pred)
    explained_variance = explained_variance_score(Y_test, y_pred)
    print(r2, explained_variance)
    #
    # eval_set = [(X_train, Y_train), (X_test, Y_test)]
    # model.fit(X_train, Y_train,  eval_set=eval_set, verbose=True)

    return model


def get_masks_inf(tag, project, fp_ecoinvent, factor, cutoff, max_calc, iterations, seed, num_lowinf, num_inf):
    # XGBoost model
    fp = SCREENING_DIR / f"xgboost.{tag}.pickle"
    model = read_pickle(fp)
    feature_importances = model.feature_importances
    where = np.argsort(feature_importances)[:num_inf]
    # Determine which model inputs are influential
    _, indices = get_x_data(iterations, seed)
    indices_all = np.hstack([inds for inds in indices.values()])

    len_tech = indices["technosphere"]
    where_tech = where < len_tech
    inds_tech = indices["technosphere"][where_tech]
    return
