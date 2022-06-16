from collections import defaultdict
from pathlib import Path
from copy import deepcopy

import bw2data as bd
import bw_processing as bwp
import numpy as np
from fs.zipfs import ZipFS
from scipy.stats import dirichlet
from thefuzz import fuzz
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from gsa_framework.utils import read_pickle, write_pickle

from utils import setup_bw_project, get_activities_from_indices

DATA_DIR = Path(__file__).parent.resolve() / "data"
SAMPLES = 25000


def similar_exact(a, b):
    """Exact comparison between `a` and `b` strings."""
    return a == b


def similar_fuzzy(a, b):
    """Fuzzy comparison between `a` and `b` strings using partial ratio."""
    return fuzz.partial_ratio(a, b) > 90 or fuzz.ratio(a, b) > 40


def find_markets(database, similarity_func, check_uncertainty):
    """Find markets based on similar reference product names based on exact or fuzzy string comparison."""

    db = bd.Database(database)

    found = {}

    for act in tqdm(db):
        rp = act.get("reference product")
        if not rp:
            continue

        inpts = defaultdict(list)
        for exc in act.technosphere():
            if exc.input == exc.output:
                continue
            elif check_uncertainty and exc.get("uncertainty type", 0) < 2:
                continue
            elif exc.input.get("reference product", None) is None:
                continue
            inpts[exc.input["reference product"]].append(exc)

        for key, lst in inpts.items():
            # print(key[:50], sum([exc["amount"] for exc in lst]))
            if (
                len(lst) > 1
                and similarity_func(rp, key)
                and 0.98 <= sum([exc["amount"] for exc in lst]) <= 1.02
            ):
                found[act] = lst

        # print(rp, len(found))

    return found


def get_beta_variance(a, b):
    """Compute variance of beta distribution based on shape parameters `a` and `b`."""
    return a*b/(a+b)**2/(a+b+1)


def get_beta_skewness(a, b):
    """Compute skewness of beta distribution based on shape parameters `a` and `b`."""
    return 2*(b-a)*((a+b+1)**0.5) / (a+b+2) / (a*b)**0.5


def get_lognormal_variance(loc, scale):
    """Compute variance of lognormal distribution based on parameters `loc` and `scale`."""
    return (np.exp(scale**2)-1) * np.exp(2*loc+scale**2)


def get_lognormal_skewness(scale):
    """Compute skewness of lognormal distribution based on parameters `loc` and `scale`."""
    return (np.exp(scale**2)+2) * ((np.exp(scale**2)-1)**0.5)


def select_contributing_exchanges(amounts_exchanges, use_threshold, return_scores=False):
    """Select exchanges in the given market that have contribution scores higher than average."""

    bd.projects.set_current("GSA for archetypes")
    lca = setup_bw_project()

    scores = {}
    for amount, exc in amounts_exchanges.items():
        lca.redo_lci({exc.input.id: amount})
        lca.redo_lcia()
        scores[exc.input] = lca.score

    threshold = np.mean(list(scores.values()))

    exchanges = {}
    for amount, exc in amounts_exchanges.items():
        if scores[exc.input] >= threshold or use_threshold:
            if exc['uncertainty type'] != 2:
                print(exc['uncertainty type'])
            exchanges[amount] = exc
    if return_scores:
        return exchanges, scores
    else:
        return exchanges


def select_higher_amount_exchanges(amounts_exchanges, use_average=True):
    """Select exchanges in the given market that have amounts higher than average."""

    alphas = list(amounts_exchanges.keys())
    threshold = np.mean(alphas)

    exchanges = {}

    for amount, exc in amounts_exchanges.items():
        if amount >= threshold or use_average:
            if exc['uncertainty type'] != 2:
                print(exc['uncertainty type'])
        exchanges[amount] = exc

    return exchanges


def get_dirichlet_scale(amounts_exchanges, fit_variance, based_on_contributions, use_threshold):
    """Compute dirichlet scale for exchanges, where the Dirichlet parameter `alpha` is set to exchange amounts."""
    alphas = list(amounts_exchanges.keys())
    beta = sum(alphas)

    scaling_factors = []

    if based_on_contributions:
        selected_exchanges = select_contributing_exchanges(amounts_exchanges, use_threshold)
    else:
        selected_exchanges = select_higher_amount_exchanges(amounts_exchanges, use_threshold)

    for ialpha, iexc in selected_exchanges.items():
        loc = iexc['loc']
        scale = iexc['scale']
        if fit_variance:
            beta_variance = get_beta_variance(ialpha, beta-ialpha)
            lognormal_variance = get_lognormal_variance(loc, scale)
            scaling_factors.append(beta_variance / lognormal_variance * 2)
        else:
            beta_skewness = get_beta_skewness(ialpha, beta)
            lognormal_skewness = get_lognormal_skewness(scale)
            scaling_factors.append(beta_skewness / lognormal_skewness)

    scaling_factor = np.mean(scaling_factors)

    return scaling_factor


def get_dirichlet_scales(implicit_markets, fit_variance, based_on_contributions, use_threshold):
    """Get Diriechlet scales for all implicit markets.

    TODO
    This code is not good because it assumes that the file dirichlet_scales.pickle exists for validation steps.
    If this file does not exist, some scales might be equal to zero which is incorrect.

    """

    dirichlet_scales = {}
    for market, exchanges in implicit_markets.items():
        x = np.array([exc['amount'] for exc in exchanges])
        amounts = x.copy()
        amounts_exchanges_dict = {amounts[i]: exchanges[i] for i in range(len(amounts))}
        ds = get_dirichlet_scale(amounts_exchanges_dict, fit_variance, based_on_contributions, use_threshold)
        dirichlet_scales[market] = ds
    return dirichlet_scales


def predict_dirichlet_scales_generic_markets(generic_markets, fit_variance, based_on_contributions, use_threshold):
    """Predict Dirichlet scales for all generic markets from implicit ones."""
    fp_implicit_markets = DATA_DIR / "implicit-markets.pickle"
    if fp_implicit_markets.exists():
        implicit_markets = read_pickle(fp_implicit_markets)
    else:
        implicit_markets = find_markets("ecoinvent 3.8 cutoff", similar_fuzzy, True)
        write_pickle(implicit_markets, fp_implicit_markets)
    Xtrain = get_market_lmeans(implicit_markets)
    ytrain = get_dirichlet_scales(implicit_markets, fit_variance, based_on_contributions, use_threshold)
    Xtest = get_market_lmeans(generic_markets)
    reg = LinearRegression().fit(Xtrain, ytrain)
    ytest = Xtest * reg.coef_
    ytest = ytest.flatten()
    return ytest


def generate_markets_datapackage(
        similarity_func,
        get_dirichlet_scales_func,
        name,
        num_samples=SAMPLES,
        seed=42
):
    bd.projects.set_current("GSA for archetypes")

    if 'generic' in name:
        check_uncertainty = False
    else:
        check_uncertainty = True

    fp_markets = DATA_DIR / f"{name}.pickle"
    if fp_markets.exists():
        markets = read_pickle(fp_markets)
    else:
        markets = find_markets("ecoinvent 3.8 cutoff", similarity_func, check_uncertainty)
        write_pickle(markets, fp_markets)

    indices_array = np.array(
        [(exc.input.id, exc.output.id) for lst in markets.values() for exc in lst],
        dtype=bwp.INDICES_DTYPE,
    )
    mask = np.ones(len(indices_array), dtype=bool)
    dp = create_dynamic_datapackage(
        name, indices_array, mask, get_dirichlet_scales_func, num_samples, seed
    )

    dp.finalize_serialization()


def get_market_lmeans(markets):
    """Use large means as predictor fpr diricihlet scales."""
    lmeans = []
    for i, act in enumerate(markets.keys()):
        exchanges = markets[act]
        amounts = np.array([exc['amount'] for exc in exchanges])
        mean = np.mean(amounts)
        lmeans.append(np.mean(amounts[amounts >= mean]))
    X = 1/(np.array(lmeans))**3
    return X.reshape((-1, 1))


def check_dirichlet_samples(markets, indices, data_array):
    for market, exchanges in markets.items():
        col = market.id
        where = []
        for exc in exchanges:
            where.append(np.where(indices == np.array((exc.input.id, col), dtype=bwp.INDICES_DTYPE))[0][0])
        where = np.array(where)
        sum_ = data_array[where].sum(axis=0)
        assert np.allclose(min(sum_), max(sum_))


def create_dynamic_datapackage(name, indices, mask, get_dirichlet_scales_func, num_samples=SAMPLES, seed=42):

    markets = get_activities_from_indices(indices)

    selected_indices = indices[mask]
    selected_markets = get_activities_from_indices(selected_indices)
    dirichlet_scales = get_dirichlet_scales_func(
        selected_markets,
        fit_variance=True,
        based_on_contributions=True,
        use_threshold=False,
    )

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}-{seed}.zip"), write=True),
        name=name,
        seed=seed,
        sequential=True,
    )

    data = []
    flip = []

    # Dirichlet samples of different exchanges from one market should be generated with the same seed
    np.random.seed(seed)
    seeds = {market: np.random.randint(0, 2**32-1) for market in markets}

    for inds in indices:

        market = bd.get_activity(int(inds['col']))
        exchanges = markets[market]
        where_exc = [i for i in range(len(exchanges)) if exchanges[i].input.id == inds['row']][0]

        selected_exchanges = selected_markets.get(market, [])

        if len(selected_exchanges) > 1 and inds in selected_indices:

            total_amount = sum([exc['amount'] for exc in selected_exchanges])
            where_selected_exc = [
                i for i in range(len(selected_exchanges)) if selected_exchanges[i].input.id == inds['row']
            ][0]
            np.random.seed(seeds[market])
            samples = dirichlet.rvs(
                np.array([exc["amount"] for exc in selected_exchanges]) * dirichlet_scales[market],
                size=num_samples,
                ) * total_amount
            data.append(samples[:, where_selected_exc])
            flip.append(selected_exchanges[where_selected_exc]['type'] != "production")

        else:

            data.append(np.ones(num_samples) * exchanges[where_exc]['amount'])
            flip.append(exchanges[where_exc]["type"] != "production")

    data_array = np.vstack(data)
    flip_array = np.array(flip, dtype=bool)

    # Sanity check to ensure that samples in each market sum up to 1
    check_dirichlet_samples(markets, indices, data_array)

    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=data_array,
        # Resource group name that will show up in provenance
        name=name,
        indices_array=indices,
        flip_array=flip_array,
    )

    return dp


def create_validation_all_datapackage(name, dp_varying, mask, num_samples=SAMPLES, seed=42):

    indices = dp_varying.data[0]
    data = deepcopy(dp_varying.data[1])
    flip = dp_varying.data[2]

    dp_inverse = create_dynamic_datapackage(
        "validation.temp", indices, ~mask, get_dirichlet_scales, num_samples, seed,
    )
    data_inverse = dp_inverse.data[1]
    data[~mask] = data_inverse[~mask]

    dp_all = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}-{seed}.zip"), write=True),
        name=name,
        seed=seed,
        sequential=True,
    )
    dp_all.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=data,
        # Resource group name that will show up in provenance
        name=name,
        indices_array=indices,
        flip_array=flip,
    )

    return dp_all


def generate_validation_datapackages(indices, mask, num_samples, seed=42):

    dp_validation_inf = create_dynamic_datapackage(
        "validation.implicit-markets.influential", indices, mask, get_dirichlet_scales, num_samples, seed,
    )
    dp_validation_all = create_validation_all_datapackage(
        "validation.implicit-markets.all", dp_validation_inf, mask, num_samples, seed,
    )
    return dp_validation_all, dp_validation_inf


if __name__ == "__main__":

    # random_seeds = [85, 86]
    # num_samples = 15000
    # for random_seed in random_seeds:
    #     print(f"Random seed {random_seed}")
    #     generate_markets_datapackage(
    #         similar_fuzzy,
    #         get_dirichlet_scales,
    #         "implicit-markets",
    #         num_samples,
    #         random_seed,
    #     )

    im = bwp.load_datapackage(ZipFS(str(DATA_DIR / "implicit-markets-91.zip")))
    # im_data = im.get_resource('implicit-markets.data')[0]
    # im_indices = im.get_resource('implicit-markets.indices')[0]

    # print(im_data)

    # np.random.seed(42)
    # mask_random = np.random.choice([True, False], size=517, p=[0.1, 0.9])
    # mask_random = np.ones(517, dtype=bool)
    # dp_vall, dp_vinf = generate_validation_datapackages(im_indices, mask_random, num_samples=2000)

    # generate_markets_datapackage(
    #     similar_exact,
    #     predict_dirichlet_scales_generic_markets,
    #     "generic",
    #     SAMPLES,
    # )

    print("")
