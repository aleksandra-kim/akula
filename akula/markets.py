from collections import defaultdict
from pathlib import Path

import bw2data as bd
import bw_processing as bwp
import numpy as np
from fs.zipfs import ZipFS
from scipy.stats import dirichlet
from thefuzz import fuzz
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from gsa_framework.utils import read_pickle, write_pickle

DATA_DIR = Path(__file__).parent.resolve() / "data"


def similar_im(a, b):
    return fuzz.partial_ratio(a, b) > 90 or fuzz.ratio(a, b) > 40


def similar_gm(a, b):
    return a == b


def find_markets(database, similarity_func, check_uncertainty):

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
            elif check_uncertainty and exc["uncertainty type"] < 2:
                continue
            inpts[exc.input["reference product"]].append(exc)

        for key, lst in inpts.items():
            if (
                len(lst) > 1
                and similarity_func(rp, key)
                and 0.98 <= sum([exc["amount"] for exc in lst]) <= 1.02
            ):
                found[act] = lst

    return found


def get_beta_variance(a, b):
    return a*b/(a+b)**2/(a+b+1)


def get_beta_skewness(a, b):
    return 2*(b-a)*((a+b+1)**0.5) / (a+b+2) / (a*b)**0.5


def get_lognormal_variance(loc, scale):
    return (np.exp(scale**2)-1) * np.exp(2*loc+scale**2)


def get_lognormal_skewness(scale):
    return (np.exp(scale**2)+2) * ((np.exp(scale**2)-1)**0.5)


def get_dirichlet_scale(alpha_exchanges, fit_variance=True):
    alphas = list(alpha_exchanges.keys())
    beta = sum(alphas)
    alpha_threshold = np.mean(alphas)
    scaling_factors = []
    for ialpha, iexc in alpha_exchanges.items():
        if ialpha >= alpha_threshold:
            assert iexc['uncertainty type'] == 2
            loc = iexc['loc']
            scale = iexc['scale']
            if fit_variance:
                beta_variance = get_beta_variance(ialpha, beta)
                lognormal_variance = get_lognormal_variance(loc, scale)
                scaling_factors.append(beta_variance / lognormal_variance * 2)
            else:
                beta_skewness = get_beta_skewness(ialpha, beta)
                lognormal_skewness = get_lognormal_skewness(scale)
                scaling_factors.append(beta_skewness / lognormal_skewness)
    scaling_factor = np.mean(scaling_factors)
    return scaling_factor


def get_dirichlet_scales(implicit_markets):
    dirichlet_scales = []
    for exchanges in implicit_markets.values():
        x = np.array([exc['amount'] for exc in exchanges])
        alpha = x.copy()
        alpha_exchanges_dict = {alpha[i]: exchanges[i] for i in range(len(alpha))}
        dirichlet_scales.append(get_dirichlet_scale(alpha_exchanges_dict))
    return dirichlet_scales


def predict_dirichlet_scales_generic_markets(generic_markets):
    """Get dirichlet scores for generic markets from implicit ones."""
    fp_implicit_markets = DATA_DIR / "implicit_markets.pickle"
    if fp_implicit_markets.exists():
        implicit_markets = read_pickle(fp_implicit_markets)
    else:
        implicit_markets = find_markets("ecoinvent 3.8 cutoff", similar_im, True)
        write_pickle(implicit_markets, fp_implicit_markets)
    Xtrain = get_market_lmeans(implicit_markets)
    ytrain = get_dirichlet_scales(implicit_markets)
    Xtest = get_market_lmeans(generic_markets)
    reg = LinearRegression().fit(Xtrain, ytrain)
    ytest = Xtest * reg.coef_
    ytest = ytest.flatten()
    return ytest


def generate_markets_datapackage(
        similarity_func,
        get_dirichlet_scales_func,
        markets_type,
        num_samples=25000
):
    bd.projects.set_current("GSA for archetypes")

    fp_markets = DATA_DIR / f"{markets_type}_markets.pickle"
    if fp_markets.exists():
        markets = read_pickle(fp_markets)
    else:
        markets = find_markets("ecoinvent 3.8 cutoff", similarity_func, False)
        write_pickle(markets, fp_markets)
    dirichlet_scales = get_dirichlet_scales_func(markets)

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{markets_type}-markets.zip"), write=True),
        name=f"{markets_type} markets",
        # set seed to have reproducible (though not sequential) sampling
        seed=42,
    )

    data_array = np.hstack(
        [
            dirichlet.rvs(
                np.array([exc["amount"] for exc in lst]) * dirichlet_scales[i],
                size=num_samples,
            )
            for i, lst in enumerate(markets.values())
        ]
    ).T
    indices_array = np.array(
        [(exc.input.id, exc.output.id) for lst in markets.values() for exc in lst],
        dtype=bwp.INDICES_DTYPE,
    )
    # All inputs -> all True
    flip_array = np.ones(len(indices_array), dtype=bool)
    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=data_array,
        # Resource group name that will show up in provenance
        name=f"{markets_type} markets",
        indices_array=indices_array,
        flip_array=flip_array,
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


if __name__ == "__main__":
    num_samples = 25000
    generate_markets_datapackage(
        similar_im,
        get_dirichlet_scales,
        "implicit",
        num_samples=num_samples,
    )
    generate_markets_datapackage(
        similar_gm,
        predict_dirichlet_scales_generic_markets,
        "generic",
        num_samples=num_samples,
    )
