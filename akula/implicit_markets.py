from collections import defaultdict
from pathlib import Path

import bw2data as bd
import bw_processing as bwp
import numpy as np
from fs.zipfs import ZipFS
from scipy.stats import dirichlet
from thefuzz import fuzz
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.resolve() / "data"
MINIMUM_DIRICHLET_SCALE = 50


def similar(a, b):
    return fuzz.partial_ratio(a, b) > 90 or fuzz.ratio(a, b) > 40


def find_uncertain_implicit_markets(database):
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
            elif exc["uncertainty type"] < 2:
                continue
            inpts[exc.input["reference product"]].append(exc)

        for key, lst in inpts.items():
            if (
                len(lst) > 1
                and similar(rp, key)
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
    return max(scaling_factor, MINIMUM_DIRICHLET_SCALE)


def get_dirichlet_scales(implicit_markets):
    dirichlet_scales = []
    for exchanges in implicit_markets.values():
        x = np.array([exc['amount'] for exc in exchanges])
        alpha = x.copy()
        alpha_exchanges_dict = {alpha[i]: exchanges[i] for i in range(len(alpha))}
        dirichlet_scales.append(get_dirichlet_scale(alpha_exchanges_dict))
    return dirichlet_scales


def generate_implicit_markets_datapackage(num_samples=25000):
    bd.projects.set_current("GSA for archetypes")

    im = find_uncertain_implicit_markets("ecoinvent 3.8 cutoff")
    dirichlet_scales = get_dirichlet_scales(im)

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / "implicit-markets.zip"), write=True),
        name="implicit markets",
        # set seed to have reproducible (though not sequential) sampling
        seed=42,
    )

    data_array = np.hstack(
        [
            dirichlet.rvs(
                np.array([exc["amount"] for exc in lst]) * dirichlet_scales[i],
                size=num_samples,
            )
            for i, lst in enumerate(im.values())
        ]
    ).T
    indices_array = np.array(
        [(exc.input.id, exc.output.id) for lst in im.values() for exc in lst],
        dtype=bwp.INDICES_DTYPE,
    )
    # All inputs -> all True
    flip_array = np.ones(len(indices_array), dtype=bool)
    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=data_array,
        # Resource group name that will show up in provenance
        name="implicit markets",
        indices_array=indices_array,
        flip_array=flip_array,
    )
    dp.finalize_serialization()


if __name__ == "__main__":
    generate_implicit_markets_datapackage(num_samples=2000)
