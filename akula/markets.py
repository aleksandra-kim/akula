from collections import defaultdict
from pathlib import Path
from copy import deepcopy

import bw2data as bd
import bw_processing as bwp
import numpy as np
import stats_arrays as sa
from fs.zipfs import ZipFS
from scipy.stats import dirichlet, lognorm
from thefuzz import fuzz
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import (
    read_pickle, write_pickle,
    update_fig_axes, COLOR_BRIGHT_PINK_RGB, COLOR_DARKGRAY_HEX, COLOR_PSI_LPURPLE, COLOR_PSI_DGREEN,
)

DATA_DIR = Path(__file__).parent.parent.resolve() / "data" / "datapackages"


def similar_fuzzy(a, b):
    """Fuzzy comparison between `a` and `b` strings using partial ratio."""
    return fuzz.partial_ratio(a, b) > 90 or fuzz.ratio(a, b) > 40


def find_markets(database, similarity_func):
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
            elif exc.get("uncertainty type", 0) < 2:
                continue
            elif exc.input.get("reference product", None) is None:
                continue
            inpts[exc.input["reference product"]].append(exc)

        for key, lst in inpts.items():
            if (
                len(lst) > 1
                and similarity_func(rp, key)
                and 0.98 <= sum([exc["amount"] for exc in lst]) <= 1.02
            ):
                found[act] = lst

        # print(rp, len(found))

    return found


def find_entsoe_markets(similarity_func):
    dp = bwp.load_datapackage(ZipFS(str(DATA_DIR / "entsoe-timeseries.zip")))

    indices = dp.get_resource('timeseries ENTSO electricity values.indices')[0]
    data = dp.get_resource('timeseries ENTSO electricity values.data')[0]

    # Fit lognormal distributions to ENTSO-E timeseries data
    distributions = fit_distributions(data, indices)
    distributions_dict = {tuple(i): d for i, d in zip(indices, distributions)}

    found = {}
    unique_cols = sorted(list(set(indices['col'])))
    for col in unique_cols:

        act = bd.get_activity(col)

        rp = act.get("reference product")
        if not rp:
            continue

        rows = sorted(indices[indices['col'] == col]['row'])

        inpts = defaultdict(list)
        for exc in act.technosphere():

            if exc.input.id in rows:

                exc_dict = deepcopy(exc.as_dict())
                params = distributions_dict[(exc.input.id, col)]
                distribution = {p[0].replace("_", " "): params[p[0]] for p in bwp.UNCERTAINTY_DTYPE}
                exc_dict.update(**distribution)

                if exc.input['name'] == "swiss residual electricity mix":
                    inpts["swiss residual electricity mix"].append(exc_dict)
                else:
                    inpts[exc.input["reference product"]].append(exc_dict)

        for key, lst in inpts.items():
            if (
                len(lst) > 1
                and similarity_func(rp, key)
                and 0.98 <= sum([exc["amount"] for exc in lst]) <= 1.02
            ):
                found[act] = lst

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


def select_contributing_exchanges(amounts_exchanges, return_scores=False):
    """Select exchanges in the given market that have contribution scores higher than average."""

    lca = setup_bw_project()

    scores = {}
    for amount, exc in amounts_exchanges.items():
        lca.redo_lci({exc.input.id: amount})
        lca.redo_lcia()
        scores[exc.input] = lca.score

    threshold = np.mean(list(scores.values()))

    exchanges = {}
    for amount, exc in amounts_exchanges.items():
        if scores[exc.input] >= threshold:
            if exc['uncertainty type'] != 2:
                print(exc['uncertainty type'])
            exchanges[amount] = exc
    if return_scores:
        return exchanges, scores
    else:
        return exchanges


def select_higher_amount_exchanges(amounts_exchanges):
    """Select exchanges in the given market that have amounts higher than average."""

    alphas = list(amounts_exchanges)
    threshold = np.mean(alphas)

    exchanges = {}

    for amount, exc in amounts_exchanges.items():
        if amount >= threshold:
            if exc['uncertainty type'] != 2:
                print(exc['uncertainty type'])
        exchanges[amount] = exc

    return exchanges


def get_dirichlet_scale(amounts_exchanges, fit_variance, based_on_contributions):
    """Compute dirichlet scale for exchanges, where the Dirichlet parameter `alpha` is set to exchange amounts."""
    alphas = list(amounts_exchanges)
    beta = sum(alphas)

    scaling_factors = []

    if based_on_contributions:
        selected_exchanges = select_contributing_exchanges(amounts_exchanges)
    else:
        selected_exchanges = select_higher_amount_exchanges(amounts_exchanges)

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


def get_dirichlet_scales(markets, fit_variance, based_on_contributions):
    """Get Dirichlet scales for all implicit markets.

    TODO
    This code is not good because it assumes that the file dirichlet_scales.pickle exists for validation steps.
    If this file does not exist, some scales might be equal to zero which is incorrect.

    """

    dirichlet_scales = {}

    for market, exchanges in markets.items():
        x = np.array([exc['amount'] for exc in exchanges])
        amounts = x.copy()
        amounts_exchanges_dict = {amounts[i]: exchanges[i] for i in range(len(amounts))}
        ds = get_dirichlet_scale(amounts_exchanges_dict, fit_variance, based_on_contributions)  # TODO scales are nan for some markets
        dirichlet_scales[market] = ds

    return dirichlet_scales


def predict_dirichlet_scales_generic_markets(generic_markets, fit_variance, based_on_contributions, use_threshold):
    """Predict Dirichlet scales for all generic markets from implicit ones."""
    fp_implicit_markets = DATA_DIR / "markets.pickle"
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


def generate_markets_datapackage(name, num_samples, seed=42, for_entsoe=False):
    fp_datapackage = DATA_DIR / f"{name}-{seed}-{num_samples}.zip"

    if not fp_datapackage.exists():

        fp_markets = DATA_DIR / f"{name}.pickle"
        if fp_markets.exists():
            markets = read_pickle(fp_markets)
        else:
            if for_entsoe:
                markets = find_entsoe_markets(similar_fuzzy)
            else:
                markets = find_markets("ecoinvent 3.8 cutoff", similar_fuzzy)
            write_pickle(markets, fp_markets)

        data, indices, flip = generate_market_samples(markets, num_samples, seed=seed)

        dp = bwp.create_datapackage(
            fs=ZipFS(str(fp_datapackage), write=True),
            name=name,
            seed=seed,
            sequential=True,
        )

        dp.add_persistent_array(
            matrix="technosphere_matrix",
            data_array=data,
            # Resource group name that will show up in provenance
            name=name,
            indices_array=indices,
            flip_array=flip,
        )

        dp.finalize_serialization()

    else:

        dp = bwp.load_datapackage(ZipFS(str(fp_datapackage)))

    return dp


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
            try:
                where.append(np.where(indices == np.array((exc.input.id, col), dtype=bwp.INDICES_DTYPE))[0][0])
            except AttributeError:
                where.append(np.where(indices == np.array((bd.get_activity(exc["input"]).id, col),
                                                          dtype=bwp.INDICES_DTYPE))[0][0])
        where = np.array(where)
        sum_ = data_array[where].sum(axis=0)
        assert np.allclose(min(sum_), max(sum_))


def generate_market_samples(markets, num_samples, seed=42):

    try:
        indices_array = np.array(
            [(exc.input.id, exc.output.id) for lst in markets.values() for exc in lst],
            dtype=bwp.INDICES_DTYPE,
        )
    except AttributeError:
        indices_array = np.array(
            [(bd.get_activity(exc["input"]).id, bd.get_activity(exc["output"]).id)
             for lst in markets.values() for exc in lst],
            dtype=bwp.INDICES_DTYPE,
        )

    dirichlet_scales = get_dirichlet_scales(
        markets,
        fit_variance=True,
        based_on_contributions=False,
    )

    data = []
    flip = []

    # Dirichlet samples of different exchanges from one market should be generated with the same seed
    np.random.seed(seed)
    seeds = {market: np.random.randint(0, 2**32-1) for market in markets}

    for inds in indices_array:

        market = bd.get_activity(int(inds['col']))
        exchanges = markets[market]
        try:
            where_exc = [i for i in range(len(exchanges)) if exchanges[i].input.id == inds['row']][0]
        except AttributeError:
            where_exc = [
                i for i in range(len(exchanges)) if bd.get_activity(exchanges[i]["input"]).id == inds['row']
            ][0]
        selected_exchanges = markets.get(market, [])

        if len(selected_exchanges) > 1 and inds in indices_array:

            total_amount = sum([exc['amount'] for exc in selected_exchanges])
            try:
                where_selected_exc = [
                    i for i in range(len(selected_exchanges)) if selected_exchanges[i].input.id == inds['row']
                ][0]
            except AttributeError:
                where_selected_exc = [
                    i for i in range(len(selected_exchanges))
                    if bd.get_activity(selected_exchanges[i]["input"]).id == inds['row']
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
    check_dirichlet_samples(markets, indices_array, data_array)

    return data_array, indices_array, flip_array


# def create_dynamic_datapackage(name, indices, mask, get_dirichlet_scales_func, num_samples, seed=42):
#
#     markets = get_activities_from_indices(indices)
#
#     selected_indices = indices[mask]
#     selected_markets = get_activities_from_indices(selected_indices)
#     dirichlet_scales = get_dirichlet_scales_func(
#         selected_markets,
#         fit_variance=True,
#         based_on_contributions=True,
#         use_threshold=False,
#     )
#
#     dp = bwp.create_datapackage(
#         fs=ZipFS(str(DATA_DIR / f"{name}-{seed}.zip"), write=True),
#         name=name,
#         seed=seed,
#         sequential=True,
#     )
#
#     data = []
#     flip = []
#
#     # Dirichlet samples of different exchanges from one market should be generated with the same seed
#     np.random.seed(seed)
#     seeds = {market: np.random.randint(0, 2**32-1) for market in markets}
#
#     for inds in indices:
#
#         market = bd.get_activity(int(inds['col']))
#         exchanges = markets[market]
#         where_exc = [i for i in range(len(exchanges)) if exchanges[i].input.id == inds['row']][0]
#
#         selected_exchanges = selected_markets.get(market, [])
#
#         if len(selected_exchanges) > 1 and inds in selected_indices:
#
#             total_amount = sum([exc['amount'] for exc in selected_exchanges])
#             where_selected_exc = [
#                 i for i in range(len(selected_exchanges)) if selected_exchanges[i].input.id == inds['row']
#             ][0]
#             np.random.seed(seeds[market])
#             samples = dirichlet.rvs(
#                 np.array([exc["amount"] for exc in selected_exchanges]) * dirichlet_scales[market],
#                 size=num_samples,
#                 ) * total_amount
#             data.append(samples[:, where_selected_exc])
#             flip.append(selected_exchanges[where_selected_exc]['type'] != "production")
#
#         else:
#
#             data.append(np.ones(num_samples) * exchanges[where_exc]['amount'])
#             flip.append(exchanges[where_exc]["type"] != "production")
#
#     data_array = np.vstack(data)
#     flip_array = np.array(flip, dtype=bool)
#
#     # Sanity check to ensure that samples in each market sum up to 1
#     check_dirichlet_samples(markets, indices, data_array)
#
#     dp.add_persistent_array(
#         matrix="technosphere_matrix",
#         data_array=data_array,
#         # Resource group name that will show up in provenance
#         name=name,
#         indices_array=indices,
#         flip_array=flip_array,
#     )
#
#     return dp


def create_validation_all_datapackage(name, dp_varying, mask, num_samples, seed=42):

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


# if __name__ == "__main__":

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

    # im = bwp.load_datapackage(ZipFS(str(DATA_DIR / "implicit-markets-91.zip")))
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

    # print("")


# def get_scaling_based_on_variance(alpha, beta, variance):
#     """Derive scaling factor by fitting variance of the given distribution to the variance of the Beta distribution.
#
#     Alpha and beta are parameters of the Beta distribution, variance is the variance of the given distribution.
#     """
#     factor = (alpha * beta / (alpha + beta)**2 / variance - 1) / (alpha + beta)
#     return factor
#
#
# def get_dirichlet_factor(distributions):
#
#     alphas = distributions["mean"]
#
#     # Since it is impossible to choose one Dirichlet factor that fits beta distributions to all the exchanges perfectly,
#     # we use the mean value of exchange amounts as a threshold to decide which exchange distributions to optimize.
#     threshold = np.mean(alphas)
#
#     factors = []
#     for i, alpha in enumerate(alphas):
#         _, variance = distributions[i]["mean"], distributions[i]["variance"]
#         beta = sum(alphas) - alpha
#         if alpha >= threshold:  # alphas are equals to means
#             factor = get_scaling_based_on_variance(alpha, beta, variance)
#             factors.append(factor)
#
#     return np.mean(factors)
#
#
# def get_dirichlet_samples(indices, distributions, iterations):
#     means = distributions["mean"]
#
#     data = np.zeros((len(indices), iterations))
#     data[:] = np.nan
#
#     unique_cols = sorted(list(set(indices['col'])))
#     for i, col in enumerate(unique_cols):
#         mask = indices['col'] == col
#
#         mask0 = mask & (means == 0)
#         data[mask0, :] = np.zeros((sum(mask0), iterations))
#
#         mask1 = mask & (means == 1)
#         data[mask1, :] = np.ones((sum(mask1), iterations))
#
#         mask_not_01 = mask & ~mask0 & ~mask1
#
#         if sum(mask_not_01):
#             factor = get_dirichlet_factor(distributions[mask_not_01])
#             samples = dirichlet.rvs(
#                 means[mask_not_01] * abs(factor),
#                 size=iterations,
#             ) * 1
#             data[mask_not_01, :] = samples.T
#
#     return data


# def get_activities_from_indices(indices):
#
#     activities = {}
#
#     if indices is not None:
#
#         cols = sorted(set(indices['col']))
#         for col in cols:
#
#             rows = sorted(indices[indices['col'] == col]['row'])
#             act = bd.get_activity(int(col))
#
#             exchanges = []
#             for exc in act.exchanges():
#                 if exc.input.id in rows:
#                     exchanges.append(exc)
#
#             if len(exchanges) > 0:
#                 activities[act] = exchanges
#
#     return activities


def get_distributions(indices):

    ddict = dict()
    assert bwp.UNCERTAINTY_DTYPE[-1][0] == "negative"

    cols = sorted(set(indices['col']))
    for col in cols:
        rows = sorted(indices[indices['col'] == col]['row'])
        act = bd.get_activity(int(col))
        for exc in act.exchanges():
            if exc.input.id in rows:
                ddict[(exc.input.id, col)] = (
                    [exc.get(p[0].replace("_", " "), np.nan) for p in bwp.UNCERTAINTY_DTYPE[:-1]]
                    + [exc["type"] == "production"])

    dlist = [tuple(ddict[row, col]) for row, col in indices]
    distributions = np.array(dlist, dtype=bwp.UNCERTAINTY_DTYPE)

    return distributions


def plot_dirichlet_samples(act_id, dp_markets, num_bins=100):

    data = dp_markets.get_resource('markets.data')[0]
    indices = dp_markets.get_resource('markets.indices')[0]
    mask = indices['col'] == act_id
    data = data[mask, :]
    indices = indices[mask]

    num_exchanges = len(indices)

    distributions = get_distributions(indices)
    assert np.all(distributions["uncertainty_type"] == sa.LognormalUncertainty.id)

    fig = make_subplots(rows=num_exchanges, cols=1, subplot_titles=["placeholder"]*num_exchanges)
    opacity = 0.65
    showlegend = True

    for i in range(num_exchanges):

        row = indices[i]['row']
        act = bd.get_activity(row)
        fig.layout.annotations[i]['text'] = f"{act['name'][:60]} -- {act['location']}"

        loc = distributions[i]['loc']
        scale = distributions[i]['scale']
        min_distr = lognorm.ppf(0.001, s=scale, scale=np.exp(loc))
        max_distr = lognorm.ppf(0.999, s=scale, scale=np.exp(loc))

        Y = data[i, :]
        min_samples = min(Y)
        max_samples = max(Y)

        bin_min = min(min_distr, min_samples)
        bin_max = max(max_distr, max_samples)

        bins_ = np.linspace(bin_min, bin_max, num_bins + 1, endpoint=True)
        Y_samples, _ = np.histogram(Y, bins=bins_, density=True)

        midbins = (bins_[1:] + bins_[:-1]) / 2
        Y_distr = lognorm.pdf(midbins, s=scale, scale=np.exp(loc))

        # Plot Dirichlet samples
        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_samples,
                name=r"$\text{Dirichlet samples}$",
                showlegend=showlegend,
                opacity=opacity,
                line=dict(color=COLOR_DARKGRAY_HEX, width=1, shape="hvh"),
                fill="tozeroy",
            ),
            row=i+1,
            col=1,
        )
        # Plot lognormal distribution
        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_distr,
                line=dict(color=COLOR_PSI_LPURPLE),
                name=r"$\text{Defined lognormal}$",
                showlegend=showlegend,
                legendrank=1,
            ),
            row=i+1,
            col=1,
        )
        showlegend = False
    #
    fig.update_xaxes(title_text=r"$\text{Production volume share}$")
    fig.update_yaxes(title_text=r"$\text{Frequency}$")
    fig = update_fig_axes(fig)
    if num_exchanges < 10:
        offset = 0.1
    else:
        offset = 0.005
    fig.update_layout(
        width=600, height=180*num_exchanges + 40,
        legend=dict(yanchor="bottom", y=1 + offset, xanchor="center", x=0.5,
                    orientation='h', font=dict(size=13)),
        margin=dict(t=10, b=10, l=10, r=10),
    )

    return fig


def fit_distributions(data, indices):

    dlist = []
    assert bwp.UNCERTAINTY_DTYPE[0][0] == "uncertainty_type"
    assert bwp.UNCERTAINTY_DTYPE[-1][0] == "negative"

    for i, d in enumerate(data):
        dpos = d[d > 0]
        if len(dpos):
            shape, loc, scale = lognorm.fit(dpos, floc=0)
            params = dict(uncertainty_type=sa.LognormalUncertainty.id, loc=np.log(scale), scale=shape)
        else:
            params = dict(uncertainty_type=sa.NoUncertainty.id)
        distribution = ([params.get(p[0], np.nan) for p in bwp.UNCERTAINTY_DTYPE[:-1]]
                        + [indices[i]["row"] == indices[i]["col"]])
        dlist.append(tuple(distribution))

    distributions = np.array(dlist, dtype=bwp.UNCERTAINTY_DTYPE)

    return distributions


def plot_dirichlet_entsoe_samples(act_id, dp_entsoe, dp_dirichlet, num_bins=100):

    # Extract ENTSO-E data for the given activity
    indices = dp_entsoe.get_resource('entsoe.indices')[0]
    mask = indices['col'] == act_id
    num_exchanges = sum(mask)
    data_entsoe = dp_entsoe.get_resource('entsoe.data')[0]
    data_entsoe = data_entsoe[mask, :]

    # Fit lognormal distributions to ENTSO-E data
    distributions_lognorm = fit_distributions(data_entsoe, indices)
    assert np.all(distributions_lognorm["uncertainty_type"] == 2)

    # Get Dirichlet distributions samples that were fit to ENTSO-E data
    indices = dp_dirichlet.get_resource('entsoe-dirichlet.indices')[0]
    mask = indices['col'] == act_id
    data_dirichlet = dp_dirichlet.get_resource('entsoe-dirichlet.data')[0]
    data_dirichlet = data_dirichlet[mask, :]
    # assert np.all(distributions_dirichlet["uncertainty_type"] == sa.LognormalUncertainty.id)

    fig = make_subplots(rows=num_exchanges, cols=1, subplot_titles=["placeholder"]*num_exchanges)
    opacity = 0.65
    showlegend = True

    for i in range(num_exchanges):

        row = indices[i]['row']
        act = bd.get_activity(row)

        fig.layout.annotations[i]['text'] = f"{act['name'][:60]} -- {act['location']}"

        Y_entsoe = data_entsoe[i, :]
        min_entsoe = min(Y_entsoe)
        max_entsoe = max(Y_entsoe)

        loc = distributions_lognorm[i]['loc']
        scale = distributions_lognorm[i]['scale']
        min_lognorm = lognorm.ppf(0.001, s=scale, scale=np.exp(loc))
        max_lognorm = lognorm.ppf(0.999, s=scale, scale=np.exp(loc))

        Y_dirichlet = data_dirichlet[i, :]
        min_dirichlet = min(Y_dirichlet)
        max_dirichlet = max(Y_dirichlet)

        bin_min = min(min_lognorm, min_entsoe, min_dirichlet)
        bin_max = max(max_lognorm, max_entsoe, max_dirichlet)

        bins_ = np.linspace(bin_min, bin_max, num_bins + 1, endpoint=True)
        midbins = (bins_[1:] + bins_[:-1]) / 2
        Y_entsoe, _ = np.histogram(Y_entsoe, bins=bins_, density=True)
        Y_lognorm = lognorm.pdf(midbins, s=scale, scale=np.exp(loc))
        Y_dirichlet, _ = np.histogram(Y_dirichlet, bins=bins_, density=True)

        # Plot ENTSO-E samples
        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_entsoe,
                name=r"$\text{ENTSO-E samples}$",
                showlegend=showlegend,
                opacity=opacity,
                line=dict(color=COLOR_DARKGRAY_HEX, width=1, shape="hvh"),
                fill="tozeroy",
            ),
            row=i+1,
            col=1,
        )
        # Plot fitted lognormal distribution
        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_lognorm,
                line=dict(color="red"),
                name=r"$\text{Fitted lognormal}$",
                showlegend=showlegend,
                legendrank=1,
            ),
            row=i+1,
            col=1,
        )
        # Plot Dirichlet samples
        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_dirichlet,
                name=r"$\text{Dirichlet samples}$",
                showlegend=showlegend,
                opacity=opacity,
                line=dict(color=COLOR_PSI_DGREEN, width=1, shape="hvh"),
                fill="tozeroy",
            ),
            row=i+1,
            col=1,
        )

        showlegend = False

    fig.update_xaxes(title_text=r"$\text{Production volume share}$")
    fig.update_yaxes(title_text=r"$\text{Frequency}$")
    fig = update_fig_axes(fig)
    if num_exchanges < 10:
        offset = 0.1
    else:
        offset = 0.005
    fig.update_layout(
        width=600, height=180*num_exchanges + 40,
        legend=dict(yanchor="bottom", y=1 + offset, xanchor="center", x=0.5,
                    orientation='h', font=dict(size=13)),
        margin=dict(t=10, b=10, l=10, r=10),
    )

    return fig
