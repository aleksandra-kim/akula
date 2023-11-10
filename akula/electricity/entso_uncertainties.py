import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from fs.zipfs import ZipFS
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy.stats import dirichlet

# Local files
from ..constants import *

DAYTIME_MASK = np.hstack([
    np.zeros(DAYTIME_START_AM, dtype=bool),
    np.ones(HOURS_IN_DAYTIME, dtype=bool),
    np.zeros(HOURS_IN_DAY - DAYTIME_START_AM - HOURS_IN_DAYTIME, dtype=bool)
])
DAYTIME_MASK_YEAR = np.hstack([
    np.tile(DAYTIME_MASK, DAYS_IN_YEAR - 1),
    DAYTIME_MASK[:-1]  # 31st of December is only 23 hours long in ENTSOE
])
DAYTIME_MASK_2020 = np.hstack([
    np.tile(DAYTIME_MASK, DAYS_IN_2020 - 1),
    DAYTIME_MASK[:-1]  # 31st of December is only 23 hours long in ENTSOE
])


def get_one_activity(db_name, **kwargs):
    possibles = [
        act
        for act in bd.Database(db_name)
        if all(act.get(key) == value for key, value in kwargs.items())
    ]
    if len(possibles) == 1:
        return possibles[0]
    else:
        raise ValueError(
            f"Couldn't get exactly one activity in database `{db_name}` for arguments {kwargs}"
        )


def get_winter_data(data):
    # Select hours in spring 2019, 2020 and 2021
    data_winter = np.hstack([
        data[:, JAN_START_2019: SPRING_START_2019],
        data[:,    DEC_START_2019: SPRING_START_2020],
        data[:,    DEC_START_2020: SPRING_START_2021],
        data[:,    DEC_START_2021:],
    ])
    # `+3` comes from the fact that each December 31st is only 23 hours long in ENTSOE,
    # `DAYS_IN_WINTER + 1` because 2020 is a leap year
    assert data_winter.shape[1] + 3 == DAYS_IN_WINTER * HOURS_IN_DAY * (YEARS - 1) + (DAYS_IN_WINTER + 1) * HOURS_IN_DAY
    return data_winter


def get_spring_data(data):
    # Select hours in spring 2019, 2020 and 2021
    data_spring = np.hstack([
        data[:, SPRING_START_2019: SUMMER_START_2019],
        data[:, SPRING_START_2020: SUMMER_START_2020],
        data[:, SPRING_START_2021: SUMMER_START_2021],
    ])
    assert data_spring.shape[1] == DAYS_IN_SPRING * HOURS_IN_DAY * YEARS
    return data_spring


def get_summer_data(data):
    # Select hours in spring 2019, 2020 and 2021
    data_summer = np.hstack([
        data[:, SUMMER_START_2019: AUTUMN_START_2019],
        data[:, SUMMER_START_2020: AUTUMN_START_2020],
        data[:, SUMMER_START_2021: AUTUMN_START_2021],
    ])
    assert data_summer.shape[1] == DAYS_IN_SUMMER * HOURS_IN_DAY * YEARS
    return data_summer


def get_autumn_data(data):
    # Select hours in spring 2019, 2020 and 2021
    data_autumn = np.hstack([
        data[:, AUTUMN_START_2019: DEC_START_2019],
        data[:, AUTUMN_START_2020: DEC_START_2020],
        data[:, AUTUMN_START_2021: DEC_START_2021],
    ])
    assert data_autumn.shape[1] == DAYS_IN_AUTUMN * HOURS_IN_DAY * YEARS
    return data_autumn


def get_daytime_data(data):
    data_daytime = np.hstack([
        data[:, JAN_START_2019: JAN_START_2020][:, DAYTIME_MASK_YEAR],
        data[:, JAN_START_2020: JAN_START_2021][:, DAYTIME_MASK_2020],
        data[:, JAN_START_2021:][:, DAYTIME_MASK_YEAR],
    ])
    # `+1 * HOURS_IN_DAY` comes from the additional day in 2020
    assert (data_daytime.shape[1] == int(DAYS_IN_YEAR * HOURS_IN_DAY / HOURS_IN_DAY * HOURS_IN_DAYTIME * YEARS
                                         + 1 * HOURS_IN_DAY / HOURS_IN_DAY * HOURS_IN_DAYTIME))
    return data_daytime


def get_nighttime_data(data):
    data_nighttime = np.hstack([
        data[:, JAN_START_2019: JAN_START_2020][:, ~DAYTIME_MASK_YEAR],
        data[:, JAN_START_2020: JAN_START_2021][:, ~DAYTIME_MASK_2020],
        data[:, JAN_START_2021:][:, ~DAYTIME_MASK_YEAR],
    ])
    # `+1 * HOURS_IN_DAY` comes from the additional day in 2020
    # `+3` comes from the fact that each December 31st is only 23 hours long in ENTSOE
    assert (data_nighttime.shape[1] + 3 == int(DAYS_IN_YEAR * HOURS_IN_DAY / HOURS_IN_DAY * HOURS_IN_NIGHTTIME * YEARS
                                               + 1 * HOURS_IN_DAY / HOURS_IN_DAY * HOURS_IN_NIGHTTIME))
    return data_nighttime


def get_average_mixes(data, indices):

    data_2019 = data[:, JAN_START_2019: JAN_START_2020]
    data_2020 = data[:, JAN_START_2020: JAN_START_2021]
    data_2021 = data[:, JAN_START_2021:]

    average_2019 = np.mean(data_2019, axis=1)
    average_2020 = np.mean(data_2020, axis=1)
    average_2021 = np.mean(data_2021, axis=1)

    average_mix = np.zeros(shape=(data_2019.shape[0], 3))
    average_mix[:] = np.nan

    unique_cols = sorted(list(set(indices['col'])))
    for col in unique_cols:
        mask = indices['col'] == col
        data = np.vstack([average_2019[mask], average_2020[mask], average_2021[mask]]).T
        average_mix[mask, :] = data

    return average_mix


def fit_pedigree_distributions(average_mix):

    mean_lognormal = np.mean(average_mix, axis=1)

    # Expert judgement
    indicator_reliability = 1
    indicator_completeness = 2
    indicator_temporal_correlation = 2
    indicator_geographical_correlation = 1
    indicator_technical_correlation = 1
    indicator_sample_size = 5

    sigma_reliability = PEDIGREE_MATRIX["reliability"][indicator_reliability]
    sigma_completeness = PEDIGREE_MATRIX["completeness"][indicator_completeness]
    sigma_temporal_correlation = PEDIGREE_MATRIX["temporal_correlation"][indicator_temporal_correlation]
    sigma_geographical_correlation = PEDIGREE_MATRIX["geographical_correlation"][indicator_geographical_correlation]
    sigma_technical_correlation = PEDIGREE_MATRIX["technical_correlation"][indicator_technical_correlation]
    sigma_sample_size = PEDIGREE_MATRIX["sample_size"][indicator_sample_size]

    # Compute sigma of the underlying normal distribution
    variance_normal = sigma_reliability**2 + sigma_completeness**2 + sigma_temporal_correlation**2 + \
                      sigma_geographical_correlation**2 + sigma_technical_correlation**2 + sigma_sample_size**2
    variance_normal = np.ones(len(average_mix)) * variance_normal

    # Compute mean and sigma of lognormal distribution
    mean_normal = np.log(mean_lognormal) - variance_normal / 2
    variance_lognormal = np.sqrt((np.exp(variance_normal) - 1) * np.exp(2 * mean_normal + variance_normal))

    distributions = np.array(
        list(zip(mean_lognormal, variance_lognormal)), dtype=[('mean', 'float64'), ('variance', 'float64')]
    )

    return distributions


def fit_distributions(average_mix):
    means = np.mean(average_mix, axis=1)
    variances = np.var(average_mix, axis=1)
    distributions = np.array(
        list(zip(means, variances)), dtype=[('mean', 'float64'), ('variance', 'float64')]
    )
    return distributions


def get_scaling_based_on_variance(alpha, beta, variance):
    """Derive scaling factor by fitting variance of the given distribution to the variance of the Beta distribution.

    Alpha and beta are parameters of the Beta distribution, variance is the variance of the given distribution.
    """
    factor = (alpha * beta / (alpha + beta)**2 / variance - 1) / (alpha + beta)
    return factor


def get_dirichlet_factor(distributions):

    alphas = distributions["mean"]

    # Since it is impossible to choose one Dirichlet factor that fits beta distributions to all the exchanges perfectly,
    # we use the mean value of exchange amounts as a threshold to decide which exchange distributions to optimize.
    threshold = np.mean(alphas)

    factors = []
    for i, alpha in enumerate(alphas):
        _, variance = distributions[i]["mean"], distributions[i]["variance"]
        beta = sum(alphas) - alpha
        if alpha >= threshold:  # alphas are equals to means
            factor = get_scaling_based_on_variance(alpha, beta, variance)
            factors.append(factor)

    return np.mean(factors)


def get_dirichlet_samples(indices, distributions, iterations):
    means = distributions["mean"]

    data = np.zeros((len(indices), iterations))
    data[:] = np.nan

    unique_cols = sorted(list(set(indices['col'])))
    for i, col in enumerate(unique_cols):
        mask = indices['col'] == col

        mask0 = mask & (means == 0)
        data[mask0, :] = np.zeros((sum(mask0), iterations))

        mask1 = mask & (means == 1)
        data[mask1, :] = np.ones((sum(mask1), iterations))

        mask_not_01 = mask & ~mask0 & ~mask1

        if sum(mask_not_01):
            factor = get_dirichlet_factor(distributions[mask_not_01])
            samples = dirichlet.rvs(
                means[mask_not_01] * abs(factor),
                size=iterations,
            ) * 1
            data[mask_not_01, :] = samples.T

    return data


def plot_data_fitted(data_fitted, average_mix, indices):

    unique_cols = sorted(list(set(indices['col'])))

    for col in unique_cols:
        mask = indices['col'] == col

        rows = indices[mask]['row']
        fig = make_subplots(rows=len(rows), cols=1)

        for i, row in enumerate(rows):
            averages = average_mix[mask][i, :]
            data = data_fitted[mask][i, :]
            fig.add_trace(go.Scatter(
                x=averages, y=np.zeros(len(rows)), name=f"{i} averages", mode="markers"
            ), row=i+1, col=1)
            fig.add_trace(go.Histogram(
                x=data, histnorm="probability", name=f"{i} samples", nbinsx=100
            ), row=i+1, col=1)
        fig.update_layout(width=800, height=200*len(rows))

        fig.show()


def get_fitted_data(data, indices, iterations):
    average_mix = get_average_mixes(data, indices)
    distributions = fit_distributions(average_mix)
    data_fitted = get_dirichlet_samples(indices, distributions, iterations)
    # plot_data_fitted(data_fitted, average_mix, indices)
    return data_fitted


def create_entsoe_dp(project_dir, option, iterations):
    """
    Possible options are: winter, spring, summer, autumn, daytime, nighttime, fitted.

    Random seed will be ignored in the bwp.create_datapackage() if sequential=True.
    But when running MC simulations, if bc.LCA sets seed_override to a random seed,
    then the data will no longer be taken sequentially from this datapackage.

    """
    dp = bwp.load_datapackage(ZipFS(str(project_dir / "akula" / "data" / "entso-timeseries.zip")))

    data = dp.get_resource("timeseries ENTSO electricity values.data")[0]
    indices = dp.get_resource("timeseries ENTSO electricity values.indices")[0]
    flip = dp.get_resource("timeseries ENTSO electricity values.flip")[0]

    if option == "winter":
        data_new = get_winter_data(data)
    elif option == "spring":
        data_new = get_spring_data(data)
    elif option == "summer":
        data_new = get_summer_data(data)
    elif option == "autumn":
        data_new = get_autumn_data(data)
    elif option == "daytime":
        data_new = get_daytime_data(data)
    elif option == "nighttime":
        data_new = get_nighttime_data(data)
    elif option == "fitted":
        data_new = get_fitted_data(data, indices, iterations)
    else:
        print("Selecting complete ENTSO-E timeseries.")
        data_new = data

    dp_new = bwp.create_datapackage(sequential=True, seed=None)
    dp_new.add_persistent_array(
        matrix='technosphere_matrix',
        indices_array=indices,
        data_array=data_new,
        flip_array=flip,
    )
    return dp_new


def compute_lcia(project, dp, iterations=1000, seed=None):
    """
    If we set seed_override to a random seed, and use_distributions=True, then we can reproduce LCIA scores,
    but it is not clear in which order the data is taken from the sequential datapackage `dp`,
    because seed_override controls randomness of both drawing samples from random distributions,
    and taking samples from the data array defined in the datapackage `dp`.
    If we set seed_override to None, then samples are drawn randomly from distributions, but the order of samples
    in the sequential datapackage is predefined and follows the data array of `dp`.
    """

    bd.projects.set_current(project)
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    activity = get_one_activity("ecoinvent 3.8 cutoff", name="market for electricity, low voltage", location="CH")

    fu, data_objs, _ = bd.prepare_lca_inputs({activity: 1}, method=method, remapping=False)

    if dp is not None:
        data_objs += [dp]

    lca = bc.LCA(
        demand=fu,
        data_objs=data_objs,
        use_arrays=True,
        use_distributions=True,
        seed_override=seed,
    )
    lca.lci()
    lca.lcia()

    scores = [lca.score for _ in zip(range(iterations), lca)]

    return scores


def plot_lcia_scores(data, labels):

    fig = ff.create_distplot(
        hist_data=list(data.values()),
        group_labels=labels,
        bin_size=.005,
    )
    fig.update_layout(width=1000, height=800, title_text="LCIA scores")

    return fig


def plot_electricity_profile(project, project_dir):
    bd.projects.set_current(project)
    activity = get_one_activity("ecoinvent 3.8 cutoff", name="market for electricity, low voltage", location="CH")
    dp = bwp.load_datapackage(ZipFS(str(project_dir / "akula" / "data" / "entso-timeseries.zip")))

    data = dp.get_resource("timeseries ENTSO electricity values.data")[0]
    indices = dp.get_resource("timeseries ENTSO electricity values.indices")[0]

    mask = indices['col'] == activity.id
    # data_2021 = data[:, JAN_START_2021:JAN_START_2021+DAYS_IN_JAN*HOURS_IN_DAY]
    data_2021 = data[:, JAN_START_2021:]
    data_2021_one_country = data_2021[mask, :]
    indices_one_country = indices[mask]
    names = [bd.get_activity(ind)["name"] for ind in indices_one_country["row"]]
    argsort = np.argsort(names)

    names = np.array(names)[argsort]
    data_2021_one_country = data_2021_one_country[argsort, :]

    fig = go.Figure()

    for i, d in enumerate(data_2021_one_country):
        fig.add_trace(go.Scatter(
            x=np.arange(len(d)),
            y=d,
            name=names[i],
            stackgroup='one'
        ))

    fig.update_layout(width=2500, height=700,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))

    return fig
