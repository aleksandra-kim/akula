import numpy as np
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
import stats_arrays as sa
from fs.zipfs import ZipFS
import plotly.graph_objects as go


# Local files
from ..constants import *
from ..utils import (update_fig_axes,
                     COLOR_DARKGRAY_HEX, COLOR_PSI_LPURPLE, COLOR_PSI_LPURPLE_OPAQUE, COLOR_DARKGRAY_HEX_OPAQUE)

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
    """Select time-series electricity generation data in winter 2019, 2020 and 2021."""
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
    """Select time-series electricity generation data in spring 2019, 2020 and 2021."""
    data_spring = np.hstack([
        data[:, SPRING_START_2019: SUMMER_START_2019],
        data[:, SPRING_START_2020: SUMMER_START_2020],
        data[:, SPRING_START_2021: SUMMER_START_2021],
    ])
    assert data_spring.shape[1] == DAYS_IN_SPRING * HOURS_IN_DAY * YEARS
    return data_spring


def get_summer_data(data):
    """Select time-series electricity generation data in summer 2019, 2020 and 2021."""
    data_summer = np.hstack([
        data[:, SUMMER_START_2019: AUTUMN_START_2019],
        data[:, SUMMER_START_2020: AUTUMN_START_2020],
        data[:, SUMMER_START_2021: AUTUMN_START_2021],
    ])
    assert data_summer.shape[1] == DAYS_IN_SUMMER * HOURS_IN_DAY * YEARS
    return data_summer


def get_autumn_data(data):
    """Select time-series electricity generation data in autumn 2019, 2020 and 2021."""
    data_autumn = np.hstack([
        data[:, AUTUMN_START_2019: DEC_START_2019],
        data[:, AUTUMN_START_2020: DEC_START_2020],
        data[:, AUTUMN_START_2021: DEC_START_2021],
    ])
    assert data_autumn.shape[1] == DAYS_IN_AUTUMN * HOURS_IN_DAY * YEARS
    return data_autumn


def get_daytime_data(data):
    """Select time-series electricity generation data during daytime in 2019, 2020 and 2021."""
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
    """Select time-series electricity generation data during nighttime in 2019, 2020 and 2021."""
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


def get_fitted_data(data, indices, iterations):
    average_mix = get_average_mixes(data, indices)
    distributions = fit_uniform_distributions(average_mix)
    data = generate_fitted_samples(indices, distributions, iterations)
    return data


def get_average_mixes(data, indices):
    """Get average electricity market mixes for years 2019, 2020 and 2021."""
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


def fit_uniform_distributions(average_mix):
    """Derive min and max of uniform distributions based on yearly averages of market mixes for 2019, 2020 and 2021."""
    ids = np.ones(len(average_mix)) * sa.NoUncertainty.id
    mins = np.min(average_mix, axis=1)
    maxs = np.max(average_mix, axis=1)
    mask = mins == maxs
    ids[~mask] = sa.UniformUncertainty.id
    distributions = np.array(
        list(zip(ids, mins, maxs)),
        dtype=[('uncertainty_type', np.uint8), ('minimum', np.float64), ('maximum', np.float64)]
    )
    return distributions


def generate_fitted_samples(indices, distributions, iterations):
    """Generate samples for market mixes based on the derived uniform distributions."""
    samples = np.zeros([len(indices), iterations])
    samples[:] = np.nan
    # Some mixes do not have uncertainties
    mask = distributions['uncertainty_type'] == sa.NoUncertainty.id
    samples[mask, :] = np.tile(distributions['minimum'][mask], (iterations, 1)).T
    # Rest of the exchanges in market mixes are modelled with uniform distributions
    dicts = [
        {'minimum': d['minimum'], 'maximum': d['maximum'], 'uncertainty_type': sa.UniformUncertainty.id}
        for d in distributions[~mask]
    ]
    params = sa.UniformUncertainty.from_dicts(*dicts)
    rng = sa.RandomNumberGenerator(sa.UniformUncertainty, params)
    samples[~mask, :] = rng.generate_random_numbers(size=iterations)
    # Normalize samples to ensure unit sum constraint in the market mixes
    nsamples = normalize_samples(indices, samples)
    return nsamples


def normalize_samples(indices, samples):
    """Normalize samples, such that amounts of exchanges sum up to 1 in the market mixes."""
    nsamples = np.zeros(samples.shape)
    nsamples[:] = np.nan
    unique_cols = sorted(list(set(indices['col'])))
    for col in unique_cols:
        mask = indices['col'] == col
        nsamples[mask] = samples[mask] / samples[mask].sum(axis=0)
    return nsamples


def create_entsoe_dp(project_dir, option, iterations):
    """
    Possible options are: all, winter, spring, summer, autumn, daytime, nighttime, fitted.

    Note
    ====
    Random seed will be ignored in the bwp.create_datapackage() if sequential=True.
    But when running MC simulations, if bc.LCA sets seed_override to a random seed,
    then the data will no longer be taken sequentially from this datapackage.

    """
    dp = bwp.load_datapackage(ZipFS(str(project_dir / "akula" / "data" / "entso-timeseries.zip")))

    data = dp.get_resource("timeseries ENTSO electricity values.data")[0]
    indices = dp.get_resource("timeseries ENTSO electricity values.indices")[0]
    flip = dp.get_resource("timeseries ENTSO electricity values.flip")[0]

    if option == "all":
        print("Selecting complete ENTSO-E timeseries.")
        samples = data
    elif option == "winter":
        samples = get_winter_data(data)
    elif option == "spring":
        samples = get_spring_data(data)
    elif option == "summer":
        samples = get_summer_data(data)
    elif option == "autumn":
        samples = get_autumn_data(data)
    elif option == "daytime":
        samples = get_daytime_data(data)
    elif option == "nighttime":
        samples = get_nighttime_data(data)
    elif option == "fitted":
        samples = get_fitted_data(data, indices, iterations)
    else:
        print("No valid option specified, selecting complete ENTSO-E timeseries.")
        samples = data

    dp_samples = bwp.create_datapackage(sequential=True, seed=None)
    dp_samples.add_persistent_array(
        matrix='technosphere_matrix',
        indices_array=indices,
        data_array=samples,
        flip_array=flip,
    )
    return dp_samples


def compute_low_voltage_ch_lcia(project, dp_entsoe, iterations=1000, seed=None):
    """
    Compute climate change scores for the activity `market for electricity, low voltage, CH` based on ENTSOE data in dp.

    Note
    ====
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

    if dp_entsoe is not None:
        data_objs += [dp_entsoe]

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


def plot_lcia_scores(data):
    fig = go.Figure()
    showlegend = True
    visited = False

    for option, d in data.items():
        if option == 'ecoinvent':
            fillcolor = COLOR_DARKGRAY_HEX_OPAQUE
            linecolor = COLOR_DARKGRAY_HEX
            showlegend = True
            name = r'$\text{Ecoinvent}$'
            rank = 1
        else:
            fillcolor = COLOR_PSI_LPURPLE_OPAQUE
            linecolor = COLOR_PSI_LPURPLE
            if visited:
                showlegend = False
            visited = True
            name = r'$\text{ENTSO-E}$'
            rank = 2
        if option == "fitted":
            option = "yearly averages"
        latex = r'$\text{' + option + r'}$'

        fig.add_trace(go.Violin(x=d, y=[latex]*2000, name=name, showlegend=showlegend, legendrank=rank,
                                fillcolor=fillcolor, line=dict(color=linecolor)))

    fig.update_traces(orientation='h', side='positive', width=2, points=False)
    fig = update_fig_axes(fig)
    fig.update_xaxes(range=(-0.1, 0.9), title_text=r"$\text{LCIA scores, [kg CO}_2\text{-eq.]}$")
    fig.update_layout(xaxis_showgrid=True, xaxis_zeroline=False, yaxis_showgrid=False,
                      width=350, height=460, legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.55))

    return fig


def plot_electricity_profile(project, project_dir):
    bd.projects.set_current(project)
    activity = get_one_activity("ecoinvent 3.8 cutoff", name="market for electricity, low voltage", location="CH")
    dp = bwp.load_datapackage(ZipFS(str(project_dir / "akula" / "data" / "entso-timeseries.zip")))

    data = dp.get_resource("timeseries ENTSO electricity values.data")[0]
    indices = dp.get_resource("timeseries ENTSO electricity values.indices")[0]

    mask = indices['col'] == activity.id
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


def fit_pedigree_distributions(average_mix):
    """Instead of fitting uniform distributions to average mixes, it is also possible to use the pedigree approach.

    First draft of the implementation is provided below.
    """
    mean_normal = np.mean(average_mix, axis=1)

    # Expert judgement based on pedigree approach
    # ===========================================
    # Reliability 2:                Verified data partly based on assumptions / non-verified data based on measurements.
    # Completeness 3:               Representative data from only some sited (≪ 50%) relevant for the market considered
    #                               or >50% of sites but from shorter periods.
    # Temporal Correlation 1:       Less than 3 years of difference to the time period of the dataset.
    # Geographical Correlation 1:   Data from area under study.
    # Technical Correlation 2:      Data from processes and materials under study (i.e. identical technology)
    #                               but from different enterprises
    # Sample Size 1:                Following ecoinvent approach, this is set to 1, assuming that sample size
    #                               uncertainty is implicitly considered in other indicators.
    indicator_reliability = 2
    indicator_completeness = 3
    indicator_temporal_correlation = 1
    indicator_geographical_correlation = 1
    indicator_technical_correlation = 2
    indicator_sample_size = 1

    # Uncertainty factors that are equal to squared geometric standard deviation.
    gsigma2_reliability = PEDIGREE_MATRIX["reliability"][indicator_reliability]
    gsigma2_completeness = PEDIGREE_MATRIX["completeness"][indicator_completeness]
    gsigma2_temporal_correlation = PEDIGREE_MATRIX["temporal_correlation"][indicator_temporal_correlation]
    gsigma2_geographical_correlation = PEDIGREE_MATRIX["geographical_correlation"][indicator_geographical_correlation]
    gsigma2_technical_correlation = PEDIGREE_MATRIX["technical_correlation"][indicator_technical_correlation]
    gsigma2_sample_size = PEDIGREE_MATRIX["sample_size"][indicator_sample_size]

    # Compute standard deviation of the underlying normal distribution
    sigma_normal = 0.5 * np.sqrt(
        (np.log(gsigma2_reliability)) ** 2 + (np.log(gsigma2_completeness)) ** 2 +
        (np.log(gsigma2_temporal_correlation)) ** 2 + (np.log(gsigma2_geographical_correlation)) ** 2 +
        (np.log(gsigma2_technical_correlation)) ** 2 + (np.log(gsigma2_sample_size)) ** 2
    )
    sigma_normal = np.ones(len(mean_normal)) * sigma_normal
    # Variance of the normal distribution
    # sigma2_normal = sigma_normal**2

    # Variance of the lognormal distribution
    # sigma2_lognormal = np.sqrt((np.exp(sigma2_normal) - 1) * np.exp(2 * mean_normal + sigma2_normal))

    distributions = np.array(
        list(zip(mean_normal, sigma_normal)),
        dtype=[('loc', np.float64), ('scale', np.float64)]
    )

    return distributions
