from bw2data.backends import ExchangeDataset as ED
from fs.zipfs import ZipFS
from pathlib import Path
import bw2data as bd
import bw_processing as bwp
import math
import numpy as np
import stats_arrays as sa
from tqdm import tqdm

# Local files
from utils import create_static_datapackage


DATA_DIR = Path(__file__).parent.resolve() / "data"
SAMPLES = 25000


def carbon_fuel_emissions_balanced(activity, fuels, co2):
    """Check that we can rescale carbon in fuel to CO2 emissions by checking their stoichiometry.

    Returns a ``bool`."""
    try:
        total_carbon = sum(
            # Carbon content amount is fraction of mass, unitless
            exc['amount'] * exc['properties']['carbon content']['amount']
            for exc in activity.technosphere()
            if exc.input in fuels)
    except KeyError:
        return False
    conversion = 12 / (12 + 16 * 2)
    total_carbon_in_co2 = sum(
        exc['amount'] * conversion
        for exc in activity.biosphere()
        if exc.input in co2
    )
    return math.isclose(total_carbon, total_carbon_in_co2, rel_tol=1e-06, abs_tol=1e-3)


def get_samples_and_scaling_vector(activity, fuels, size=10, seed=42):
    """Draw ``size`` samples from technosphere exchanges for ``activity`` whose inputs are in ``fuels``.

    Returns:
        * Numpy indices array with shape ``(len(found_exchanges)),)``
        * Numpy flip array with shape ``(len(found_exchanges,))``
        * Numpy data array with shape ``(size, len(found_exchanges))``
        * Scaling vector with relative total carbon consumption and shape ``(size,)``.
    """
    exchanges = [exc for exc in activity.technosphere() if exc.input in fuels]
    static_total = sum(exc['amount'] * exc['properties']['carbon content']['amount'] for exc in exchanges)
    sample = sa.MCRandomNumberGenerator(
        sa.UncertaintyBase.from_dicts(*[exc.as_dict() for exc in exchanges]),
        seed=seed
    ).generate(samples=size)
    indices = np.array([(exc.input.id, exc.output.id) for exc in exchanges], dtype=bwp.INDICES_DTYPE)
    carbon_fraction = np.array([exc['properties']['carbon content']['amount'] for exc in exchanges]).reshape((-1, 1))
    carbon_total_per_sample = (sample * carbon_fraction).sum(axis=0).ravel()
    flip = np.ones(indices.shape, dtype=bool)
    assert carbon_total_per_sample.shape == (size,)
    return indices, flip, sample, carbon_total_per_sample / static_total


def get_static_samples_and_scaling_vector(activity, fuels, const_factor=10.0):
    exchanges = [exc for exc in activity.technosphere() if exc.input in fuels]
    static_total = sum(exc['amount'] * exc['properties']['carbon content']['amount'] for exc in exchanges)
    static = np.array([exc['amount'] for exc in activity.technosphere() if exc.input in fuels])
    sample = np.tile(static, (len(static), 1))
    np.fill_diagonal(sample, static*const_factor)
    indices = np.array([(exc.input.id, exc.output.id) for exc in exchanges], dtype=bwp.INDICES_DTYPE)
    carbon_fraction = np.array([exc['properties']['carbon content']['amount'] for exc in exchanges]).reshape((-1, 1))
    carbon_total_per_sample = (sample * carbon_fraction).sum(axis=0).ravel()
    assert carbon_total_per_sample.shape == (len(static),)
    return indices, static, carbon_total_per_sample / static_total


def get_static_samples_validation():
    return


def rescale_biosphere_exchanges_by_factors(activity, factors, flows):
    """Rescale biosphere exchanges with flows ``flows`` from ``activity`` by vector ``factors``.

    ``flows`` are biosphere flow objects, with e.g. all the CO2 flows, but also other flows such as metals,
    volatile organics, etc. Only rescales flows in ``flows`` which are present in ``activity`` exchanges.

    Assumes the static values are balanced, i.e. won't calculate CO2 emissions from carbon in
    fuels but just rescales given values.

    Returns: Numpy indices and data arrays with shape (number of exchanges found, len(factors)).
    Returns:
        * Numpy indices array with shape ``(len(found_exchanges)),)``
        * Numpy flip array with shape ``(len(found_exchanges,))``
        * Numpy data array with shape ``(len(found_exchanges), len(factors))``
    """
    indices, sample, static = [], [], []
    assert isinstance(factors, np.ndarray) and len(factors.shape) == 1

    for exc in activity.biosphere():
        if exc.input in flows:
            indices.append((exc.input.id, exc.output.id))
            sample.append(factors * exc['amount'])
            static.append(exc['amount'])

    # return np.array(indices, dtype=bwp.INDICES_DTYPE), np.zeros(len(indices), dtype=bool), np.vstack(data)
    return np.array(indices, dtype=bwp.INDICES_DTYPE), np.vstack(sample), np.vstack(static)


def get_liquid_fuels():
    flows = ('market for diesel,', 'diesel,', 'petrol,', 'market for petrol,')
    return [x
            for x in bd.Database("ecoinvent 3.8 cutoff")
            if x['unit'] == 'kilogram'
            and (any(x['name'].startswith(flow) for flow in flows) or x['name'] == 'market for diesel')
            ]


def get_candidates(co2, ei_name='ecoinvent 3.8 cutoff'):
    ei = bd.Database(ei_name)
    candidate_codes = ED.select(ED.output_code).distinct().where(
        (ED.input_code << {o['code'] for o in co2}) & (ED.output_database == ei_name)).tuples()
    candidates = [ei.get(code=o[0]) for o in candidate_codes]
    print("Found {} candidate activities".format(len(candidates)))
    return candidates


def get_co2():
    return [x for x in bd.Database('biosphere3') if x['name'] == 'Carbon dioxide, fossil']


def generate_liquid_fuels_combustion_correlated_samples(size=25000, seed=42):
    bd.projects.set_current('GSA for archetypes')

    fuels = get_liquid_fuels()
    co2 = get_co2()
    candidates = get_candidates(co2)

    processed_log = open(DATA_DIR / "liquid-fuels.processed.log", "w")
    unbalanced_log = open(DATA_DIR / "liquid-fuels.unbalanced.log", "w")
    error_log = open(DATA_DIR / "liquid-fuels.error.log", "w")

    indices_tech, flip_tech, data_tech = [], [], []
    indices_bio, data_bio = [], []

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"liquid-fuels-kilogram-{seed}.zip"), write=True),
        name="liquid-fuels",
        seed=seed,
        sequential=True,
    )

    for candidate in tqdm(candidates):
        if not carbon_fuel_emissions_balanced(candidate, fuels, co2):
            unbalanced_log.write("{}\t{}\n".format(candidate.id, str(candidate)))

        try:
            tindices, tflip, tsample, factors = get_samples_and_scaling_vector(candidate, fuels, size=size, seed=seed)
            if tindices.shape == (0, ):
                continue

            bindices, bsample, _ = rescale_biosphere_exchanges_by_factors(candidate, factors, co2)

            indices_tech.append(tindices)
            flip_tech.append(tflip)
            data_tech.append(tsample)

            indices_bio.append(bindices)
            data_bio.append(bsample)

            processed_log.write("{}\t{}\n".format(candidate.id, str(candidate)))
        except KeyError:
            error_log.write("{}\t{}\n".format(candidate.id, str(candidate)))

    processed_log.close()
    unbalanced_log.close()
    error_log.close()

    # for a, b, c in zip(indices, flip, data):
    #     print(a.shape, b.shape, c.shape)

    indices = indices_tech + indices_bio
    print("Found {} exchanges in {} datasets".format(sum(len(x) for x in indices), len(indices)))

    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=np.vstack(data_tech),
        # Resource group name that will show up in provenance
        name="liquid-fuels-tech",
        indices_array=np.hstack(indices_tech),
        flip_array=np.hstack(flip_tech),
    )

    dp.add_persistent_array(
        matrix="biosphere_matrix",
        data_array=np.vstack(data_bio),
        # Resource group name that will show up in provenance
        name="liquid-fuels-bio",
        indices_array=np.hstack(indices_bio),
    )

    dp.finalize_serialization()


def create_technosphere_dp(indices_tech, static_tech, const_factor):
    # Create data for technosphere
    tindices = np.hstack(indices_tech)
    tstatic = np.hstack(static_tech)
    tdata = np.tile(tstatic, (len(tstatic), 1))
    np.fill_diagonal(tdata, tstatic*const_factor)
    targsort = np.argsort(tindices)
    assert len(tindices) == len(tstatic)
    return tindices[targsort], tdata[targsort]
    # return tindices, tdata


def create_biosphere_dp(indices_tech, indices_bio, sample_bio, static_bio):
    # Create data for biosphere
    num_acts = len(indices_tech)
    bindices = np.hstack(indices_bio)
    bstatic, blist = [], []
    ib, it = 0, 0
    for i in range(num_acts):
        bstatic.append(static_bio[i].reshape(-1, 1))
        len_tech = len(indices_tech[i])
        len_bio = len(indices_bio[i])
        assert len_tech == sample_bio[i].shape[1]
        for j in range(len_bio):
            blist.append(
                [ib, it, list(sample_bio[i][j, :])]
            )
        it += len_tech
        ib += 1
    bstatic = np.vstack(bstatic).flatten()
    assert len(bindices) == len(bstatic)

    bdata = np.tile(bstatic, (it, 1)).T
    for el in blist:
        end = el[1] + len(el[2])
        bdata[el[0], el[1]:end] = el[2]

    bargsort = np.argsort(bindices)

    return bindices[bargsort], bdata[bargsort]
    # return bindices, bdata


def generate_liquid_fuels_combustion_local_sa_samples(const_factor=10.0, seed=42):
    bd.projects.set_current('GSA for archetypes')

    fuels = get_liquid_fuels()
    co2 = get_co2()
    candidates = get_candidates(co2)

    indices_tech, static_tech = [], []
    indices_bio, sample_bio, static_bio = [], [], []

    name = "liquid-fuels-kilogram"

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"local-sa-{const_factor:.0e}-{name}.zip"), write=True),
        name="local-sa-liquid-fuels",
        seed=seed,
        sequential=True,
    )

    for candidate in tqdm(candidates):
        try:
            tindices, tstatic, factors = get_static_samples_and_scaling_vector(candidate, fuels, const_factor)
            if tindices.shape == (0,):
                continue

            bindices, bsample, bstatic = rescale_biosphere_exchanges_by_factors(candidate, factors, co2)

            indices_tech.append(tindices)
            static_tech.append(tstatic)

            indices_bio.append(bindices)
            sample_bio.append(bsample)
            static_bio.append(bstatic)

        except KeyError:
            pass

        print("")

    assert len(indices_tech) == len(indices_bio)

    tindices, tdata = create_technosphere_dp(indices_tech, static_tech, const_factor)
    bindices, bdata = create_biosphere_dp(indices_tech, indices_bio, sample_bio, static_bio)

    indices = indices_tech + indices_bio
    print("Found {} exchanges in {} datasets".format(sum(len(x) for x in indices), len(indices)))

    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=tdata.T,
        # Resource group name that will show up in provenance
        name=f"local-sa-{name}-tech",
        indices_array=tindices,
        flip_array=np.ones(len(tindices), dtype=bool),
    )

    dp.add_persistent_array(
        matrix="biosphere_matrix",
        data_array=bdata,
        # Resource group name that will show up in provenance
        name=f"local-sa-{name}-bio",
        indices_array=bindices,
    )

    dp.finalize_serialization()


def create_static_data(indices_tech, indices_bio):

    cols = list(set(indices_tech['cols']))

    for col in cols:
        activity = bd.get_activity(int(col))
        tindices, tstatic, factors = get_static_samples_and_scaling_vector(activity, fuels)
        if tindices.shape == (0,):
            continue

        bindices, bsample, bstatic = rescale_biosphere_exchanges_by_factors(candidate, factors, co2)
    return



def create_dynamic_datapackage(name, indices_tech, indices_bio, num_samples):
    dp = []
    return dp


def generate_validation_datapackage(mask_tech, mask_bio, num_samples=SAMPLES, seed=42):

    name = "liquid-fuels-kilogram"
    dp = bwp.load_datapackage(ZipFS(str(DATA_DIR / f"validation-{name}-{seed}.zip")))  # TODO this dp might not exist yet

    tindices = dp.get_resource('liquid-fuels-tech.indices')[0]
    assert len(tindices) == len(mask_tech)
    static_tindices = tindices[~mask_tech]
    dynamic_tindices = tindices[mask_tech]

    bindices = dp.get_resource('liquid-fuels-bio.indices')[0]
    assert len(bindices) == len(mask_bio)
    static_bindices = bindices[~mask_bio]
    dynamic_bindices = bindices[mask_bio]

    indices_tech, data_tech, flip_tech, indices_bio, data_bio = create_static_data(static_tindices, static_bindices)

    dp_static = create_static_datapackage(
        name, indices_tech=None, data_tech=None, flip_tech=None, indices_bio=None, data_bio=None
    )

    dp_dynamic = create_dynamic_datapackage(name, dynamic_tindices, dynamic_bindices, num_samples)

    return dp_static, dp_dynamic


if __name__ == "__main__":
    random_seeds = [85, 86]
    num_samples = 15000
    for random_seed in random_seeds:
        print(f"Random seed {random_seed}")
        generate_liquid_fuels_combustion_correlated_samples(num_samples, random_seed)

    # dp = bwp.load_datapackage(ZipFS(str(DATA_DIR / "liquid-fuels-kilogram-43.zip")))
    # dp_data_tech = dp.get_resource('liquid-fuels-tech.data')[0]
    # dp_indices_tech = dp.get_resource('liquid-fuels-tech.indices')[0]
    # dp_data_bio = dp.get_resource('liquid-fuels-bio.data')[0]
    # dp_indices_bio = dp.get_resource('liquid-fuels-bio.indices')[0]
    #
    # print(dp_data_tech)
    # print(dp_data_bio)

    # generate_liquid_fuels_combustion_local_sa_samples(const_factor=10.0)
    # generate_liquid_fuels_combustion_local_sa_samples(const_factor=0.1)

    # tmask = np.random.choice([True, False], size=1403, p=[0.1, 0.9])
    # bmask = np.random.choice([True, False], size=407, p=[0.1, 0.9])
    # dps, dpd = generate_validation_datapackage(tmask, bmask, SAMPLES)

    print("")
