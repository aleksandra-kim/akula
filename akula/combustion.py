from bw2data.backends import ExchangeDataset as ED
from fs.zipfs import ZipFS
from pathlib import Path
import bw2data as bd
import bw_processing as bwp
import math
import numpy as np
import stats_arrays as sa
from tqdm import tqdm


DATA_DIR = Path(__file__).parent.resolve() / "data"


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


def get_samples_and_scaling_vector(activity, fuels, size=10, seed=None):
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


def get_static_samples_and_scaling_vector(activity, fuels, const_factor=10):
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


def generate_liquid_fuels_combustion_correlated_samples(size=25000, seed=42):
    bd.projects.set_current('GSA for archetypes')

    fuels = get_liquid_fuels()
    co2 = [x for x in bd.Database('biosphere3') if x['name'] == 'Carbon dioxide, fossil']
    ei = bd.Database("ecoinvent 3.8 cutoff")

    candidate_codes = ED.select(ED.output_code).distinct().where(
        (ED.input_code << {o['code'] for o in co2}) & (ED.output_database == 'ecoinvent 3.8 cutoff')).tuples()
    candidates = [ei.get(code=o[0]) for o in candidate_codes]
    print("Found {} candidate activities".format(len(candidates)))

    processed_log = open(DATA_DIR / "liquid-fuels.processed.log", "w")
    unbalanced_log = open(DATA_DIR / "liquid-fuels.unbalanced.log", "w")
    error_log = open(DATA_DIR / "liquid-fuels.error.log", "w")

    indices_tech, flip_tech, data_tech = [], [], []
    indices_bio, data_bio = [], []

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / "liquid-fuels-kilogram.zip"), write=True),
        name="liquid-fuels",
        # set seed to have reproducible (though not sequential) sampling
        seed=42,
    )

    for candidate in tqdm(candidates):
        if not carbon_fuel_emissions_balanced(candidate, fuels, co2):
            unbalanced_log.write("{}\t{}\n".format(candidate.id, str(candidate)))

        try:
            tindices, tflip, tsample, factors = get_samples_and_scaling_vector(candidate, fuels, size=size, seed=seed)
            if tindices.shape == (0,):
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


def generate_liquid_fuels_combustion_local_sa_samples(const_factor=10):
    bd.projects.set_current('GSA for archetypes')

    fuels = get_liquid_fuels()
    co2 = [x for x in bd.Database('biosphere3') if x['name'] == 'Carbon dioxide, fossil']
    ei = bd.Database("ecoinvent 3.8 cutoff")

    candidate_codes = ED.select(ED.output_code).distinct().where(
        (ED.input_code << {o['code'] for o in co2}) & (ED.output_database == 'ecoinvent 3.8 cutoff')).tuples()
    candidates = [ei.get(code=o[0]) for o in candidate_codes]

    indices_tech, static_tech = [], []
    indices_bio, sample_bio, static_bio = [], [], []

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"local-sa-liquid-fuels-factor{const_factor}.zip"), write=True),
        name="local-sa-liquid-fuels",
        # set seed to have reproducible (though not sequential) sampling
        seed=42,
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

    # for a, b, c in zip(indices, flip, data):
    #     print(a.shape, b.shape, c.shape)

    indices = indices_tech + indices_bio
    print("Found {} exchanges in {} datasets".format(sum(len(x) for x in indices), len(indices)))

    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=np.vstack(data_tech),
        # Resource group name that will show up in provenance
        name="local-sa-liquid-fuels-tech",
        indices_array=np.hstack(indices_tech),
        flip_array=np.ones(len(indices_tech)),
    )

    dp.add_persistent_array(
        matrix="biosphere_matrix",
        data_array=np.vstack(data_bio),
        # Resource group name that will show up in provenance
        name="local-sa-liquid-fuels-bio",
        indices_array=np.hstack(indices_bio),
    )

    dp.finalize_serialization()


if __name__ == "__main__":
    # generate_liquid_fuels_combustion_correlated_samples()
    generate_liquid_fuels_combustion_local_sa_samples()
