import bw2data as bd
import bw_processing as bwp
from fs.zipfs import ZipFS
from pathlib import Path
from bentso.constants import ENTSO_COUNTRIES
from tqdm import tqdm
import numpy as np
import os

from .entso_data_converter import ENTSODataConverter
from .add_residual_mix import add_swiss_residual_mix

BENTSO_DATA_DIR = os.environ["BENTSO_DATA_DIR"]
DATA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "datapackages"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def create_average_entso_datapackages(project, years=(2019, 2020, 2021)):

    bd.projects.set_current(project)
    add_swiss_residual_mix(project)

    dc = ENTSODataConverter("ecoinvent 3.8 cutoff")

    # Start with average datapackages
    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / "entsoe-average.zip"), write=True),
        name="Average ENTSO generation and trade",
        # set seed to have reproducible (though not sequential) sampling
        seed=42,
    )

    indices, data = [], []

    for country in tqdm(ENTSO_COUNTRIES):

        print(country)

        if country in ("LU", "MT", "MK"):
            continue

        tqdm.write(country)

        hv = dc.data_dict_for_high_voltage_market(country, years=years)
        keys = sorted(hv)

        indices.append(np.array([(x, y) for x, y, _ in keys], dtype=bwp.INDICES_DTYPE))
        data.append(np.array([hv[key].fillna(0).mean() for key in keys]))

        mv = dc.data_dict_for_medium_voltage_market(country, years=years)
        keys = sorted(mv)

        indices.append(np.array([(x, y) for x, y, _ in keys], dtype=bwp.INDICES_DTYPE))
        data.append(np.array([mv[key].fillna(0).mean() for key in keys]))

        lv = dc.data_dict_for_low_voltage_market(country, years=years)
        keys = sorted(lv)

        indices.append(np.array([(x, y) for x, y, _ in keys], dtype=bwp.INDICES_DTYPE))
        data.append(np.array([lv[key].fillna(0).mean() for key in keys]))

    data = np.hstack(data)

    dp.add_persistent_vector(
        matrix="technosphere_matrix",
        data_array=data,
        # Resource group name that will show up in provenance
        name="average ENTSO electricity values",
        indices_array=np.hstack(indices),
        flip_array=np.ones(len(data), dtype=bool),
    )
    dp.finalize_serialization()


def create_timeseries_entso_datapackages(project, years=(2019, 2020, 2021)):
    bd.projects.set_current(project)
    add_swiss_residual_mix(project)

    dc = ENTSODataConverter("ecoinvent 3.8 cutoff")

    # Start with average datapackages
    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / "entsoe-timeseries.zip"), write=True),
        name="2019-2021 ENTSO generation and trade timeseries",
        # set seed to have reproducible (though not sequential) sampling
        seed=42,
    )

    indices, data = [], []

    for country in tqdm(sorted(ENTSO_COUNTRIES)):

        print(country)

        if country in ("LU", "MT", "MK"):
            continue

        tqdm.write(country)

        hv = dc.data_dict_for_high_voltage_market(country, years=years)
        keys = sorted(hv)

        indices.append(np.array([(x, y) for x, y, _ in keys], dtype=bwp.INDICES_DTYPE))
        data.append(np.array([hv[key].fillna(0) for key in keys]))

        mv = dc.data_dict_for_medium_voltage_market(country, years=years)
        keys = sorted(mv)

        indices.append(np.array([(x, y) for x, y, _ in keys], dtype=bwp.INDICES_DTYPE))
        data.append(np.array([mv[key].fillna(0) for key in keys]))

        lv = dc.data_dict_for_low_voltage_market(country, years=years)
        keys = sorted(lv)

        indices.append(np.array([(x, y) for x, y, _ in keys], dtype=bwp.INDICES_DTYPE))
        data.append(np.array([lv[key].fillna(0) for key in keys]))

    min_hours = min(arr.shape[1] for arr in data)
    data = [arr[:, :min_hours] for arr in data]

    data = np.vstack(data)

    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=data,
        # Resource group name that will show up in provenance
        name="timeseries ENTSO electricity values",
        indices_array=np.hstack(indices),
        flip_array=np.ones(len(data), dtype=bool),
    )
    dp.finalize_serialization()


def generate_entsoe_datapackage(name, num_samples, seed=42):

    fp_datapackage = DATA_DIR / f"{name}-{seed}-{num_samples}.zip"

    if not fp_datapackage.exists():

        dp_timeseries = bwp.load_datapackage(ZipFS(str(DATA_DIR / "entsoe-timeseries.zip")))

        data = dp_timeseries.get_resource("timeseries ENTSO electricity values.data")[0]
        indices = dp_timeseries.get_resource("timeseries ENTSO electricity values.indices")[0]
        flip = dp_timeseries.get_resource("timeseries ENTSO electricity values.flip")[0]

        np.random.seed(seed)
        inds = np.random.choice(data.shape[1], num_samples, replace=True)

        dp = bwp.create_datapackage(
            fs=ZipFS(str(fp_datapackage), write=True),
            name=name,
            seed=seed,
            sequential=True,
        )

        dp.add_persistent_array(
            matrix='technosphere_matrix',
            name=name,
            indices_array=indices,
            data_array=data[:, inds],
            flip_array=flip,
        )

        dp.finalize_serialization()

    else:

        dp = bwp.load_datapackage(ZipFS(str(fp_datapackage)))

    return dp
