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


def similar(a, b):
    return fuzz.partial_ratio(a, b) > 90 or fuzz.ratio(a, b) > 40


def find_uncertain_virtual_markets(database):
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


def generate_virtual_markets_datapackage(num_samples=25000, dirichlet_scale=250):
    bd.projects.set_current("GSA for archetypes")

    vm = find_uncertain_virtual_markets("ecoinvent 3.8 cutoff")

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / "virtual-markets.zip"), write=True),
        name="virtual markets",
        # set seed to have reproducible (though not sequential) sampling
        seed=42,
    )

    data_array = np.hstack(
        [
            dirichlet.rvs(
                np.array([exc["amount"] for exc in lst]) * dirichlet_scale,
                size=num_samples,
            )
            for lst in vm.values()
        ]
    ).T
    indices_array = np.array(
        [(exc.input.id, exc.output.id) for lst in vm.values() for exc in lst],
        dtype=bwp.INDICES_DTYPE,
    )
    # All inputs -> all True
    flip_array = np.ones(len(indices_array), dtype=bool)
    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=data_array,
        # Resource group name that will show up in provenance
        name="virtual markets",
        indices_array=indices_array,
        flip_array=flip_array,
    )
    dp.finalize_serialization()


if __name__ == "__main__":
    generate_virtual_markets_datapackage()
