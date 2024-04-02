import bw2data as bd
import bw_processing as bwp
from fs.zipfs import ZipFS
from pathlib import Path
import numpy as np
import sys

PROJECT = "GSA with correlations"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = Path(__file__).parent.parent.resolve() / "data"

SAMPLES = 25000

sys.path.append(str(PROJECT_DIR))
from akula.electricity import ENTSODataConverter
from akula.electricity import add_swiss_residual_mix

if __name__ == "__main__":

    # Create datapackages for xgboost
    bd.projects.set_current(PROJECT)

    name = "entso-timeseries"

    dp = bwp.load_datapackage(ZipFS(str(DATA_DIR / f"{name}.zip")))
    indices = dp.data[0]
    data = dp.data[1]
    flip = dp.data[2]

    ndata = data.shape[1]

    random_seeds = [85, 86]
    num_samples = 15000

    for random_seed in random_seeds:
        print(f"Random seed {random_seed}")

        np.random.seed(random_seed)
        choice = np.random.choice(np.arange(ndata), num_samples)

        data_current = data[:, choice]

        # Create datapackage
        dp = bwp.create_datapackage(
            fs=ZipFS(str(DATA_DIR / "xgboost" / f"{name}-{random_seed}.zip"), write=True),
            name='timeseries ENTSO electricity values',
            seed=random_seed,
            sequential=True,
        )
        dp.add_persistent_array(
            matrix="technosphere_matrix",
            data_array=data_current,
            # Resource group name that will show up in provenance
            name='timeseries ENTSO electricity values',
            indices_array=indices,
            flip_array=flip,
        )
        dp.finalize_serialization()

    print("")
