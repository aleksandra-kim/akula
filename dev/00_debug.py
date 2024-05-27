import numpy as np
from pathlib import Path

import os
os.environ["ENTSOE_API_TOKEN"] = "0d6ea062-f603-43d3-bc60-176159803035"
os.environ["BENTSO_DATA_DIR"] = "/home/aleksandrakim/LCAfiles/bentso_data"

from akula.utils import read_pickle, write_pickle
from akula.sensitivity_analysis import create_all_datapackages
from akula.monte_carlo import compute_consumption_lcia


iterations = 2_000
seed = 222201

FP_ECOINVENT = "/home/aleksandrakim/LCAfiles/ecoinvent_38_cutoff/datasets"
PROJECT = "GSA with correlations"

if __name__ == "__main__":
    dps = create_all_datapackages(FP_ECOINVENT, PROJECT, iterations, seed)
    compute_consumption_lcia(PROJECT, iterations, seed, datapackages=dps)
