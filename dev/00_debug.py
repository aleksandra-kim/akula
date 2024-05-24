import numpy as np
from pathlib import Path

import os
os.environ["ENTSOE_API_TOKEN"] = "0d6ea062-f603-43d3-bc60-176159803035"
os.environ["BENTSO_DATA_DIR"] = "/home/aleksandrakim/LCAfiles/bentso_data"

from akula.utils import read_pickle, write_pickle
from akula.sensitivity_analysis import get_random_seeds


iterations = 125_000
seed = 222201
_, _, seeds = get_random_seeds(iterations, seed)

fp = Path('/home/aleksandrakim/ProjectsPycharm/akula/data/sensitivity-analysis/high-dimensional-screening')
# fp_new = fp / "server"
# for seed in seeds:
#     fn = f"scores.without_lowinf.25000.{seed}.5000.pickle"
#     Y = read_pickle(fp/fn)
#     Y_new = np.hstack([Y[-1], Y[:-1]])
#     write_pickle(Y_new, fp_new / fn)

for seed in seeds:
    fn = f"scores.without_lowinf.25000.{seed}.5000.pickle"
    Y = read_pickle(fp/fn)
    Y_new = Y.tolist()
    write_pickle(Y_new, fp / fn)
