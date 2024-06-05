import numpy as np
from pathlib import Path
import bw2data as bd
import pandas as pd

import os
os.environ["ENTSOE_API_TOKEN"] = "0d6ea062-f603-43d3-bc60-176159803035"
os.environ["BENTSO_DATA_DIR"] = "/home/aleksandrakim/LCAfiles/bentso_data"

from akula.utils import read_pickle, write_pickle
from akula.sensitivity_analysis import create_all_datapackages
from akula.monte_carlo import compute_consumption_lcia
from akula.parameterization import generate_parameterization_datapackage
from akula.combustion import generate_combustion_datapackage
from akula.electricity import generate_entsoe_datapackage
from akula.markets import generate_markets_datapackage
from akula.sensitivity_analysis.ranking import compute_shap_values, get_feature_importances_shap_values, get_influential_shapley

iterations = 2_000
seed = 222201

FP_ECOINVENT = "/home/aleksandrakim/LCAfiles/ecoinvent_38_cutoff/datasets"
PROJECT = "GSA with correlations"


if __name__ == "__main__":
    bd.projects.set_current(PROJECT)
    ei = bd.Database("ecoinvent 3.8. cutoff")
    bi = bd.Database("biosphere3")

    params, dp_parameterization = generate_parameterization_datapackage(
        FP_ECOINVENT, "parameterization", iterations, seed
    )
    dp_combustion = generate_combustion_datapackage("combustion", iterations, seed)
    dp_entsoe = generate_entsoe_datapackage("entsoe", iterations, seed)
    dp_markets = generate_markets_datapackage("markets", iterations, seed)

    # Compute feature importance values
    num_lowinf_lsa = 25000
    # tag = "2"
    # correlations = True
    # num_inf = 200
    # shap_values = compute_shap_values(tag, iterations, seed, num_lowinf_lsa, correlations)
    # features = np.arange(num_lowinf_lsa)
    # feature_importances = get_feature_importances_shap_values(shap_values, features, num_inf)
    #
    # # Get top `num_inf` features with highest shapley value scores
    # top_features = get_influential_shapley(feature_importances, num_inf, iterations, seed, correlations)
    #
    # print()
