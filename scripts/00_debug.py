import numpy as np
from pathlib import Path
import bw2data as bd
import pandas as pd
import bw_processing as bwp

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
DATA_DIR = Path(__file__).parent.parent.resolve() / "data"
GSA_DIR = DATA_DIR / "sensitivity-analysis"
DP_DIR = DATA_DIR / "datapackages"

if __name__ == "__main__":
    bd.projects.set_current(PROJECT)
    ei = bd.Database("ecoinvent 3.8. cutoff")
    bi = bd.Database("biosphere3")

    rcorr = read_pickle(GSA_DIR / "correlated" / "ranking.indices.model_9.200.222201.20000.pickle")
    rindp = read_pickle(GSA_DIR / "independent" / "ranking.indices.model_3.200.222201.20000.pickle")

    bcorr, tcorr, ccorr = rcorr["biosphere"], rcorr["technosphere"], rcorr["characterization"]
    bindp, tindp, cindp = rindp["biosphere"], rindp["technosphere"], rindp["characterization"]

    tcorr = np.array([el[0:2] for el in tcorr], dtype=bwp.INDICES_DTYPE)
    tindp = np.array([el[0:2] for el in tindp], dtype=bwp.INDICES_DTYPE)
    bcorr = np.array([el[0:2] for el in bcorr], dtype=bwp.INDICES_DTYPE)
    bindp = np.array([el[0:2] for el in bindp], dtype=bwp.INDICES_DTYPE)

    params, dp_parameterization = generate_parameterization_datapackage(
        FP_ECOINVENT, "parameterization", iterations, seed
    )
    dp_combustion = generate_combustion_datapackage("combustion", iterations, seed)
    dp_entsoe = generate_entsoe_datapackage("entsoe", iterations, seed)
    dp_markets = generate_markets_datapackage("markets", iterations, seed)

    tparm = dp_parameterization.data[0]
    bparm = dp_parameterization.data[3]
    tcomb = dp_combustion.data[0]
    bcomb = dp_combustion.data[3]
    tents = dp_entsoe.data[0]
    tmrkt = dp_markets.data[0]

    t2000corr = read_pickle(GSA_DIR / "correlated" / "indices.tech.without_lowinf.2000.xgb.model_9.pickle")
    b2000corr = read_pickle(GSA_DIR / "correlated" / "indices.bio.without_lowinf.2000.xgb.model_9.pickle")
    c2000corr = read_pickle(GSA_DIR / "correlated" / "indices.cf.without_lowinf.2000.xgb.model_9.pickle")

    t2000indp = read_pickle(GSA_DIR / "independent" / "indices.tech.without_lowinf.2000.xgb.model_3.pickle")
    b2000indp = read_pickle(GSA_DIR / "independent" / "indices.bio.without_lowinf.2000.xgb.model_3.pickle")
    c2000indp = read_pickle(GSA_DIR / "independent" / "indices.cf.without_lowinf.2000.xgb.model_3.pickle")

    inds = np.intersect1d(tcorr, tents)
    for ind in inds:
        row = ind["row"]
        act = bd.get_activity(row)
        print(act["name"], act.get("location"))

        col = ind["col"]
        act = bd.get_activity(col)
        print(act["name"], act["location"])

        print()
    print()
