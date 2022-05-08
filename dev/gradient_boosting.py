import numpy as np
from pathlib import Path
from fs.zipfs import ZipFS
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from time import time
from sklearn.linear_model import LinearRegression

from gsa_framework.utils import read_pickle, write_pickle
from gsa_framework.sensitivity_analysis.gradient_boosting import GradientBoosting
from gsa_framework.sensitivity_methods.gradient_boosting import xgboost_indices_base


# local files
from akula.combustion import DATA_DIR


if __name__ == "__main__":

    project = 'GSA for archetypes'
    bd.projects.set_current(project)

    co = bd.Database('swiss consumption 1.0')
    fu = [act for act in co if "ch hh average consumption aggregated, years 151617" == act['name']][0]

    write_dir = Path("write_files") / project.lower().replace(" ", "_") \
        / fu['name'].lower().replace(" ", "_").replace(",", "")

    dp_names_bg = ["technosphere", "biosphere", "characterization"]
    dp_names_mo = [
        "ecoinvent-parameterization-parameters", "liquid-fuels-kilogram", "implicit-markets", "entso-timeseries",
    ]

    iterations = 25000
    num_parameters = 25000
    random_seeds = [43]  #, 44, 45, 46]
    Y = []
    X = []
    for random_seed in random_seeds:

        y_ = read_pickle(write_dir / f"mc.xgboost.{iterations}.{random_seed}.pickle")
        Y.append(np.array(y_))

        x_ = []
        for dp_name in dp_names_bg + dp_names_mo:
            fp = DATA_DIR / "xgboost" / f"{dp_name}-{random_seed}.zip"
            dp = bwp.load_datapackage(ZipFS(str(fp)))
            if dp_name not in dp_names_mo:
                mask = np.ones(len(dp.data[0]), dtype=bool)
            else:
                if dp_name == "entso-timeseries":
                    dp_name = "entso-average"
                mask = read_pickle(write_dir / f"mask.{dp_name}.without_lowinf.params_{num_parameters:.1e}")
            x_.append(dp.data[1][mask])

        X.append(np.vstack(x_).T)

    Y = np.hstack(Y)
    X = np.vstack(X)

    print(X.shape)

    sigma_Y = np.std(Y)
    sigma_X = np.std(X, axis=0)

    # Check linearity
    reg = LinearRegression().fit(X, Y)
    write_pickle(reg, "regression.pickle")
    coef = reg.coef_ * sigma_X / sigma_Y
    r2 = sum(coef**2)
    print(r2)

    # 3.1.3. gradient boosting
    tuning_parameters = dict(
        learning_rate=0.15,
        gamma=0,
        min_child_weight=300,
        max_depth=4,
        reg_lambda=0,
        reg_alpha=0,
        n_estimators=100,  # 600,
        subsample=0.3,
        colsample_bytree=0.2,
    )
    t0 = time()
    xgboost_model = xgboost_indices_base(
        Y=Y,
        X=X,
        tuning_parameters=tuning_parameters,
        test_size=0.2,
        xgb_model=None,
        importance_types=None,  # TODO set default to empty list?
        flag_return_xgb_model=True,
    )
    t1 = time()

    print(f"Xgboost training took {t1-t0} seconds")


