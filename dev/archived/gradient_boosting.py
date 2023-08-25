import numpy as np
from pathlib import Path
from fs.zipfs import ZipFS
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from time import time
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from copy import deepcopy

from gsa_framework.utils import read_pickle, write_pickle
from gsa_framework.sensitivity_analysis.gradient_boosting import GradientBoosting
from gsa_framework.sensitivity_methods.gradient_boosting import xgboost_indices_base

from akula.sensitivity_analysis.local_sensitivity_analysis import get_mask


# local files
from akula.combustion import DATA_DIR


if __name__ == "__main__":

    project = 'GSA for archetypes'
    bd.projects.set_current(project)

    ei = bd.Database("ecoinvent 3.8 cutoff").datapackage()
    tei = ei.filter_by_attribute('matrix', 'technosphere_matrix')
    tindices_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.indices')[0]
    tdata_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.data')[0]
    tflip_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.flip')[0]
    tdistributions_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.distributions')[0]

    co = bd.Database('swiss consumption 1.0')
    fu = [act for act in co if "ch hh average consumption aggregated, years 151617" == act['name']][0]

    # write_dir = Path("write_files") / project.lower().replace(" ", "_") \
    #     / fu['name'].lower().replace(" ", "_").replace(",", "")

    write_dir = Path("write_files") / project.lower().replace(" ", "_") / \
        "food_and_non-alcoholic_beverages_sector_years_151617"
    # dp_names_bg = ["technosphere", "biosphere", "characterization"]
    # dp_names_mo = [
    #     "ecoinvent-parameterization-parameters", "liquid-fuels-kilogram", "implicit-markets", "entso-timeseries",
    # ]
    dp_names_bg = ['technosphere']

    fp = DATA_DIR / "xgboost" / f"tech-technosphere-61.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp)))
    mask_wo_lowinf = get_mask(tindices_ei, dp.data[0], False)
    mask_sign = ~(tflip_ei[mask_wo_lowinf] ^ tdistributions_ei['negative'][mask_wo_lowinf])

    iterations = 5000
    num_parameters = 10000
    random_seeds = np.arange(61, 69)
    Y = []
    X = []
    for random_seed in random_seeds:

        y_ = read_pickle(write_dir / f"mc.tech.xgboost.{iterations}.{random_seed}.pickle")
        Y.append(np.array(y_))

        x_ = []
        # for dp_name in dp_names_bg + dp_names_mo:
        for dp_name in dp_names_bg:
            fp = DATA_DIR / "xgboost" / f"tech-{dp_name}-{random_seed}.zip"
            dp = bwp.load_datapackage(ZipFS(str(fp)))
            if dp_name in dp_names_bg:
                mask = np.ones(len(dp.data[0]), dtype=bool)
            else:
                if dp_name == "entso-timeseries":
                    dp_name = "entso-average"
                mask = read_pickle(write_dir / f"mask.{dp_name}.without_lowinf.params_{num_parameters:.1e}")
            x_temp = deepcopy(dp.data[1][mask])
            x_temp[mask_sign] *= -1
            x_.append(x_temp)

        X.append(np.vstack(x_).T)

    Y = np.hstack(Y)
    X = np.vstack(X)

    split = int(0.2*X.shape[0])
    X_train = X[:-split, :]
    X_test = X[-split:, :]
    Y_train = Y[:-split]
    Y_test = Y[-split:]

    print(X_train.shape, X_test.shape)

    del X, Y

    sigma_Y = np.std(Y_train)
    sigma_X = np.std(X_train, axis=0)

    # Check linearity
    reg = LinearRegression().fit(X_train, Y_train)
    write_pickle(reg, write_dir / "regression.pickle")
    coef = reg.coef_ * sigma_X / sigma_Y
    r2 = sum(coef**2)
    print(r2)

    print(reg.score(X_train, Y_train))
    print(reg.score(X_test, Y_test))

    # 3.1.3. gradient boosting
    # tuning_parameters = dict(
    #     learning_rate=0.15,
    #     gamma=0,
    #     min_child_weight=300,
    #     max_depth=4,
    #     reg_lambda=0,
    #     reg_alpha=0,
    #     n_estimators=100,  # 600,
    #     subsample=0.3,
    #     colsample_bytree=0.2,
    # )
    # t0 = time()
    # xgboost_model = xgboost_indices_base(
    #     Y=Y,
    #     X=X,
    #     tuning_parameters=tuning_parameters,
    #     test_size=0.2,
    #     xgb_model=None,
    #     importance_types=None,  # TODO set default to empty list?
    #     flag_return_xgb_model=True,
    # )
    # t1 = time()
    #
    # print(f"Xgboost training took {t1-t0} seconds")


    # max_depth = 3
    # model = XGBRegressor(
    #     n_estimators=1000,
    #     max_depth=max_depth,
    #     eta=0.1,
    #     subsample=0.2,
    #     colsample_bytree=0.9,
    #     base_score=np.mean(Y_train),
    #     booster='gbtree',
    #     #     tree_method="hist",
    #     #     objective='reg:linear',
    # )
    # eval_set = [(X_train, Y_train), (X_test, Y_test)]
    # model.fit(X_train, Y_train, eval_metric=["error", "rmse"], eval_set=eval_set, verbose=True)
    #
    # score_train = model.score(X_train, Y_train)
    # score_test = model.score(X_test, Y_test)
    # print(f"{score_train:4.3f}  train score")
    # print(f"{score_test:4.3f}  test score")
    #
    # fp_model = write_dir / f"xgboost1_simplistic_depth{max_depth}.pickle"
    # write_pickle(model, fp_model)


    # fu = [act for act in co if "Food" in act['name']][0]
    # demand = {fu: 1}
    # method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    # fu_mapped, pkgs, _ = bd.prepare_lca_inputs(demand=demand, method=method, remapping=False)
    #
    # lca = bc.LCA(
    #     demand=fu_mapped,
    #     data_objs=pkgs,
    #     use_arrays=True,
    #     use_distributions=False,
    #     seed_override=22222000
    # )
    # lca.lci()
    # lca.lcia()
    # scores = [lca.score for _, _ in zip(lca, range(5))]
    # print(scores)
    # print(Y[:5])
    # print("")
