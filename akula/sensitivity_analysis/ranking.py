import numpy as np
import xgboost as xgb
import shap
from pathlib import Path
import bw2data as bd
from sklearn.model_selection import train_test_split

from .high_dimensional_screening import get_x_data, get_y_scores

GSA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "sensitivity-analysis"
SCREENING_DIR = GSA_DIR / "high-dimensional-screening"
# SCREENING_DIR_CORR.mkdir(exist_ok=True, parents=True)
#
# GSA_DIR_INDEP = Path(__file__).parent.parent.parent.resolve() / "data" / "sensitivity-analysis-independent"
# SCREENING_DIR_INDEP = GSA_DIR_INDEP / "high-dimensional-screening"
# SCREENING_DIR_INDEP.mkdir(exist_ok=True, parents=True)


def compute_shap_values(project, tag, num_inf, iterations, seed, num_lowinf_lsa, test_size=0.2):
    fp = SCREENING_DIR_CORR / f"xgboost_model.{tag}.pickle"
    model = xgb.Booster()
    model.load_model(fp)

    # Read X and Y data
    X, indices = get_x_data(iterations, seed)
    X = X.T
    Y = get_y_scores(iterations, seed, num_lowinf_lsa)
    X_train, _, Y_train, _ = train_test_split(
        X, Y, test_size=test_size, random_state=seed, shuffle=False,
    )
    del X, Y

    dtrain = xgb.DMatrix(X_train, Y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(dtrain)

    shap.plots.bar(shap_values)
    shap.plots.beeswarm(shap_values)

    return shap_values


# def rank_inputs():
#     return ranking
#
#
# def print_ranking(shap_values, num_inf, iterations, seed):
#     ranking = rank_inputs(shap_values, num_inf, iterations, seed)
#     where = np.argsort(shap_values)[:num_inf]
#     print(shap_values[where])

    # bd.projects.set_current(project)
    # for key, inds in indices.items():
    #     if key == "characterization":
    #         for ind in inds:
    #             input_ = bd.get_activity(ind[0])
    #             output = bd.get_activity(ind[0])
    #             print(f"FROM {input_['name']}, {input_.get('location', None)}")
    #             print(f"TO   {output['name']}, {output.get('location', None)}\n")
