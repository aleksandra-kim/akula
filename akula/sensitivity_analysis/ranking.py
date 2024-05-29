import numpy as np
import xgboost as xgb
import shap
from pathlib import Path
import bw2data as bd
from sklearn.model_selection import train_test_split

from ..utils import read_pickle, write_pickle
from .high_dimensional_screening import get_x_data, get_y_scores

GSA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "sensitivity-analysis"
GSA_DIR_CORR = GSA_DIR / "correlated"
GSA_DIR_INDP = GSA_DIR / "independent"
SCREENING_DIR = GSA_DIR / "high-dimensional-screening"
SCREENING_DIR_CORR = SCREENING_DIR / "correlated"
SCREENING_DIR_INDP = SCREENING_DIR / "independent"


def compute_shap_values(tag, iterations, seed, num_lowinf_lsa, correlations, test_size=0.2):

    gsa_directory = GSA_DIR_CORR if correlations else GSA_DIR_INDP

    fp_shapley = gsa_directory / f"shapley.model_{tag}.{num_lowinf_lsa}.{seed}.{iterations}"

    if fp_shapley.exists():
        shap_values = read_pickle(fp_shapley)

    else:
        screening_directory = SCREENING_DIR_CORR if correlations else SCREENING_DIR_INDP
        fp_model = screening_directory / f"xgboost_model.{tag}.pickle"
        model = xgb.Booster()
        model.load_model(fp_model)

        # Read X and Y data
        X, indices = get_x_data(iterations, seed, correlations)
        X = X.T
        Y = get_y_scores(iterations, seed, num_lowinf_lsa, correlations)
        X_train, _, Y_train, _ = train_test_split(
            X, Y, test_size=test_size, random_state=seed, shuffle=False,
        )
        del X, Y

        dtrain = xgb.DMatrix(X_train, Y_train)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(dtrain)

        write_pickle(shap_values, fp_shapley)

    return shap_values


def plot_shap_values(shap_values):
    shap.plots.bar(shap_values)
    shap.plots.beeswarm(shap_values)


def get_ranked_list(project, tag, iterations, seed, num_lowinf_lsa, correlations):

    shap_values = compute_shap_values(tag, iterations, seed, num_lowinf_lsa, correlations)

    bd.projects.set_current(project)

    ranking = []

    return ranking


# def rank_inputs():
#     return ranking
#
#
# def print_ranking(shap_values, num_inf, iterations, seed):
#     ranking = rank_inputs(shap_values, num_inf, iterations, seed)
#     where = np.argsort(shap_values)[:num_inf]
#     print(shap_values[where])

    #
    # for key, inds in indices.items():
    #     if key == "characterization":
    #         for ind in inds:
    #             input_ = bd.get_activity(ind[0])
    #             output = bd.get_activity(ind[0])
    #             print(f"FROM {input_['name']}, {input_.get('location', None)}")
    #             print(f"TO   {output['name']}, {output.get('location', None)}\n")
