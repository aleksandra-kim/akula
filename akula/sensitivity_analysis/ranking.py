import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from pathlib import Path
import bw2data as bd
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import country_converter as coco
import logging

from ..utils import read_pickle, write_pickle, get_locations_ecoinvent
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


def get_feature_importances_shap_values(shap_values, features, num_inf):
    """
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer.
    Link: https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137
    """
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = np.mean(np.abs(shap_values.values), axis=0)
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances_norm = {feature: importance for importance, feature in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances_norm = {
        feature: importance for feature, importance in
        sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)[:num_inf]
    }
    return feature_importances_norm


def get_influential_shapley(dict_inf, num_inf, iterations, seed, correlations):
    list_inf = sorted(dict_inf.items(), key=lambda item: item[1], reverse=True)[:num_inf]
    where_inf = np.array([element[0] for element in list_inf])

    # Attribute influential inputs to correct input types
    _, indices = get_x_data(iterations, seed, correlations)
    start = 0
    indices_inf = dict()
    for key, inds in indices.items():
        size = len(inds)
        mask = np.logical_and(where_inf >= start, where_inf < start + size)
        where = where_inf[mask] - start
        list_ = list()
        for element in where:
            ind = inds[element]
            list_.append((ind[0], ind[1], dict_inf[element+start]))
        indices_inf[key] = list_
        start += size

    return indices_inf


def get_ranked_list(project, tag, iterations, seed, num_lowinf_lsa, num_inf, correlations):

    directory = GSA_DIR_CORR if correlations else GSA_DIR_INDP
    fp = directory / f"ranking.model_{tag}.{num_inf}.{seed}.{iterations}.csv"

    if fp.exists():
        ranking = pd.read_csv(fp)

    else:
        logging.disable(logging.CRITICAL)
        locations = get_locations_ecoinvent()

        # Compute feature importance values
        shap_values = compute_shap_values(tag, iterations, seed, num_lowinf_lsa, correlations)
        features = np.arange(num_lowinf_lsa)
        feature_importances = get_feature_importances_shap_values(shap_values, features, num_inf)

        # Get top `num_inf` features with highest shapley value scores
        top_features = get_influential_shapley(feature_importances, num_inf, iterations, seed, correlations)

        write_pickle(top_features, directory / f"ranking.indices.model_{tag}.{num_inf}.{seed}.{iterations}.pickle")

        # Assign top feature indices to LCA inputs and save it in a dataframe `ranking`
        bd.projects.set_current(project)

        list_ = list()
        id_ = 1
        for key, inds in top_features.items():
            for ind in inds:
                input_ = bd.get_activity(ind[0])
                output = bd.get_activity(ind[1])
                if key != "characterization":
                    dict1 = {
                        "Rank": "",
                        "ID": id_,
                        "Type": "",
                        "Link": "to",
                        "Name": output['name'],
                        "Reference product": output.get('reference product', ""),
                        "Location": output.get('location', ""),
                        "Categories": input_.get("categories", ""),
                        "Sensitivity index": ind[2],
                    }
                    list_.append(dict1)
                    id_ += 1
                dict2 = {
                    "ID": id_,
                    "Type": key,
                    "Link": "from",
                    "Name": input_['name'],
                    "Reference product": input_.get('reference product', ""),
                    "Location": input_.get("location", ""),
                    "Categories": input_.get("categories", ""),
                    "Sensitivity index": ind[2],
                }
                list_.append(dict2)
                id_ += 1

        ranking = pd.DataFrame.from_records(list_)
        ranking.sort_values(inplace=True, by=["Sensitivity index", "ID"], ascending=False)

        # Polish data in the dataframe for the sake of readability
        ranking["Rank"] = ""
        rank = 1
        for i, row in ranking.iterrows():
            if len(row["Type"]) > 0:
                ranking.at[i, "Rank"] = int(rank)
                ranking.at[i, "Sensitivity index"] = f"{row['Sensitivity index']:5.4e}"
                rank += 1
            else:
                ranking.at[i, "Sensitivity index"] = ""

            location = coco.convert(row["Location"], to="name_short", not_found=row["Location"])
            ranking.at[i, "Location"] = locations.get(location, location)

        ranking.reset_index(inplace=True)
        ranking = ranking[["Rank", "Type", "Link", "Name", "Location", "Categories", "Sensitivity index"]]

        ranking.to_csv(fp, index=False)

    return ranking
