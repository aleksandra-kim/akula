import numpy as np
import shap
from pathlib import Path

from ..utils import read_pickle, write_pickle
from .high_dimensional_screening import get_x_data

GSA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "sensitivity-analysis"
SCREENING_DIR = GSA_DIR / "high-dimensional-screening"


def compute_shap_values(tag, iterations, seed, num_inf):
    fp = SCREENING_DIR / f"xgboost.{tag}.pickle"
    model = read_pickle(fp)
    explainer = shap.TreeExplainer(model)
    X, indices = get_x_data(iterations, seed)
    shap_values = explainer.shap_values(X)
    where = np.argsort(shap_values)[:num_inf]
    print(shap_values[where])
    print(indices[where])
