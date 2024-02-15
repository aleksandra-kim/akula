from pathlib import Path
import bw2data as bd

import os
os.environ["ENTSOE_API_TOKEN"] = "0d6ea062-f603-43d3-bc60-176159803035"
os.environ["BENTSO_DATA_DIR"] = "/home/aleksandrakim/LCAfiles/bentso_data"

from akula.sensitivity_analysis import get_tmask_wo_noninf, get_bmask_wo_noninf, get_cmask_wo_noninf

PROJECT = "GSA with correlations"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
FIGURES_DIR = PROJECT_DIR / "figures"

# Parameters for supply chain traversal
CUTOFF = 1e-9
MAX_CALC = 1e20


if __name__ == "__main__":

    bd.projects.set_current(PROJECT)

    # =========================================================================
    # 1. Remove non-influential inputs
    # =========================================================================
    tmask_wo_noninf = get_tmask_wo_noninf(PROJECT, CUTOFF, MAX_CALC)
    bmask_wo_noninf = get_bmask_wo_noninf(PROJECT)
    cmask_wo_noninf = get_cmask_wo_noninf(PROJECT)

    print(f"{sum(tmask_wo_noninf):6d} / {len(tmask_wo_noninf):6d} TECH inputs after removing non-influential with SCT")
    print(f"{sum(bmask_wo_noninf):6d} / {len(bmask_wo_noninf):6d}  BIO inputs after removing non-influential with MAT")
    print(f"{sum(cmask_wo_noninf):6d} / {len(cmask_wo_noninf):6d}   CF inputs after removing non-influential with MAT")

    # =========================================================================
    # 2. Remove lowly influential inputs with local sensitivity analysis
    # =========================================================================

    # =========================================================================
    # 3. Check model linearity
    # =========================================================================

    # =========================================================================
    # 4. Factor fixing with XGBoost
    # =========================================================================

    # =========================================================================
    # 5. Factor prioritization with Shapley values
    # =========================================================================
