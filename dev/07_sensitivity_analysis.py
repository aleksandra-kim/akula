from pathlib import Path
import bw2data as bd

import os
os.environ["ENTSOE_API_TOKEN"] = "0d6ea062-f603-43d3-bc60-176159803035"
os.environ["BENTSO_DATA_DIR"] = "/home/aleksandrakim/LCAfiles/bentso_data"

from akula.sensitivity_analysis import (
    get_tmask_wo_noninf,
    get_bmask_wo_noninf,
    get_cmask_wo_noninf,
    get_tmask_wo_lowinf,
    get_bmask_wo_lowinf,
    get_cmask_wo_lowinf,
    get_all_lowinf_scores,
)

PROJECT = "GSA with correlations"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
FIGURES_DIR = PROJECT_DIR / "figures"

# Parameters for supply chain traversal
CUTOFF = 1e-7
MAX_CALC = 1e18
FACTOR = 10


if __name__ == "__main__":

    bd.projects.set_current(PROJECT)

    # =========================================================================
    # 1. Remove NON-influential inputs
    # =========================================================================
    tmask_wo_noninf = get_tmask_wo_noninf(PROJECT, CUTOFF, MAX_CALC)  # takes ~25 min for cutoff=1e-9, max_calc=1e20
    bmask_wo_noninf = get_bmask_wo_noninf(PROJECT)
    cmask_wo_noninf = get_cmask_wo_noninf(PROJECT)

    print(f"{sum(tmask_wo_noninf):6d} / {len(tmask_wo_noninf):6d} TECH inputs after removing NON influential "
                                                                       f"with Supply Chain Traversal")
    print(f"{sum(bmask_wo_noninf):6d} / {len(bmask_wo_noninf):6d}  BIO inputs after removing NON influential "
                                                                       f"with Biosphere Matrix Analysis")
    print(f"{sum(cmask_wo_noninf):6d} / {len(cmask_wo_noninf):6d}   CF inputs after removing NON influential "
                                                                       f"with Characterization Matrix Analysis\n")

    # =========================================================================
    # 2. Remove LOWLY influential inputs with local sensitivity analysis
    # =========================================================================
    tscores, bscores, cscores = get_all_lowinf_scores(PROJECT, FACTOR, CUTOFF, MAX_CALC)
    print("done")
    # tmask_wo_lowinf = get_tmask_wo_lowinf(PROJECT, FACTOR, CUTOFF, MAX_CALC)
    # bmask_wo_lowinf = get_bmask_wo_lowinf(PROJECT)
    # cmask_wo_lowinf = get_cmask_wo_lowinf(PROJECT)
    #
    # print(f"{sum(tmask_wo_lowinf):6d} / {len(tmask_wo_lowinf):6d} TECH inputs after removing LOWLY influential "
    #       f"with Local Sensitivity Analysis")
    # print(f"{sum(bmask_wo_lowinf):6d} / {len(bmask_wo_lowinf):6d}  BIO inputs after removing LOWLY influential "
    #       f"with Local Sensitivity Analysis")
    # print(f"{sum(cmask_wo_lowinf):6d} / {len(cmask_wo_lowinf):6d}   CF inputs after removing LOWLY influential "
    #       f"with Local Sensitivity Analysis\n")


    # =========================================================================
    # 3. Check model linearity
    # =========================================================================

    # =========================================================================
    # 4. Factor fixing with XGBoost
    # =========================================================================

    # =========================================================================
    # 5. Factor prioritization with Shapley values
    # =========================================================================
