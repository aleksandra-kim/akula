from pathlib import Path

import os
os.environ["ENTSOE_API_TOKEN"] = "0d6ea062-f603-43d3-bc60-176159803035"
os.environ["BENTSO_DATA_DIR"] = "/home/aleksandrakim/LCAfiles/bentso_data"

from akula.sensitivity_analysis import (
    get_tmask_wo_noninf,
    get_bmask_wo_noninf,
    get_cmask_wo_noninf,
    get_masks_wo_lowinf,
    # get_masks_inf,
    run_mc_simulations_all_inputs,
    run_mc_simulations_wo_noninf,
    run_mc_simulations_wo_lowinf,
    run_mc_simulations_screening,
    train_xgboost_model,
    # get_x_data, get_x_data_v2,
)
from akula.utils import compute_deterministic_score, write_pickle
from akula.monte_carlo import plot_lcia_scores_from_two_cases

PROJECT = "GSA with correlations"
PROJECT_EXIOBASE = "GSA with correlations, exiobase"

PROJECT_DIR = Path(__file__).parent.parent.resolve()
FIGURES_DIR = PROJECT_DIR / "figures"
FP_ECOINVENT = "/home/aleksandrakim/LCAfiles/ecoinvent_38_cutoff/datasets"

# Parameters for GSA
SEED = 222201
CUTOFF = 1e-7
MAX_CALC = 1e18
FACTOR = 10
ITERATIONS_VALIDATION = 2_000
ITERATIONS_SCREENING = 10_000
NUM_LOWINF = 25_000
NUM_INF = 100

GSA_DIR = Path(__file__).parent.parent.resolve() / "data" / "sensitivity-analysis"
SCREENING_DIR = GSA_DIR / "high-dimensional-screening"


if __name__ == "__main__":

    # =========================================================================
    # 0. Setups
    # =========================================================================
    # 0.1 Compute LCIA score offset when exiobase is used
    compute = False  # purely out of impatience
    if compute:
        exiobase_lcia = compute_deterministic_score(PROJECT_EXIOBASE)
        no_exiobase_lcia = compute_deterministic_score(PROJECT)
        exiobase_offset = exiobase_lcia - no_exiobase_lcia
    else:
        exiobase_offset = 703.1540208909953

    # 0.2 Run Monte Carlo simulations when all TECH, BIO and CF inputs vary, including 4 sampling modules
    scores_all = run_mc_simulations_all_inputs(PROJECT, FP_ECOINVENT, ITERATIONS_VALIDATION, SEED)

    # =========================================================================
    # 1. Remove NON-influential inputs
    # - Takes ~25 min for technosphere with cutoff=1e-7, max_calc=1e18
    # - Tweak CUTOFF and MAX_CALC to get the desired number of technosphere inputs based on validation results.
    # =========================================================================
    tmask_wo_noninf = get_tmask_wo_noninf(PROJECT, CUTOFF, MAX_CALC)
    bmask_wo_noninf = get_bmask_wo_noninf(PROJECT)
    cmask_wo_noninf = get_cmask_wo_noninf(PROJECT)

    print()
    print(f"{sum(tmask_wo_noninf):6d} / {len(tmask_wo_noninf):6d} TECH INPUTS after removing NON influential "
                                                                       "with Supply Chain Traversal")
    print(f"{sum(bmask_wo_noninf):6d} / {len(bmask_wo_noninf):6d}  BIO INPUTS after removing NON influential "
                                                                       "with Biosphere Matrix Analysis")
    print(f"{sum(cmask_wo_noninf):6d} / {len(cmask_wo_noninf):6d}   CF INPUTS after removing NON influential "
                                                                       "with Characterization Matrix Analysis\n")

    # Validate results
    num_noninf = sum(tmask_wo_noninf) + sum(bmask_wo_noninf) + sum(cmask_wo_noninf)
    scores_wo_noninf = run_mc_simulations_wo_noninf(
        PROJECT, FP_ECOINVENT, CUTOFF, MAX_CALC, ITERATIONS_VALIDATION, SEED, num_noninf
    )
    figure = plot_lcia_scores_from_two_cases(scores_all, scores_wo_noninf, exiobase_offset)
    figure.write_image(FIGURES_DIR / f"validation_noninf.{num_noninf}.pdf")

    # =========================================================================
    # 2. Remove LOWLY influential inputs with local sensitivity analysis
    # =========================================================================
    # - LSA takes 14h for technosphere, 15 min for biosphere, and seconds for characterization inputs.
    # - Tweak NUM_LOWINF to get the desired number of inputs to be removed based on validation results.
    # =========================================================================
    tmask_wo_lowinf, bmask_wo_lowinf, cmask_wo_lowinf = get_masks_wo_lowinf(
        PROJECT, FACTOR, CUTOFF, MAX_CALC, NUM_LOWINF
    )

    print(f"{sum(tmask_wo_lowinf):6d} / {len(tmask_wo_lowinf):6d} TECH INPUTS after removing LOWLY influential "
                                                                       "with Local Sensitivity Analysis")
    print(f"{sum(bmask_wo_lowinf):6d} / {len(bmask_wo_lowinf):6d}  BIO INPUTS after removing LOWLY influential "
                                                                       "with Local Sensitivity Analysis")
    print(f"{sum(cmask_wo_lowinf):6d} / {len(cmask_wo_lowinf):6d}   CF INPUTS after removing LOWLY influential "
                                                                       "with Local Sensitivity Analysis\n")

    # Validate results
    scores_wo_lowinf = run_mc_simulations_wo_lowinf(
        PROJECT, FP_ECOINVENT, FACTOR, CUTOFF, MAX_CALC, ITERATIONS_VALIDATION, SEED, NUM_LOWINF
    )
    figure = plot_lcia_scores_from_two_cases(scores_all, scores_wo_lowinf, exiobase_offset)
    figure.write_image(FIGURES_DIR / f"validation.wo_lowinf.{NUM_LOWINF}.{SEED}.{ITERATIONS_VALIDATION}.pdf")

    # =========================================================================
    # 3. Run MC for high dimensional screening
    # =========================================================================
    # - Takes 40 min per 5000 MC simulations
    # - Sufficient number of MC simulations is in the order of 2*NUM_LOWINF
    # =========================================================================
    scores_screening = run_mc_simulations_screening(
        PROJECT, FP_ECOINVENT, FACTOR, CUTOFF, MAX_CALC, ITERATIONS_SCREENING, SEED, NUM_LOWINF
    )
    # =========================================================================
    # 4. Remove LOWLY influential inputs based on trained XGBoost model and feature importance
    # =========================================================================
    # tag = "2"
    # model = train_xgboost_model(tag, ITERATIONS_SCREENING, SEED, NUM_LOWINF)
    # tmask_inf, bmask_inf, cmask_inf, pmask_inf = get_masks_inf(
    #     tag, PROJECT, FP_ECOINVENT, FACTOR, CUTOFF, MAX_CALC, ITERATIONS_SCREENING, SEED, NUM_LOWINF, NUM_INF
    # )
    #
    # print(f"{sum(tmask_inf):6d} / {len(tmask_wo_lowinf):6d} TECH INPUTS after removing LOWLY influential "
    #                                                             "with Gradient Boosting")
    # print(f"{sum(bmask_inf):6d} / {len(bmask_wo_lowinf):6d}  BIO INPUTS after removing LOWLY influential "
    #                                                             "with Gradient Boosting")
    # print(f"{sum(cmask_inf):6d} / {len(cmask_wo_lowinf):6d}   CF INPUTS after removing LOWLY influential "
    #                                                             "with Gradient Boosting")
    # print(f"{sum(pmask_inf):6d} / {len(pmask_inf):6d}        PARAMETERS after removing LOWLY influential "
    #                                                             "with Gradient Boosting\n")

    # Validate results

    # =========================================================================
    # 6. Factor prioritization with Shapley values
    # =========================================================================

    # Validate results
