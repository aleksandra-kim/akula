from pathlib import Path
import bw2data as bd

import os
os.environ["ENTSOE_API_TOKEN"] = "0d6ea062-f603-43d3-bc60-176159803035"
os.environ["BENTSO_DATA_DIR"] = "/home/aleksandrakim/LCAfiles/bentso_data"

from akula.parameterization import generate_parameterization_datapackage
from akula.combustion import generate_combustion_datapackage
from akula.electricity import generate_entsoe_datapackage
from akula.markets import generate_markets_datapackage
from akula.monte_carlo import compute_scores, plot_sampling_modules
from akula.utils import compute_deterministic_score

PROJECT = "GSA with correlations"
PROJECT_EXIOBASE = "GSA with correlations, exiobase"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
FP_ECOINVENT = "/home/aleksandrakim/LCAfiles/ecoinvent_38_cutoff/datasets"

FIGURES_DIR = PROJECT_DIR / "figures"


if __name__ == "__main__":
    exiobase_lcia = compute_deterministic_score(PROJECT_EXIOBASE)
    no_exiobase_lcia = compute_deterministic_score(PROJECT)

    bd.projects.set_current(PROJECT)

    iterations = 2000
    seed = 222201

    # =========================================================================
    # 1. Generate datapackages for all sampling modules
    # =========================================================================
    _, dp_parameterization = generate_parameterization_datapackage(
        FP_ECOINVENT, "parameterization", iterations, seed
    )
    dp_combustion = generate_combustion_datapackage("combustion", iterations, seed)
    dp_entsoe = generate_entsoe_datapackage("entsoe", iterations, seed)
    dp_markets = generate_markets_datapackage("markets", iterations, seed)

    # =========================================================================
    # 2. Run MC simulations, and compute LCIA WITHOUT any sampling module
    # =========================================================================
    scores_no_sampling = compute_scores(PROJECT, "no_sampling_modules", iterations, seed)

    # =========================================================================
    # 3. Run MC simulations, and compute LCIA WITH sampling modules
    # =========================================================================
    scores_parameterization = compute_scores(PROJECT, "parameterization", iterations, seed, dp_parameterization)
    scores_combustion = compute_scores(PROJECT, "combustion", iterations, seed, dp_combustion)
    scores_entsoe = compute_scores(PROJECT, "entsoe", iterations, seed, dp_entsoe)
    scores_markets = compute_scores(PROJECT, "markets", iterations, seed, dp_markets)

    # =========================================================================
    # 4. Figure 5 in the paper
    # =========================================================================
    exiobase_offset = exiobase_lcia - no_exiobase_lcia
    print(exiobase_offset)
    # exiobase_offset = 703.1540208909953

    figure = plot_sampling_modules(scores_no_sampling, scores_parameterization, exiobase_offset)
    figure.show()
    figure.write_image(FIGURES_DIR / "sampling_parameterization.pdf")

    figure = plot_sampling_modules(scores_no_sampling, scores_combustion, exiobase_offset)
    figure.show()
    figure.write_image(FIGURES_DIR / "sampling_combustion.pdf")

    figure = plot_sampling_modules(scores_no_sampling, scores_entsoe, exiobase_offset)
    figure.show()
    figure.write_image(FIGURES_DIR / "sampling_entsoe.pdf")

    figure = plot_sampling_modules(scores_no_sampling, scores_markets, exiobase_offset)
    figure.show()
    figure.write_image(FIGURES_DIR / "sampling_markets.pdf")
