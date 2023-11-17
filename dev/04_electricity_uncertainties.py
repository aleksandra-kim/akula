from pathlib import Path

import os
os.environ["ENTSOE_API_TOKEN"] = "0d6ea062-f603-43d3-bc60-176159803035"
os.environ["BENTSO_DATA_DIR"] = "/home/aleksandrakim/LCAfiles/bentso_data"

from akula.electricity import (
    compute_scores,
    plot_entsoe_seasonal,
    plot_entsoe_ecoinvent,
)

PROJECT = "GSA with correlations"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
FIGURES_DIR = PROJECT_DIR / "figures"


if __name__ == "__main__":
    iterations = 2000
    seed = 111111

    directory = PROJECT_DIR / "akula" / "data" / "monte-carlo" / "electricity-uncertainties"
    directory.mkdir(exist_ok=True, parents=True)

    options = ["winter", "autumn", "summer", "spring", "nighttime", "daytime", "fitted", "entsoe", "ecoinvent"]
    results = compute_scores(directory, PROJECT, PROJECT_DIR, options, iterations, seed)

    # Figure 1 in the paper
    figure = plot_entsoe_ecoinvent(PROJECT, PROJECT_DIR, results["ecoinvent"], results["entsoe"])
    figure.show()
    figure.write_image(FIGURES_DIR / "ecoinvent_entsoe_uncertainties.pdf")

    # Figure 1 in the SI
    figure = plot_entsoe_seasonal(results)
    figure.show()
    figure.write_image(FIGURES_DIR / "entsoe_seasonal_uncertainties.pdf")
