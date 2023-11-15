from pathlib import Path
import pickle

import os
os.environ["ENTSOE_API_TOKEN"] = "0d6ea062-f603-43d3-bc60-176159803035"
os.environ["BENTSO_DATA_DIR"] = "/home/aleksandrakim/LCAfiles/bentso_data"

from akula.electricity import (
    create_entsoe_dp,
    compute_low_voltage_ch_lcia,
    plot_lcia_scores,
)

PROJECT = "GSA with correlations"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
FIGURES_DIR = PROJECT_DIR / "figures"


if __name__ == "__main__":
    mc_iterations = 2000
    random_seed = 111111
    options = ["winter", "autumn", "summer", "spring", "nighttime", "daytime", "fitted", "ecoinvent"]

    directory = PROJECT_DIR / "akula" / "data" / "monte_carlo"
    directory.mkdir(exist_ok=True, parents=True)

    results = {}
    for opt in options:

        print(f"Computing {opt} scores")

        fp = directory / f"{opt}.N{mc_iterations}.seed{random_seed}.pickle"
        if fp.exists():
            with open(fp, "rb") as f:
                lcia_scores = pickle.load(f)
        else:
            if opt == "ecoinvent":
                datapackage = None
            else:
                datapackage = create_entsoe_dp(PROJECT_DIR, option=opt, iterations=mc_iterations)
            lcia_scores = compute_low_voltage_ch_lcia(PROJECT, datapackage, iterations=mc_iterations, seed=random_seed)
            if mc_iterations > 20:
                with open(fp, "wb") as f:
                    pickle.dump(lcia_scores, f)
        results[opt] = lcia_scores

    figure = plot_lcia_scores(results)
    figure.show()

    figure.write_image(FIGURES_DIR / "entsoe_seasonal_uncertainties.pdf")
