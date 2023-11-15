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


if __name__ == "__main__":
    mc_iterations = 2000
    random_seed = 111111
    options = ["ecoinvent", "winter", "spring", "summer", "autumn", "nighttime", "daytime", "fitted"]

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

    figure = plot_lcia_scores(results, labels=options)
    figure.show()
