from pathlib import Path
import sys
import bw2data as bd

import os
os.environ["ENTSOE_API_TOKEN"] = "0d6ea062-f603-43d3-bc60-176159803035"
os.environ["BENTSO_DATA_DIR"] = "/home/aleksandrakim/LCAfiles/bentso_data"

from akula.markets import (
    generate_markets_datapackage,
    plot_dirichlet_samples,
    plot_dirichlet_entsoe_samples,
    plot_markets_validation,
)
from akula.electricity import generate_entsoe_datapackage

PROJECT = "GSA with correlations"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_DIR))

FIGURES_DIR = PROJECT_DIR / "figures"

FIGURES_DIR_MARKETS = FIGURES_DIR / "dirichlet" / "markets"
FIGURES_DIR_MARKETS.mkdir(parents=True, exist_ok=True)

FIGURES_DIR_ENTSOE = FIGURES_DIR / "dirichlet" / "entsoe"
FIGURES_DIR_ENTSOE.mkdir(parents=True, exist_ok=True)

FIGURES_DIR_EUROPE = FIGURES_DIR / "europe"
FIGURES_DIR_EUROPE.mkdir(parents=True, exist_ok=True)

entsoe = False
markets = False
denmark = False
europe = True

if __name__ == "__main__":
    bd.projects.set_current(PROJECT)

    iterations = 20000
    seed = 111111

    # =========================================================================
    # 1. Validation of Dirichlet sampling
    # =========================================================================
    if entsoe:
        dp_entsoe = generate_entsoe_datapackage("entsoe", iterations, seed)
        dp_dirichlet = generate_markets_datapackage("markets-entsoe", iterations, seed,
                                                    for_entsoe=True, fit_lognormal=False)

        indices = dp_dirichlet.get_resource('markets-entsoe.indices')[0]
        unique_cols = sorted(list(set(indices['col'])))

        for col in unique_cols:
            act = bd.get_activity(col)
            figure = plot_dirichlet_entsoe_samples(col, dp_entsoe, dp_dirichlet)
            figure.write_image(FIGURES_DIR_ENTSOE /
                               f"{col}.{act['name'][:100]}.{act['location']}.png".replace("w/o", "wo"))

    # =========================================================================
    # 2. Plot uncertainties for all markets
    # =========================================================================
    if markets:
        dp_markets = generate_markets_datapackage("markets", iterations, seed)

        indices = dp_markets.get_resource('markets.indices')[0]
        unique_cols = sorted(list(set(indices['col'])))

        for col in unique_cols:
            act = bd.get_activity(col)
            figure = plot_dirichlet_samples(col, dp_markets)
            figure.write_image(FIGURES_DIR_MARKETS /
                               f"{col}.{act['name'][:100]}.{act['location']}.png".replace("w/o", "wo"))

    # =========================================================================
    # 3. Plot uncertainties for high, medium and low voltage markets in Denmark
    # =========================================================================
    if denmark:
        dp_entsoe = generate_entsoe_datapackage("entsoe", iterations, seed)
        dp_dirichlet = generate_markets_datapackage("markets-entsoe", iterations, seed,
                                                    for_entsoe=True, fit_lognormal=False)
        figure = plot_markets_validation(dp_entsoe, dp_dirichlet, location="DK")
        figure.write_image(FIGURES_DIR / "denmark.eps")

    if europe:
        dp_entsoe = generate_entsoe_datapackage("entsoe", iterations, seed)
        dp_dirichlet = generate_markets_datapackage("markets-entsoe", iterations, seed,
                                                    for_entsoe=True, fit_lognormal=False)
        cols = list(set(dp_entsoe.data[0]['col']))
        locations = sorted(set([bd.get_activity(col)["location"] for col in cols]))
        for location in locations:
            print(location)
            if location in ["DK", "CH"]:
                continue
            figure = plot_markets_validation(dp_entsoe, dp_dirichlet, location=location, plot_zoomed=False)
            figure.write_image(FIGURES_DIR_EUROPE / f"{location}.eps")
