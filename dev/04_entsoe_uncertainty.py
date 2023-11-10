import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from fs.zipfs import ZipFS
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pickle
from scipy.stats import dirichlet


PROJECT = "GSA with correlations"
PROJECT_DIR = Path(__file__).parent.parent.resolve()


if __name__ == "__main__":

    # figure = plot_electricity_profile()
    # figure.show()

    mc_iterations = 2000
    random_seed = 111111
    results = {}
    options = ["ecoinvent", "winter", "spring", "summer", "autumn", "nighttime", "daytime", "fitted"]
    directory = PROJECT_DIR / "akula" / "data" / "monte_carlo"
    directory.mkdir(exist_ok=True, parents=True)

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
                datapackage = create_entsoe_dp(option=opt, iterations=mc_iterations)
            lcia_scores = compute_lcia(datapackage, iterations=mc_iterations, seed=random_seed)
            if mc_iterations > 20:
                with open(fp, "wb") as f:
                    pickle.dump(lcia_scores, f)
        results[opt] = lcia_scores

    figure = plot_lcia_scores(results, labels=options)
    figure.show()
