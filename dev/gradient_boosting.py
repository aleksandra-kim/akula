import numpy as np
from pathlib import Path
from fs.zipfs import ZipFS
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from gsa_framework.utils import read_pickle

from akula.combustion import DATA_DIR

if __name__ == "__main__":

    project = 'GSA for archetypes'
    bd.projects.set_current(project)

    co = bd.Database('swiss consumption 1.0')
    fu = [act for act in co if "ch hh average consumption aggregated, years 151617" == act['name']][0]

    write_dir = Path("write_files") / project.lower().replace(" ", "_") \
        / fu['name'].lower().replace(" ", "_").replace(",", "")

    dp_names_bg = ["technosphere", "biosphere", "characterization"]
    dp_names_mo = ["implicit-markets", "liquid-fuels-kilogram", "entso-timeseries"]  # "ecoinvent-parameterization"

    iterations = 25000
    num_parameters = 25000
    random_seeds = [43]  # [43, 44, 45, 46]
    Y = []
    X = []
    for random_seed in random_seeds:

        y_ = read_pickle(write_dir / f"mc.xgboost.{iterations}.{random_seed}.pickle")
        Y.append(np.array(y_))

        x_ = []
        for dp_name in dp_names_bg + dp_names_mo:
            fp = DATA_DIR / "xgboost" / f"{dp_name}-{random_seed}.zip"
            dp = bwp.load_datapackage(ZipFS(str(fp)))
            if dp_name not in dp_names_mo:
                mask = np.ones(len(dp.data[0]), dtype=bool)
            else:
                if dp_name == "entso-timeseries":
                    dp_name = "entso-average"
                mask = read_pickle(write_dir / f"mask.{dp_name}.without_lowinf.params_{num_parameters:.1e}")
            x_.append(dp.data[1][mask])

        X.append(np.vstack(x_).T)

    Y = np.hstack(Y)
    X = np.vstack(X)

    print(X.shape)

    # 3.1.3. Check model linearity
    print("")
