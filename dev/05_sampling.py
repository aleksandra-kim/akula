from pathlib import Path

import os
os.environ["ENTSOE_API_TOKEN"] = "0d6ea062-f603-43d3-bc60-176159803035"
os.environ["BENTSO_DATA_DIR"] = "/home/aleksandrakim/LCAfiles/bentso_data"

PROJECT = "GSA with correlations"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
FILEPATH_ECOINVENT = "/Users/akim/Documents/LCA_files/ecoinvent_38_cutoff/datasets"


if __name__ == "__main__":
    iterations = 25000

    print("")

