import numpy as np
from pathlib import Path
from fs.zipfs import ZipFS
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from copy import deepcopy
from gsa_framework.utils import read_pickle, write_pickle

# Local files
from akula.sensitivity_analysis.local_sensitivity_analysis import (
    get_mask, get_tindices_wo_noninf, DATA_DIR
)


if __name__ == "__main__":

    project = 'GSA for archetypes'
    bd.projects.set_current(project)

    dp = bwp.load_datapackage(ZipFS(str(DATA_DIR / "entso-timeseries.zip")))
    dp_data_tech = dp.get_resource('liquid-fuels-tech.data')[0]
    dp_indices_tech = dp.get_resource('liquid-fuels-tech.indices')[0]