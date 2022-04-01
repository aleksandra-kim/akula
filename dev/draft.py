import numpy as np
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from fs.zipfs import ZipFS

from akula.parameterized_exchanges import DATA_DIR
# from akula.electricity.create_datapackages import create_average_entso_datapackages


if __name__ == "__main__":

    bd.projects.set_current("GSA for archetypes")

    co = bd.Database('swiss consumption 1.0')
    fu = [act for act in co if "ch hh average consumption aggregated, years 151617" == act['name']][0]

    demand = {fu: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    fu_mapped, packages, _ = bd.prepare_lca_inputs(demand=demand, method=method, remapping=False)

    lca = bc.LCA(demand=fu_mapped, data_objs=packages)
    lca.lci()
    lca.lcia()

    print("")

