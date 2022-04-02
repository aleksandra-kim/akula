import numpy as np
from pathlib import Path
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from fs.zipfs import ZipFS

from akula.parameterized_exchanges import DATA_DIR
# from akula.electricity.create_datapackages import create_average_entso_datapackages


if __name__ == "__main__":

    project = "GSA for archetypes"
    bd.projects.set_current(project)

    co = bd.Database('swiss consumption 1.0')
    fu = [act for act in co if "ch hh average consumption aggregated, years 151617" == act['name']][0]
    demand = {fu: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")

    fu_mapped, packages, _ = bd.prepare_lca_inputs(demand=demand, method=method, remapping=False)

    fp_flocal_sa = DATA_DIR / "local-sa-1e+01-liquid-fuels-kilogram.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp_flocal_sa)))

    tindices = dp.get_resource('local-sa-liquid-fuels-kilogram-tech.indices')[0]
    tdata = dp.get_resource('local-sa-liquid-fuels-kilogram-tech.data')[0]
    bindices = dp.get_resource('local-sa-liquid-fuels-kilogram-bio.indices')[0]
    bdata = dp.get_resource('local-sa-liquid-fuels-kilogram-bio.data')[0]

    lca = bc.LCA(demand=fu_mapped, data_objs=packages, use_distributions=False)
    lca.lci()
    lca.lcia()
    print("\n--> data_objs=packages, use_distributions=False")
    print([(lca.score, next(lca)) for _ in range(5)])

    lca = bc.LCA(demand=fu_mapped, data_objs=packages+[dp], use_distributions=False, use_arrays=True)
    lca.lci()
    lca.lcia()
    print("\n--> data_objs=packages+[dp], use_distributions=False")
    print([(lca.score, next(lca)) for _ in range(5)])

