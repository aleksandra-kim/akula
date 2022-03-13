import bw2calc as bc
import bw2data as bd
import bw_processing as bwp

from pathlib import Path
import numpy as np
from fs.zipfs import ZipFS

from gsa_framework.models.life_cycle_assessment_bw25 import LCAModel25
from akula.markets import DATA_DIR

from gsa_framework.sensitivity_analysis.correlations import correlation_coefficients

use_exiobase = False
cutoff = 1e-16


if __name__ == "__main__":

    path_base = Path('/Users/akim/Documents/LCA_files/')
    fp_virtual_markets = DATA_DIR / "virtual-markets.zip"

    if use_exiobase:
        project = "GSA for archetypes with exiobase"
    else:
        project = "GSA for archetypes"
    bd.projects.set_current(project)
    write_dir_project = Path("write_files") / project.lower().replace(" ", "_")
    write_dir_project.mkdir(parents=True, exist_ok=True)

    ei = bd.Database("ecoinvent 3.8 cutoff")

    flows = ('market for diesel,', 'diesel,', 'petrol,', 'market for petrol,')
    liquid_fuels = [x
                    for x in ei
                    if x['unit'] == 'kilogram'
                    and ((any(x['name'].startswith(flow) for flow in flows) or x['name'] == 'market for diesel'))
                    ]
    a = liquid_fuels[0]
    for production in a.production():
        pass

    print(production['properties']['carbon content'])

    consumer = list(a.consumers())[0]
    consumer

    if __name__ == "__main__":
        # write_dir_gsa = write_dir_project / demand_act['name'].lower().replace(" ", "_")
        # write_dir_gsa.mkdir(parents=True, exist_ok=True)
        # model = LCAModel25(demand, method, write_dir_gsa)
        # res = model.get_graph_traversal_params(cutoff=cutoff)
        #
        # # # For the current demand, perform GSA
        # archetypes = [act for act in co if "archetype" in act['name']]
        # for demand_act in archetypes:
        #     print(demand_act['name'])
        #     demand = {demand_act: 1}
        #     write_dir_gsa = write_dir_project / demand_act['name'].lower().replace(" ", "_")
        #     write_dir_gsa.mkdir(parents=True, exist_ok=True)
        #     model = LCAModel25(demand, method, write_dir_gsa)
        #     res = model.get_graph_traversal_params(cutoff=cutoff)


        print()
