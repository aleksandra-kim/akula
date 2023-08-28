import bw2data as bd
import bw2calc as bc
from pathlib import Path
import sys

PROJECT = "GSA with correlations"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_DIR))

if __name__ == "__main__":

    bd.projects.set_current(PROJECT)
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    co = bd.Database("swiss consumption 1.0")

    # LCIA for average consumption
    option = 'aggregated'
    year = '121314'
    co_average_act_name = f'ch hh average consumption {option}, years {year}'
    hh_average = [act for act in co if co_average_act_name == act['name']]
    assert len(hh_average) == 1
    demand_act = hh_average[0]
    lca = bc.LCA({demand_act: 1}, method)
    lca.lci()
    lca.lcia()
    print("{:8.3f}  {}".format(lca.score, demand_act['name']))