import bw2data as bd
import bw2calc as bc
from pathlib import Path
import sys

PROJECT = "GSA with correlations"
PROJECT_PATH = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_PATH))

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

    # LCIA for all Swiss consumption sectors
    sectors = sorted([act for act in co if "sector" in act['name'].lower() and str(year) in act['name']])
    for demand_act in sectors:
        lca = bc.LCA({demand_act: 1}, method)
        lca.lci()
        lca.lcia()
        print("{:8.3f}  {}".format(lca.score, demand_act['name']))

# =============================================================================
# LCA results with ORIGINAL electricity data
# =============================================================================
# 1133.317  ch hh average consumption aggregated, years 121314
#    2.164  Alcoholic beverages and tobacco sector, years 121314
#    0.000  Clothing and footwear sector, years 121314
#    0.000  Communication sector, years 121314
#  131.463  Durable goods sector, years 121314
#    0.000  Education sector, years 121314
#    0.000  Fees sector, years 121314
#  246.443  Food and non-alcoholic beverages sector, years 121314
#    0.000  Furnishings, household equipment and routine household maintenance sector, years 121314
#    0.000  Health sector, years 121314
#  351.928  Housing, water, electricity, gas and other fuels sector, years 121314
#    4.238  Miscellaneous goods and services sector, years 121314
#    0.000  Other insurance premiums sector, years 121314
#    0.000  Premiums for life insurance sector, years 121314
#    1.355  Recreation and culture sector, years 121314
#   -0.000  Restaurants and hotels sector, years 121314
#  395.727  Transport sector, years 121314

# =============================================================================
# LCA results with ENTSOE electricity data
# =============================================================================
