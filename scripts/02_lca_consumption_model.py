import bw2data as bd
import bw2calc as bc
from pathlib import Path
import sys

from akula.utils import get_consumption_activity

PROJECT = "GSA with correlations"
PROJECT_EXIOBASE = "GSA with correlations, exiobase"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_DIR))

if __name__ == "__main__":

    use_exiobase = False
    project = PROJECT_EXIOBASE if use_exiobase else PROJECT

    bd.projects.set_current(project)

    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    co = bd.Database("swiss consumption 1.0")

    # LCIA for average consumption
    demand_act = get_consumption_activity()
    lca = bc.LCA({demand_act: 1}, method)
    lca.lci()
    lca.lcia()
    print("{:8.3f}  {}".format(lca.score, demand_act['name']))

    # LCIA for all Swiss consumption sectors
    sectors = sorted([act for act in co if "sector" in act['name'].lower() and "121314" in act['name']])
    for demand_act in sectors:
        lca = bc.LCA({demand_act: 1}, method)
        lca.lci()
        lca.lcia()
        print("{:8.3f}  {}".format(lca.score, demand_act['name']))

# =============================================================================
# LCIA results with ECOINVENT electricity data
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
# LCIA results with ENTSOE electricity data
# =============================================================================
# 1165.333  ch hh average consumption aggregated, years 121314
#    2.152  Alcoholic beverages and tobacco sector, years 121314
#    0.000  Clothing and footwear sector, years 121314
#    0.000  Communication sector, years 121314
#  130.584  Durable goods sector, years 121314
#    0.000  Education sector, years 121314
#    0.000  Fees sector, years 121314
#  246.116  Food and non-alcoholic beverages sector, years 121314
#    0.000  Furnishings, household equipment and routine household maintenance sector, years 121314
#    0.000  Health sector, years 121314
#  385.674  Housing, water, electricity, gas and other fuels sector, years 121314
#    4.231  Miscellaneous goods and services sector, years 121314
#    0.000  Other insurance premiums sector, years 121314
#    0.000  Premiums for life insurance sector, years 121314
#    1.563  Recreation and culture sector, years 121314
#    0.000  Restaurants and hotels sector, years 121314
#  395.014  Transport sector, years 121314

# =============================================================================
# LCIA results with Ecoinvent electricity data and Exiobase
# =============================================================================
# 1836.471  ch hh average consumption aggregated, years 121314
#   10.329  Alcoholic beverages and tobacco sector, years 121314
#  133.666  Clothing and footwear sector, years 121314
#   16.205  Communication sector, years 121314
#  131.463  Durable goods sector, years 121314
#    7.544  Education sector, years 121314
#    9.221  Fees sector, years 121314
#  246.443  Food and non-alcoholic beverages sector, years 121314
#  103.041  Furnishings, household equipment and routine household maintenance sector, years 121314
#   39.060  Health sector, years 121314
#  375.660  Housing, water, electricity, gas and other fuels sector, years 121314
#   47.345  Miscellaneous goods and services sector, years 121314
#   11.057  Other insurance premiums sector, years 121314
#   15.783  Premiums for life insurance sector, years 121314
#  156.869  Recreation and culture sector, years 121314
#   93.498  Restaurants and hotels sector, years 121314
#  439.286  Transport sector, years 121314

# =============================================================================
# LCIA results with ENTSOE electricity data and Exiobase
# =============================================================================
# 1868.487  ch hh average consumption aggregated, years 121314
#   10.317  Alcoholic beverages and tobacco sector, years 121314
#  133.666  Clothing and footwear sector, years 121314
#   16.205  Communication sector, years 121314
#  130.584  Durable goods sector, years 121314
#    7.544  Education sector, years 121314
#    9.221  Fees sector, years 121314
#  246.116  Food and non-alcoholic beverages sector, years 121314
#  103.041  Furnishings, household equipment and routine household maintenance sector, years 121314
#   39.060  Health sector, years 121314
#  409.406  Housing, water, electricity, gas and other fuels sector, years 121314
#   47.338  Miscellaneous goods and services sector, years 121314
#   11.057  Other insurance premiums sector, years 121314
#   15.783  Premiums for life insurance sector, years 121314
#  157.077  Recreation and culture sector, years 121314
#   93.498  Restaurants and hotels sector, years 121314
#  438.573  Transport sector, years 121314
