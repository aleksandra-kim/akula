import bw2data as bd
import bw2calc as bc
import bw2io as bi
from pathlib import Path
from gwp_uncertainties import add_bw_method_with_gwp_uncertainties

# Local files
from consumption_model_ch.import_databases import (
    import_exiobase_3,
    import_consumption_db,
)
from consumption_model_ch.consumption_fus import (
    add_consumption_activities,
    add_consumption_categories,
    add_consumption_sectors,
)


def import_all_databases(use_exiobase, add_activities=True):

    path_base = Path('/Users/akim/Documents/LCA_files/')
    year = '151617'

    directory_habe = path_base / 'HABE_2017/'
    fp_ecoinvent_38 = path_base / "ecoinvent_38_cutoff" / "datasets"
    # fp_ecoinvent_38 = "/Users/cmutel/Documents/lca/Ecoinvent/3.8/cutoff/datasets"
    fp_ecoinvent_33 = path_base / "ecoinvent_33_cutoff"/ "datasets"
    fp_exiobase = path_base / "exiobase_381_monetary" / "IOT_2015_pxp"
    # fp_archetypes = path_base / "heia" / "hh_archetypes_weighted_ipcc_091011.csv"

    ei38_name = "ecoinvent 3.8 cutoff"
    ex38_name = "exiobase 3.8.1 monetary"
    co_name = "swiss consumption 1.0"

    if use_exiobase:
        project = "GSA for archetypes with exiobase"
    else:
        project = "GSA for archetypes"
    bd.projects.set_current(project)

    # Import biosphere and ecoinvent databases
    if ei38_name not in bd.databases:
        bi.bw2setup()
        ei = bi.SingleOutputEcospold2Importer(fp_ecoinvent_38, ei38_name)
        ei.apply_strategies()
        assert ei.all_linked
        ei.write_database()

    exclude_databases = [
        'heia',
        'Agribalyse 1.2',
        'Agribalyse 1.3 - {}'.format(ei38_name),
    ]

    # Import exiobase
    if use_exiobase:
        import_exiobase_3(fp_exiobase, ex38_name)
    else:
        exclude_databases.append('exiobase 2.2')

    # Import consumption database
    import_consumption_db(
        directory_habe, co_name, year, fp_ecoinvent_33, exclude_databases, fp_exiobase,
    )

    # Add uncertainties to GWP values
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    if method not in bd.methods:
        add_bw_method_with_gwp_uncertainties()

    # Add functional units
    co = bd.Database(co_name)
    option = 'aggregated'
    if add_activities:
        add_consumption_activities(co_name, option=option,)
        add_consumption_categories(co_name)
        add_consumption_sectors(co_name)

    # LCIA for average consumption
    co_average_act_name = 'ch hh average consumption {}'.format(option)
    hh_average = [act for act in co if co_average_act_name == act['name']]
    assert len(hh_average) == 1
    demand_act = hh_average[0]
    lca = bc.LCA({demand_act: 1}, method)
    lca.lci()
    lca.lcia()
    print("{:8.3f}  {}".format(lca.score, demand_act['name']))

    # LCIA for all Swiss consumption sectors
    sectors = [act for act in co if "sector" in act['name'].lower()]
    for demand_act in sectors:
        lca = bc.LCA({demand_act: 1}, method)
        lca.lci()
        lca.lcia()
        print("{:8.3f}  {}".format(lca.score, demand_act['name']))


if __name__ == "__main__":

    print("Impacts WITHOUT exiobase")
    print("------------------------")
    import_all_databases(False, False)

    print("\n")
    print("Impacts WITH exiobase")
    print("---------------------")
    import_all_databases(True, False)

    # Backup GSA project
    # bi.backup_project_directory(project)
    # # Restore GSA project
    # fp_gsa_project = path_base / "brightway2-project-GSA-backup.16-November-2021-11-50AM.tar.gz"
    # if project not in bd.projects:
    #     bi.restore_project_directory(fp_gsa_project)

