import bw2data as bd
import bw2io as bi
from pathlib import Path
import warnings
import sys
# from gwp_uncertainties import add_bw_method_with_gwp_uncertainties  # TODO fix packaging here
from consumption_model_ch.import_databases import (
    import_exiobase_3,
    import_consumption_db,
)
from consumption_model_ch.consumption_fus import (
    add_consumption_activities,
    add_consumption_categories,
    add_consumption_sectors,
)

PROJECT = "GSA with correlations"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_DIR))


def import_databases(use_exiobase=False, year='121314'):
    path_base = Path('/home/aleksandrakim/Documents/lca_files')

    directory_habe = path_base / 'HABE_2017/'
    fp_ecoinvent_38 = path_base / "ecoinvent_38_cutoff" / "datasets"
    fp_ecoinvent_33 = path_base / "ecoinvent_33_cutoff" / "datasets"
    fp_exiobase = path_base / "exiobase_381_monetary" / "IOT_2015_pxp"

    ei38_name = "ecoinvent 3.8 cutoff"
    ex38_name = "exiobase 3.8.1 monetary"
    co_name = "swiss consumption 1.0"

    if use_exiobase:
        bd.projects.set_current("GSA with correlations, exiobase")
    else:
        bd.projects.set_current(PROJECT)

    # Import biosphere
    bi.bw2setup()

    # Import ecoinvent
    if ei38_name not in bd.databases:
        ei = bi.SingleOutputEcospold2Importer(fp_ecoinvent_38, ei38_name)
        ei.apply_strategies()
        ei.write_database()

    # # Add uncertainties to GWP values
    # method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    # if method not in bd.methods:
    #     add_bw_method_with_gwp_uncertainties()

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

    # Import swiss consumption database
    if co_name not in bd.databases:
        import_consumption_db(
            directory_habe, year, co_name, fp_ecoinvent_33, exclude_databases, fp_exiobase,
        )
        # Add functional units
        option = 'aggregated'
        add_consumption_activities(co_name, year, option=option)
        add_consumption_categories(co_name)
        add_consumption_sectors(co_name, year)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    imprt = False
    if imprt:
        import_databases()

    # Backup GSA project
    backup = True
    if backup:
        bi.backup_project_directory(PROJECT)

    print(PROJECT_DIR)

    # Restore GSA project
    restore_ecoinvent = False
    if restore_ecoinvent:
        fp_gsa_project = PROJECT_DIR / "akula" / "data" / "GSA-with-correlations-backup.electricity-ecoinvent.tar.gz"
        bi.restore_project_directory(fp_gsa_project)

    restore_entsoe = False
    if restore_entsoe:
        fp_gsa_project = PROJECT_DIR / "akula" / "data" / "GSA-with-correlations-backup.electricity-entsoe.tar.gz"
        bi.restore_project_directory(fp_gsa_project)
