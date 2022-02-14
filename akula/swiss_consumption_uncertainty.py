import numpy as np
import pandas as pd
from pathlib import Path
import bw2data as bd
import bw_processing as bwp
from fs.zipfs import ZipFS
from consumption_model_ch.utils import get_habe_filepath

#  Local files
from .sensitivity_analysis import get_mask

DATA_DIR = Path(__file__).parent.resolve() / "data"

KONSUMGUETER_DICT = {
    "anzahlneuwagen11": "cg_nonewcars",
    "anzahlgebrauchtwagen11": "cg_nousedcars",
    "anzahlmotorraeder11": "cg_nomotorbikes",
    "anzahlfahrraeder11": "cg_nobicycles",
    "anzahltiefkuehler11": "cg_nofreezers",
    "anzahlgeschirrspueler11": "cg_nodishwashers",
    "anzahlwaschmaschinen11": "cg_nowashmachines",
    "anzahlwaeschetrockner11": "cg_nodriers",
    "anzahlroehrenfernseher11": "cg_nocrttvs",
    "anzahllcdfernseher11": "cg_nolcdtvs",
    "anzahlparabolantennen11": "cg_nosat",
    "anzahlvideokameras11": "cg_nocams",
    "anzahlvideorecorder11": "cg_novideorecs",
    "anzahlspielkonsolen11": "cg_novieogames",
    "anzahldesktopcomputer11": "cg_nodesktoppcs",
    "anzahllaptopcomputer11": "cg_nolaptops",
    "anzahldrucker11": "cg_noprinters",
    "anzahlmobiltelefone11": "cg_nomobilephones",
    "anzahlmp3player11": "cg_nomp3players",
    "anzahlgpsgeraete11": "cg_nogps",
}


def get_household_data(indices, co_name="swiss consumption 1.0"):
    # 1. Get some metadata from the consumption database
    co = bd.Database(co_name)
    year_habe = co.metadata['year_habe']
    dir_habe = co.metadata['dir_habe']

    # 2. Extract total demand from HABE
    path_ausgaben = get_habe_filepath(dir_habe, year_habe, 'Ausgaben')
    path_mengen = get_habe_filepath(dir_habe, year_habe, 'Mengen')
    path_konsumgueter = get_habe_filepath(dir_habe, year_habe, 'Konsumgueter')

    # change codes to be consistent with consumption database and Andi's codes
    ausgaben = pd.read_csv(path_ausgaben, sep='\t')
    mengen = pd.read_csv(path_mengen, sep='\t')
    konsumgueter = pd.read_csv(path_konsumgueter, sep='\t')
    ausgaben.columns = [col.lower() for col in ausgaben.columns]
    mengen.columns = [col.lower() for col in mengen.columns]
    konsumgueter.columns = [col.lower() for col in konsumgueter.columns]
    codes_co_db = sorted([act['code'] for act in co])
    columns_a = ausgaben.columns.values
    columns_m = [columns_a[0]]
    codes_m = []
    for code_a in columns_a[1:]:
        code_m = code_a.replace('a', 'm')
        if code_m in codes_co_db:
            columns_m.append(code_m)
            codes_m.append(code_m)
        else:
            columns_m.append(code_a)
    ausgaben.columns = columns_m
    # Replace ausgaben data with mengen data
    for code_m in codes_m:
        ausgaben[code_m] = mengen[code_m]
    # Add konsumgueter data
    columns_k = []
    for code_k in konsumgueter.columns:
        code = KONSUMGUETER_DICT.get(code_k, code_k).lower()
        columns_k.append(code)
    konsumgueter.columns = columns_k
    ausgaben = pd.merge(ausgaben, konsumgueter, on="haushaltid")

    data = np.zeros((0, len(ausgaben)))
    for inds in indices:
        input_code = bd.get_activity(inds[0])['code']
        try:
            data_current = ausgaben[input_code].values
            data = np.vstack([data, data_current])
        except:
            print(f"Cannot find {input_code} in household data")
    return data


def generate_uncertainty_in_households_fu_datapackage():
    bd.projects.set_current("GSA for archetypes")
    co = bd.Database("swiss consumption 1.0")
    fu = [act for act in co if "average consumption" in act['name']][0]

    dp_co = bd.Database("swiss consumption 1.0").datapackage()
    co_indices = dp_co.get_resource("swiss_consumption_1.0_technosphere_matrix.indices")[0]
    co_flip = dp_co.get_resource("swiss_consumption_1.0_technosphere_matrix.flip")[0]

    indices = [
        (exc.input.id, fu.id) for exc in fu.exchanges()
        if exc['type'] != 'production' and 'mx' not in exc.input['code']
    ]
    mask = get_mask(co_indices, indices)
    data = get_household_data(indices)
    flip = co_flip[mask]
    indices = np.array(indices, dtype=[('row', '<i4'), ('col', '<i4')])

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / "households-fus-uncertainty.zip"), write=True),
        name="households-fus-uncertainty",
        # set seed to have reproducible (though not sequential) sampling
        seed=42,
    )
    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=data,
        indices_array=indices,
        name="households-fus-uncertainty",
        flip_array=flip,
    )
    dp.finalize_serialization()


if __name__ == "__main__":
    generate_uncertainty_in_households_fu_datapackage()
