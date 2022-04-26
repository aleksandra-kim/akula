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
    get_mask, get_tindices_wo_noninf,
)


if __name__ == "__main__":

    project = 'GSA for archetypes'
    bd.projects.set_current(project)
    const_factor = 10
    ctff = 1e-8  # Cutoff for contribution analysis
    mclc = 1e18  # Maximum number of computations for supply chain traversal

    # Setups
    ########

    co = bd.Database('swiss consumption 1.0')
    fu = [act for act in co if "ch hh average consumption aggregated, years 151617" == act['name']][0]

    write_dir = Path("write_files") / project.lower().replace(" ", "_") \
                / fu['name'].lower().replace(" ", "_").replace(",", "")
    write_dir_sct = write_dir / "supply_chain_traversal"
    write_dir_sct.mkdir(exist_ok=True, parents=True)

    demand = {fu: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    fu_mapped, pkgs, _ = bd.prepare_lca_inputs(demand=demand, method=method, remapping=False)
    lca = bc.LCA(demand=fu_mapped, data_objs=pkgs)
    lca.lci()
    lca.lcia()
    static_score = deepcopy(lca.score)
    print(static_score)

    # Get all relevant data
    ei = bd.Database('ecoinvent 3.8 cutoff').datapackage()

    # Technosphere
    tei = ei.filter_by_attribute('matrix', 'technosphere_matrix')
    tindices_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.indices')[0]
    tdata_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.data')[0]
    tflip_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.flip')[0]
    tdistributions_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.distributions')[0]

    # STEP 1: Remove non influential with contribution analysis
    ############################################################

    # Step 1.1 Technosphere & Supply chain traversal
    ctff =
    fp_sct = write_dir_sct / f"sct.cutoff_{ctff:.0e}.maxcalc_{mclc:.0e}.pickle"
    if fp_sct.exists():
        tindices_wo_noninf = read_pickle(fp_sct)
    else:
        tindices_wo_noninf = get_tindices_wo_noninf(lca, ctff, mclc)
        write_pickle(tindices_wo_noninf, fp_sct)















    fp_tmask_wo_noninf = write_dir / f"mask.tech.without_noninf.sct.cutoff_{ctff:.0e}.maxcalc_{mclc:.0e}.pickle"
    if fp_tmask_wo_noninf.exists():
        tmask_wo_noninf = read_pickle(fp_tmask_wo_noninf)
    else:
        tmask_wo_noninf = get_mask(tindices_ei, tindices_wo_noninf)
        write_pickle(tmask_wo_noninf, fp_tmask_wo_noninf)