import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from pypardiso import spsolve
from pathlib import Path
from copy import deepcopy
from gsa_framework.utils import read_pickle, write_pickle
from fs.zipfs import ZipFS

# Local files
from akula.sensitivity_analysis.local_sa import *
from akula.markets import DATA_DIR


# def get_tresource_kind(kind, dp_ei, dp_co):
#     return np.hstack(
#         [
#             dp_ei.get_resource(f'ecoinvent_3.8_cutoff_technosphere_matrix.{kind}')[0],
#             dp_co.get_resource(f'swiss_consumption_1.0_technosphere_matrix.{kind}')[0]
#         ]
#     )


def run_tlocal_sa(
        matrix_type,
        func_unit_mapped,
        packages,
        tech_indices,
        tech_data,
        tech_has_uncertainty,
        mask_tech_without_noninf,
        flip_tech,
        factors,
        directory,
        tag,
):
    for i, factor in enumerate(factors):
        fp_factor = directory / f"local_sa.{tag}.factor_{factor:.0e}.pickle"
        if fp_factor.exists():
            local_sa_current = read_pickle(fp_factor)
        else:
            local_sa_current = run_local_sa(
                matrix_type,
                func_unit_mapped,
                packages,
                tech_indices,
                tech_data,
                tech_has_uncertainty,
                mask_tech_without_noninf,
                flip_tech,
                factor,
            )
            write_pickle(local_sa_current, fp_factor)
        if i == 0:
            local_sa_results = deepcopy(local_sa_current)
        else:
            local_sa_results = {
                k: np.hstack([local_sa_results[k], local_sa_current[k]]) for k in local_sa_results.keys()
            }
    return local_sa_results


project = 'GSA for archetypes'
bd.projects.set_current(project)
const_factor = 10
ctff = 1e-6  # Cutoff for contribution analysis
mclc = 1e16  # Maximum number of computations for supply chain traversal


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
co = bd.Database('swiss consumption 1.0').datapackage()
cf = bd.Method(method).datapackage()

# Technosphere
tei = ei.filter_by_attribute('matrix', 'technosphere_matrix')
# tco = co.filter_by_attribute('matrix', 'technosphere_matrix')

# tindices = get_tresource_kind('indices', tei, tco)
# tdata = get_tresource_kind('data', tei, tco)
# tflip = get_tresource_kind('flip', tei, tco)
tindices_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.indices')[0]
tdata_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.data')[0]
tflip_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.flip')[0]
tdistributions_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.distributions')[0]
# tindices_co = tco.get_resource('swiss_consumption_1.0_technosphere_matrix.indices')[0]

# Biosphere
bei = ei.filter_by_attribute('matrix', 'biosphere_matrix')
bindices = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.indices')[0]
bdata = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.data')[0]
bdistributions = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.distributions')[0]

# Characterization
cindices = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.indices')[0]
cdata = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.data')[0]
cdistributions = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.distributions')[0]

# # Get technosphere uncertainty boolean array
# has_uncertainty_ei = tdistributions_ei['uncertainty_type'] >= 2
# has_uncertainty_dict = {}
# for act in bd.Database('swiss consumption 1.0'):
#     exchanges = list(act.exchanges())
#     col = lca.dicts.activity[act.id]
#     for exc in exchanges:
#         if exc.get('has_uncertainty'):
#             row = lca.dicts.activity[exc.input.id]
#             has_uncertainty_dict[(exc.input.id, act.id)] = True
# has_uncertainty_co = np.array([has_uncertainty_dict.get(tuple(ids), False) for ids in tindices_co])

# has_uncertainty_tech = np.hstack(
#     [
#         has_uncertainty_ei,
#         has_uncertainty_co,
#     ]
# )

# STEP 1: Remove non influential with contribution analysis
############################################################

# Step 1.1 Technosphere & Supply chain traversal
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

# Step 1.2 Biosphere
bindices_wo_noninf = get_bindices_wo_noninf(lca)
fp_bmask_wo_noninf = write_dir / "mask.bio.without_noninf.pickle"
if fp_bmask_wo_noninf.exists():
    bmask_wo_noninf = read_pickle(fp_bmask_wo_noninf)
else:
    bmask_wo_noninf = get_mask(bindices, bindices_wo_noninf)
    write_pickle(bmask_wo_noninf, fp_bmask_wo_noninf)

# Step 1.3 Characterization
cindices_wo_noninf = get_cindices_wo_noninf(lca)
fp_cmask_wo_noninf = write_dir / "mask.cf.without_noninf.pickle"
if fp_cmask_wo_noninf.exists():
    cmask_wo_noninf = read_pickle(fp_cmask_wo_noninf)
else:
    cmask_wo_noninf = get_mask(cindices, cindices_wo_noninf)
    write_pickle(cmask_wo_noninf, fp_cmask_wo_noninf)


# STEP 2: Run local SA
######################

# 2.1.1 Technosphere
fp_tlocal_sa = write_dir / f"local_sa.tech.cutoff_{ctff:.0e}.maxcalc_{mclc:.0e}.pickle"
if fp_tlocal_sa.exists():
    tlocal_sa = read_pickle(fp_tlocal_sa)
else:
    tlocal_sa = run_tlocal_sa(
        "technosphere",
        fu_mapped,
        pkgs,
        tindices_ei,
        tdata_ei,
        tdistributions_ei,
        tmask_wo_noninf,
        tflip_ei,
        [1/const_factor, const_factor],
        write_dir,
        "tech.cutoff_{cutoff:.0e}.maxcalc_{max_calc:.0e}",
    )
    write_pickle(tlocal_sa, fp_tlocal_sa)

# 2.2.1 Biosphere
fp_blocal_sa = write_dir / f"local_sa.bio.pickle"
if fp_blocal_sa.exists():
    blocal_sa = read_pickle(fp_blocal_sa)
else:
    blocal_sa = run_local_sa(
        "biosphere",
        fu_mapped,
        pkgs,
        bindices,
        bdata,
        bdistributions,
        bmask_wo_noninf,
        None,
        const_factor,
    )
    write_pickle(blocal_sa, fp_blocal_sa)

# 2.3.1 Characterization
fp_clocal_sa = write_dir / f"local_sa.cf.pickle"
if fp_clocal_sa.exists():
    clocal_sa = read_pickle(fp_clocal_sa)
else:
    clocal_sa = run_local_sa(
        "characterization",
        fu_mapped,
        pkgs,
        cindices,
        cdata,
        cdistributions,
        cmask_wo_noninf,
        None,
        const_factor,
    )
    write_pickle(clocal_sa, fp_clocal_sa)

# 2.1.2, 2.1.4 Technosphere - Generic markets and carbon
fp_generic_markets = DATA_DIR / "generic-markets.zip"
gms = bwp.load_datapackage(ZipFS(fp_generic_markets))
gindices = gms.get_resource("generic markets.indices")[0]
gmask = get_mask(tindices_ei, gindices)

fp_glocal_sa = write_dir / f"local_sa.generic_markets.pickle"
if fp_glocal_sa.exists():
    local_sa = read_pickle(fp_glocal_sa)
else:
    tlocal_sa = run_tlocal_sa(
        "technosphere",
        fu_mapped,
        pkgs,
        tindices_ei,
        tdata_ei,
        tdistributions_ei,
        gmask,
        tflip_ei,
        [1/const_factor, const_factor],
        write_dir,
        "generic_markets",
    )
    write_pickle(tlocal_sa, fp_glocal_sa)
