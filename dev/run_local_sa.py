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


def run_local_sa_technosphere(
        func_unit_mapped,
        packages,
        tech_has_uncertainty,
        mask_tech_without_noninf,
        factors,
        directory,
        tag,
):
    ecoinvent = bd.Database('ecoinvent 3.8 cutoff').datapackage()
    tech_ecoinvent = ecoinvent.filter_by_attribute('matrix', 'technosphere_matrix')
    tech_indices = tech_ecoinvent.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.indices')[0]
    tech_data = tech_ecoinvent.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.data')[0]
    tech_flip = tech_ecoinvent.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.flip')[0]
    for i, factor in enumerate(factors):
        fp_factor = directory / f"local_sa.{tag}.factor_{factor:.0e}.pickle"
        if fp_factor.exists():
            local_sa_current = read_pickle(fp_factor)
        else:
            local_sa_current = run_local_sa(
                "technosphere",
                func_unit_mapped,
                packages,
                tech_indices,
                tech_data,
                tech_has_uncertainty,
                mask_tech_without_noninf,
                tech_flip,
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
tindices_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.indices')[0]
tdata_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.data')[0]
tflip_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.flip')[0]
tdistributions_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.distributions')[0]

# Biosphere
bei = ei.filter_by_attribute('matrix', 'biosphere_matrix')
bindices = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.indices')[0]
bdata = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.data')[0]
bdistributions = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.distributions')[0]

# Characterization
cindices = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.indices')[0]
cdata = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.data')[0]
cdistributions = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.distributions')[0]


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

# --> Technosphere
const_factors = [1/const_factor, const_factor]
# # 2.1.1 Technosphere ecoinvent
# fp_tlocal_sa = write_dir / f"local_sa.tech.cutoff_{ctff:.0e}.maxcalc_{mclc:.0e}.pickle"
# if fp_tlocal_sa.exists():
#     tlocal_sa = read_pickle(fp_tlocal_sa)
# else:
#     tlocal_sa = run_tlocal_sa(
#         fu_mapped,
#         pkgs,
#         tdistributions_ei,
#         tmask_wo_noninf,
#         const_factors,
#         write_dir,
#         f"tech.cutoff_{ctff:.0e}.maxcalc_{mclc:.0e}",
#     )
#     write_pickle(tlocal_sa, fp_tlocal_sa)

# 2.1.2 Generic markets
fp_generic_markets = DATA_DIR / "generic-markets.zip"
gms = bwp.load_datapackage(ZipFS(fp_generic_markets))
gindices = gms.get_resource("generic markets.indices")[0]
gmask = get_mask(tindices_ei, gindices)
fp_glocal_sa = write_dir / f"local_sa.generic_markets.pickle"
if fp_glocal_sa.exists():
    glocal_sa = read_pickle(fp_glocal_sa)
else:
    glocal_sa = run_local_sa_technosphere(
        fu_mapped,
        pkgs,
        gmask,
        gmask,
        const_factors,
        write_dir,
        "generic_markets",
    )
    write_pickle(glocal_sa, fp_glocal_sa)

# 2.1.3 Carbon
fp_liquid_fuels = DATA_DIR / "liquid-fuels-kilogram.zip"
lfk = bwp.load_datapackage(ZipFS(fp_liquid_fuels))
findices = lfk.get_resource("liquid fuels in kilograms.indices")[0]
fmask = get_mask(tindices_ei, findices)

fp_flocal_sa = write_dir / f"local_sa.liquid_fuels.pickle"
if fp_flocal_sa.exists():
    flocal_sa = read_pickle(fp_flocal_sa)
else:
    flocal_sa = run_local_sa_technosphere(
        fu_mapped,
        pkgs,
        tdistributions_ei,
        fmask,
        const_factors,
        write_dir,
        "liquid_fuels",
    )
    write_pickle(flocal_sa, fp_flocal_sa)


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
fp_clocal_sa = write_dir / "local_sa.cf.pickle"
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

# 2.4 Remove lowly influential based on variance
# Add static score
local_sa_list = [
    tlocal_sa,
    blocal_sa,
    clocal_sa,
    glocal_sa,
    flocal_sa,
]
for dict_ in local_sa_list:
    values = np.vstack(list(dict_.values()))
    values = np.hstack([values, np.ones((len(values), 1))*static_score])
    variances = np.var(values, axis=1)
    for i, k in enumerate(dict_.keys()):
        #         dict_.update({k: values[i,:]})
        dict_[k] = {
            "arr": values[i, :],
            "var": variances[i],
        }