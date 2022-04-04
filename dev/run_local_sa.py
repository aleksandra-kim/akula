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
    run_local_sa, run_local_sa_technosphere, run_local_sa_from_samples_technosphere,
    get_mask, get_tindices_wo_noninf, get_bindices_wo_noninf, get_cindices_wo_noninf, get_mask_parameters,
)
from akula.sensitivity_analysis.remove_non_influential import (
    get_variance_threshold, add_variances, get_indices_high_variance
)
from akula.parameterized_exchanges import get_parameters, get_lookup_cache
from akula.markets import DATA_DIR

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

# --> 2.1.1 Technosphere, 25867 exchanges

const_factors = [1/const_factor, const_factor]
# 2.1.1 Ecoinvent
# fp_tlocal_sa = write_dir / f"local_sa.tech.cutoff_{ctff:.0e}.maxcalc_{mclc:.0e}.pickle"
# if fp_tlocal_sa.exists():
#     tlocal_sa = read_pickle(fp_tlocal_sa)
# else:
#     tlocal_sa = run_local_sa_technosphere(
#         fu_mapped,
#         pkgs,
#         tdistributions_ei,
#         tmask_wo_noninf,
#         const_factors,
#         write_dir,
#         f"tech.cutoff_{ctff:.0e}.maxcalc_{mclc:.0e}",
#     )
#     write_pickle(tlocal_sa, fp_tlocal_sa)

# 2.1.2 Generic markets, 13586 exchanges
# fp_generic_markets = DATA_DIR / "generic-markets.zip"
# gms = bwp.load_datapackage(ZipFS(fp_generic_markets))
# gindices = gms.get_resource("generic markets.indices")[0]
# gmask = get_mask(tindices_ei, gindices)
# fp_glocal_sa = write_dir / f"local_sa.generic_markets.pickle"
# if fp_glocal_sa.exists():
#     glocal_sa = read_pickle(fp_glocal_sa)
# else:
#     glocal_sa = run_local_sa_technosphere(
#         fu_mapped,
#         pkgs,
#         gmask,
#         gmask,
#         const_factors,
#         write_dir,
#         "generic_markets",
#     )
#     write_pickle(glocal_sa, fp_glocal_sa)

# 2.1.3 Combustion, 1403 iterations
dp_name = "liquid-fuels-kilogram"
fp_flocal_sa = write_dir / f"local_sa.{dp_name}.pickle"
if fp_flocal_sa.exists():
    flocal_sa = read_pickle(fp_flocal_sa)
else:
    flocal_sa = run_local_sa_from_samples_technosphere(
        dp_name,
        fu_mapped,
        pkgs,
        const_factors,
        None,
        write_dir,
    )
    write_pickle(flocal_sa, fp_flocal_sa)

# 2.1.4, 2.2.2 Parameterization for tech and bio exchanges, 821 parameters
dp_name = "ecoinvent-parameterization"
fp_plocal_sa = write_dir / f"local_sa.{dp_name}.pickle"
lookup_cache = get_lookup_cache()
parameters = get_parameters()
activities = [lookup_cache[(param['activity']["database"], param['activity']["code"])] for param in parameters]
pindices = [(activities[i], p) for i, param in enumerate(parameters) for p in param['parameters']]
if fp_plocal_sa.exists():
    plocal_sa = read_pickle(fp_plocal_sa)
else:
    plocal_sa = run_local_sa_from_samples_technosphere(
        dp_name,
        fu_mapped,
        pkgs,
        const_factors,
        pindices,
        write_dir,
    )
    # Correct values
    for val in plocal_sa.values():
        if len(set(val)) == 1:
            pstatic_score = val[0]
            break
    correction = static_score - pstatic_score
    plocal_sa_corrected = {key: val+correction for key, val in plocal_sa.items()}
    write_pickle(plocal_sa_corrected, fp_plocal_sa)

# 2.1.5, 821 exchange??
dp_name = "entso-average"
resource_group = 'average ENTSO electricity values'
dp = bwp.load_datapackage(ZipFS(str(DATA_DIR / f"{dp_name}.zip")))
eindices = dp.get_resource(f"{resource_group}.indices")[0]
emask = get_mask(tindices_ei, eindices)

fp_elocal_sa = write_dir / f"local_sa.{dp_name}.pickle"
if fp_elocal_sa.exists():
    elocal_sa = read_pickle(fp_elocal_sa)
else:
    elocal_sa = run_local_sa_technosphere(
        fu_mapped,
        pkgs,
        emask,
        emask,
        const_factors,
        write_dir,
        dp_name.replace("-", "_"),
    )
    write_pickle(elocal_sa, fp_elocal_sa)


# --> 2.2.1 Biosphere, 12480 exchanges
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

# --> 2.3.1 Characterization, 77 exchanges
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


# 2.4 --> Remove lowly influential based on variance
local_sa_list = [
    # tlocal_sa,
    blocal_sa,
    clocal_sa,
    flocal_sa,
    plocal_sa,
    elocal_sa,
]
num_parameters = 10000

add_variances(local_sa_list, static_score)
# tlocal_sa_all = {**tlocal_sa, **flocal_sa, **elocal_sa}  # To exclude repetitive keys
tlocal_sa_all = {**flocal_sa, **elocal_sa}  # To exclude repetitive keys

local_sa_list = [tlocal_sa_all, blocal_sa, clocal_sa, plocal_sa]
var_threshold = get_variance_threshold(local_sa_list, num_parameters)

tindices_wo_lowinf = get_indices_high_variance(tlocal_sa_all, var_threshold)
tmask_wo_lowinf = get_mask(tindices_ei, tindices_wo_lowinf)

bindices_wo_lowinf = get_indices_high_variance(blocal_sa, var_threshold)
bmask_wo_lowinf = get_mask(bindices, bindices_wo_lowinf)

cindices_wo_lowinf = get_indices_high_variance(clocal_sa, var_threshold)
cmask_wo_lowinf = get_mask(cindices, cindices_wo_lowinf)

pindices_wo_lowinf = get_indices_high_variance(plocal_sa, var_threshold)
pmask_wo_lowinf = get_mask_parameters(pindices, pindices_wo_lowinf)

assert tmask_wo_lowinf.sum() + bmask_wo_lowinf.sum() + cmask_wo_lowinf.sum() + pmask_wo_lowinf.sum() == num_parameters

print(f"Selected {tmask_wo_lowinf.sum()} tech, {bmask_wo_lowinf.sum()} bio, {cmask_wo_lowinf.sum()} cf exchanges, "
      f"and {pmask_wo_lowinf.sum()} parameters")


# 2.5 --> Validation of results
viterations = 20
vseed = 22222000

fp_monte_carlo = write_dir
write_figs = Path("/Users/akim/PycharmProjects/akula/dev/write_files/paper3")

# Run when everything varies
fps = ["implicit-markets.zip", "liquid-fuels-kilogram.zip",  "ecoinvent-parameterization.zip", "entso-timeseries.zip"]
dps = [bwp.load_datapackage(ZipFS(fp)) for fp in fps]



fp_option = DATA_DIR / f"{option}.zip"

hh_average = [act for act in co if "ch hh average consumption aggregated, years 151617" == act['name']]
assert len(hh_average) == 1
demand_act = hh_average[0]
demand = {demand_act: 1}
demand_id = {demand_act.id: 1}

lca = bc.LCA(demand, method)
lca.lci()
lca.lcia()
print(lca.score)

iterations = 30
seed = 11111000
dict_for_lca = dict(
    use_distributions=True,
    use_arrays=True,
    seed_override=seed,
)
fp_monte_carlo_base = fp_monte_carlo / f"base.{iterations}.{seed}.pickle"
fp_monte_carlo_option = fp_monte_carlo / f"{option}.{iterations}.{seed}.pickle"

