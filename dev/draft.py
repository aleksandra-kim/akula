# import numpy as np
# from pathlib import Path
# from fs.zipfs import ZipFS
# import bw2data as bd
# import bw2calc as bc
# import bw_processing as bwp
# from copy import deepcopy
# from gsa_framework.utils import read_pickle, write_pickle
# from gsa_framework.visualization.plotting import plot_correlation_Y1_Y2
# import plotly.graph_objects as go
#
# # Local files
# from akula.sensitivity_analysis.local_sensitivity_analysis import (
#     run_local_sa_technosphere, get_mask, get_tindices_wo_noninf
# )
# from akula.sensitivity_analysis.remove_non_influential import (
#     get_variance_threshold, add_variances, get_indices_high_variance
# )
# from akula.markets import DATA_DIR
# from akula.background import get_lca_score_shift
#
#
# if __name__ == "__main__":
#
#     project = 'GSA for archetypes'
#     bd.projects.set_current(project)
#     const_factor = 10
#     ctff = 1e-6  # Cutoff for contribution analysis
#     mclc = 1e10   # Maximum number of computations for supply chain traversal
#
#     # Setups
#     ########
#
#     co_db = bd.Database('swiss consumption 1.0')
#     ei_db = bd.Database("ecoinvent 3.8 cutoff")
#     fu = [act for act in co_db if "Food" in act['name']][0]
#
#     write_dir = Path("write_files") / project.lower().replace(" ", "_") \
#         / fu['name'].lower().replace(" ", "_").replace(",", "")
#     write_dir_sct = write_dir / "supply_chain_traversal"
#     write_dir_sct.mkdir(exist_ok=True, parents=True)
#
#     demand = {fu: 1}
#     method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
#     fu_mapped, pkgs, _ = bd.prepare_lca_inputs(demand=demand, method=method, remapping=False)
#
#     lca = bc.LCA(demand=fu_mapped, data_objs=pkgs, use_distributions=False)
#     lca.lci()
#     lca.lcia()
#     static_score = deepcopy(lca.score)
#     print(static_score)
#
#     # Get all relevant data
#     ei = bd.Database('ecoinvent 3.8 cutoff').datapackage()
#     co = bd.Database('swiss consumption 1.0').datapackage()
#     cf = bd.Method(method).datapackage()
#
#     # Technosphere
#     tei = ei.filter_by_attribute('matrix', 'technosphere_matrix')
#     tindices_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.indices')[0]
#     tdata_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.data')[0]
#     tflip_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.flip')[0]
#     tdistributions_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.distributions')[0]
#
#     # Step 1.1 Technosphere & Supply chain traversal
#     fp_sct = write_dir_sct / f"sct.cutoff_{ctff:.0e}.maxcalc_{mclc:.0e}.pickle"
#     if fp_sct.exists():
#         tindices_wo_noninf = read_pickle(fp_sct)
#     else:
#         tindices_wo_noninf = get_tindices_wo_noninf(lca, ctff, mclc)
#         write_pickle(tindices_wo_noninf, fp_sct)
#
#     fp_tmask_wo_noninf = write_dir / f"mask.tech.without_noninf.sct.cutoff_{ctff:.0e}.maxcalc_{mclc:.0e}.pickle"
#     if fp_tmask_wo_noninf.exists():
#         tmask_wo_noninf = read_pickle(fp_tmask_wo_noninf)
#     else:
#         tmask_wo_noninf = get_mask(tindices_ei, tindices_wo_noninf)
#         write_pickle(tmask_wo_noninf, fp_tmask_wo_noninf)
#
#     # STEP 2: Run local SA
#     ######################
#
#     # --> 2.1.1 Technosphere
#
#     const_factors = [1/const_factor, const_factor]
#     # 2.1.1 Ecoinvent
#     fp_tlocal_sa = write_dir / f"local_sa.tech.cutoff_{ctff:.0e}.maxcalc_{mclc:.0e}.pickle"
#     if fp_tlocal_sa.exists():
#         tlocal_sa = read_pickle(fp_tlocal_sa)
#     else:
#         tlocal_sa = run_local_sa_technosphere(
#             fu_mapped,
#             pkgs,
#             tdistributions_ei,
#             tmask_wo_noninf,
#             const_factors,
#             write_dir,
#             f"tech.cutoff_{ctff:.0e}.maxcalc_{mclc:.0e}",
#         )
#         write_pickle(tlocal_sa, fp_tlocal_sa)
#
#     # 2.4.2 Determine variance threshold
#
#     add_variances([tlocal_sa], static_score)
#     num_parameters = 10000
#     var_threshold = get_variance_threshold([tlocal_sa], num_parameters)
#
#     datapackages = {
#         "technosphere": {
#             "local_sa": tlocal_sa,
#             "indices": tindices_ei,
#         },
#     }
#
#     # 2.4.3 Construct masks for all inputs after local SA
#     count = 0
#     print(f"Selected {num_parameters} exchanges after local SA:")
#     for name, data in datapackages.items():
#         dtype = bwp.INDICES_DTYPE
#         is_params = False
#         indices_wo_lowinf = get_indices_high_variance(data['local_sa'], var_threshold)
#         mask_wo_lowinf = get_mask(data["indices"], indices_wo_lowinf, is_params)
#         data['indices_wo_lowinf'] = np.array(indices_wo_lowinf, dtype=dtype)
#         data['mask_wo_lowinf'] = mask_wo_lowinf
#         print(f"    {mask_wo_lowinf.sum():5d} from {name}")
#         count += mask_wo_lowinf.sum()
#
#     # # 2.5 --> Validation of results after local SA
#     #
#     # viterations = 30
#     vseed = 22222000
#     tname = "technosphere"
#     #
#     # from akula.background import generate_validation_datapackages
#     # tmask = datapackages[tname]["mask_wo_lowinf"]
#     # tdp_vall, tdp_vinf = generate_validation_datapackages(
#     #     demand, tname, tindices_ei, tmask, num_samples=viterations, seed=vseed
#     # )
#     # datapackages[tname]['local_sa.validation_all'] = tdp_vall
#     # datapackages[tname]['local_sa.validation_inf'] = tdp_vinf
#     #
#     # # 2.6.1 All inputs vary
#     # print("computing all scores")
#     # lca_all = bc.LCA(
#     #     fu_mapped,
#     #     data_objs=pkgs + [tdp_vall],
#     #     use_distributions=False,
#     #     use_arrays=True,
#     #     seed_override=vseed,
#     # )
#     # lca_all.lci()
#     # lca_all.lcia()
#     # scores_all = [lca_all.score for _, _ in zip(lca_all, range(viterations))]
#     #
#     # # 2.6.1 Only influential inputs vary
#     # print("computing inf scores")
#     # lca_inf = bc.LCA(
#     #     fu_mapped,
#     #     data_objs=pkgs + [tdp_vinf],
#     #     use_distributions=False,
#     #     use_arrays=True,
#     #     seed_override=vseed,
#     # )
#     # lca_inf.lci()
#     # lca_inf.lcia()
#     # scores_inf = [lca_inf.score for _, _ in zip(lca_inf, range(viterations))]
#     #
#     # masks_dict_all = {
#     #     tname: np.ones(len(datapackages[tname]["mask_wo_lowinf"]), dtype=bool),
#     # }
#     # masks_dict_inf = {
#     #     tname: datapackages[tname]["mask_wo_lowinf"],
#     # }
#     #
#     # offset_all = get_lca_score_shift(demand, masks_dict_all, shift_median=False)
#     # offset_inf = get_lca_score_shift(demand, masks_dict_inf, shift_median=False)
#     # print(offset_all, offset_inf)
#     #
#     # Y1 = np.array(scores_all) - offset_all
#     # Y2 = np.array(scores_inf) - offset_inf
#     #
#     # fig = plot_correlation_Y1_Y2(
#     #     Y1,
#     #     Y2,
#     #     start=0,
#     #     end=50,
#     #     trace_name1="All vary",
#     #     trace_name2="Only inf vary"
#     # )
#     # fig.add_trace(
#     #     go.Scatter(
#     #         x=[0],
#     #         y=[static_score],
#     #         mode='markers',
#     #         marker=dict(color='black', symbol='x')
#     #     )
#     # )
#     # fig.show()
#
#     # STEP 3: Run high-dimensional screening
#     ########################################
#
#     # 3.1.1. Create background datapackages for Xgboost
#     from akula.background import create_background_datapackage
#     xiterations = 5000
#     # random_seeds = [71, 72, 73, 74]
#     # for random_seed in random_seeds:
#     #     print(random_seed)
#     #     for bg_name in [tname]:
#     #         print(bg_name)
#     #         indices = datapackages[bg_name]["indices_wo_lowinf"]
#     #         dp = create_background_datapackage(
#     #             demand, bg_name, f"tech-{bg_name}-{random_seed}", indices, num_samples=xiterations, seed=random_seed,
#     #         )
#     #         dp.finalize_serialization()
#
#     # fp_name = "tech-technosphere-61 copy.zip"
#     # dp_copy = bwp.load_datapackage(ZipFS(str(DATA_DIR / "xgboost" / fp_name)))
#
#     print("-----------------")
#
#     # 3.1.2. MC simulations for XGBoost
#     dps_xgboost_names = list(datapackages)
#     random_seeds = [71, 72, 73, 74]
#     for random_seed in random_seeds:
#         print(f"MC random seed {random_seed}")
#         fp_xgboost = write_dir / f"mc.tech.xgboost.{xiterations}.{random_seed}.pickle"
#         if fp_xgboost.exists():
#             scores_xgboost = read_pickle(fp_xgboost)
#         else:
#             dps_xgboost = []
#             for dp_name in dps_xgboost_names:
#                 dp_temp = bwp.load_datapackage(ZipFS(str(DATA_DIR / "xgboost" / f"tech-{dp_name}-{random_seed}.zip")))
#                 dps_xgboost.append(dp_temp)
#
#             lca_xgboost = bc.LCA(
#                 fu_mapped,
#                 data_objs=pkgs + dps_xgboost,
#                 use_distributions=False,
#                 use_arrays=True,
#             )
#             lca_xgboost.lci()
#             lca_xgboost.lcia()
#             scores_xgboost = [lca_xgboost.score] + [lca_xgboost.score for _, _ in zip(lca_xgboost, range(xiterations-1))]
#             write_pickle(scores_xgboost, fp_xgboost)
#
#     print("")
#
#
#     # ref_scores = []
#     # for d in dp_temp.data[1].T:
#     #     ref_dp = bwp.create_datapackage(
#     #         name='test',
#     #         sequential=True,
#     #     )
#     #     ref_dp.add_persistent_vector(
#     #         matrix="technosphere_matrix",
#     #         data_array=d,
#     #         # Resource group name that will show up in provenance
#     #         name='test',
#     #         indices_array=indices,
#     #         flip_array=dp_temp.data[2],
#     #     )
#     #     ref_lca = bc.LCA(
#     #         fu_mapped,
#     #         data_objs=pkgs + [ref_dp],
#     #         use_distributions=False,
#     #         use_arrays=True,
#     #     )
#     #     ref_lca.lci()
#     #     ref_lca.lcia()
#     #     ref_scores.append(ref_lca.score)
#     #
#     # print(scores_xgboost)
#     # print(f" Correct scores: {ref_scores}")

import pandas as pd
from pathlib import Path
import numpy as np
import bw_processing as bwp
import bw2data as bd

bd.projects.set_current("GSA for archetypes")

path = Path("/Users/akim/Documents/paper 3, env research letters/")
ifp = path / "gsa_ranking_independent.xlsx"
cfp = path / "gsa_ranking_correlated.xlsx"
idp = pd.read_excel(ifp, skiprows=[1])
cdp = pd.read_excel(cfp, skiprows=[1])

modules = list(set(cdp['module']))
ilist = idp[["input_id", "output_id"]].values
clist = cdp[["input_id", "output_id"]].values
ituple = [tuple(el) for el in ilist]
ctuple = [tuple(el) for el in clist]
iio = np.array(ituple, dtype=bwp.INDICES_DTYPE)
cio = np.array(ctuple, dtype=bwp.INDICES_DTYPE)

idata = {}
cdata = {}
for m in modules:
    iwhere = np.where(idp['module'].values == m)[0]
    idata[m] = iio[iwhere]
    cwhere = np.where(cdp['module'].values == m)[0]
    cdata[m] = cio[cwhere]
    interesection = np.intersect1d(idata[m], cdata[m]).shape[0]
    i_minus_c = np.setdiff1d(idata[m], cdata[m]).shape[0]
    c_minus_i = np.setdiff1d(cdata[m], idata[m]).shape[0]
    print(m, i_minus_c, interesection, c_minus_i)
print("")


# import bw2data as bd
# import bw2calc as bc
# import bw_processing as bwp
# from fs.zipfs import ZipFS
# from akula.markets import DATA_DIR
# from gsa_framework.utils import read_pickle
#
#
# bd.projects.set_current("GSA for archetypes")
# dp = bwp.load_datapackage(ZipFS(str(DATA_DIR / "xgboost" / "implicit-markets-81.zip")))
# fp_markets = DATA_DIR / "implicit-markets.pickle"
#
# markets = read_pickle(fp_markets)
#
#
# print("")
