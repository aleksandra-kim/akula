import numpy as np
from pathlib import Path
import bw2data as bd
import bw2calc as bc

from .utils import get_mask
from ..utils import read_pickle, write_pickle, get_lca

GSA_DIR = Path(__file__).parent.parent.parent.resolve() / "data" / "sensitivity-analysis"
GSA_DIR.mkdir(parents=True, exist_ok=True)


def get_tindices_wo_noninf(project, cutoff, max_calc):
    """Find datapackage indices with the lowest contribution scores obtained with Supply chain traversal."""

    fp = GSA_DIR / f"tech.indices.without_noninf.sct.cutoff_{cutoff:.0e}.maxcalc_{max_calc:.0e}.pickle"

    if fp.exists():
        indices = read_pickle(fp)
    else:
        bd.projects.set_current(project)
        lca = get_lca(project)
        res = bc.graph_traversal.AssumedDiagonalGraphTraversal().calculate(
            lca, cutoff=cutoff, max_calc=max_calc
        )
        indices = []
        indices_dict = {}
        for edge in res['edges']:
            if edge['to'] != -1:
                if abs(edge['impact']) > abs(lca.score * cutoff):
                    row, col = edge['from'], edge['to']
                    i, j = lca.dicts.activity.reversed[row], lca.dicts.activity.reversed[col]
                    indices.append((i, j))
                    indices_dict[(i, j)] = edge['impact']

        write_pickle(indices, fp)

    return indices


def get_bindices_wo_noninf(project):
    """
    Find datapackage indices that correspond to B * A_inverse * f, where contributions are higher than cutoff.

    Partially taken from:
    github.com/LCA-ActivityBrowser/activity-browser/blob/master/activity_browser/bwutils/sensitivity_analysis.py
    """

    fp = GSA_DIR / "bio.indices.without_noninf.pickle"

    if fp.exists():
        indices = read_pickle(fp)
    else:

        lca = get_lca(project)
        lca.lci()
        lca.lcia()

        inv = lca.characterized_inventory
        finv = inv.multiply(abs(inv) > 0)

        # Find row and column in B * A_inverse * f
        biosphere_row_col = list(zip(*finv.nonzero()))

        # Translate row and column to datapackage indices
        biosphere_reversed = lca.dicts.biosphere.reversed
        activity_reversed = lca.dicts.activity.reversed
        indices = []
        for row, col in biosphere_row_col:
            i, j = biosphere_reversed[row], activity_reversed[col]
            indices.append((i, j))

        write_pickle(indices, fp)

    return indices


def get_cindices_wo_noninf(project):
    """
    Find datapackage indices that correspond to C * B * A_inverse * f, where contributions are higher than cutoff.

    Partially taken from:
    github.com/LCA-ActivityBrowser/activity-browser/blob/master/activity_browser/bwutils/sensitivity_analysis.py
    """

    fp = GSA_DIR / "cf.indices.without_noninf.pickle"

    if fp.exists():
        indices = read_pickle(fp)
    else:
        lca = get_lca(project)
        lca.lci()
        lca.lcia()

        inv = lca.characterized_inventory
        inv_sum = np.array(np.sum(inv, axis=1)).squeeze()
        finv_sum = inv_sum * abs(inv_sum) > 0
        characterization_row = list(finv_sum.nonzero()[0])

        # Translate row to datapackage indices
        biosphere_reversed = lca.dicts.biosphere.reversed
        indices = [(biosphere_reversed[row], 1) for row in characterization_row]

        write_pickle(indices, fp)

    return indices


def get_tmask_wo_noninf(project, cutoff, max_calc):

    fp = GSA_DIR / f"tech.mask.without_noninf.sct.cutoff_{cutoff:.0e}.maxcalc_{max_calc:.0e}.pickle"

    if fp.exists():
        mask = read_pickle(fp)
    else:
        bd.projects.set_current(project)
        ei = bd.Database("ecoinvent 3.8 cutoff").datapackage()
        tei = ei.filter_by_attribute('matrix', 'technosphere_matrix')
        tindices_ei = tei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.indices')[0]
        tindices_wo_noninf = get_tindices_wo_noninf(project, cutoff, max_calc)
        mask = get_mask(tindices_ei, tindices_wo_noninf)

        write_pickle(mask, fp)

    return mask


def get_bmask_wo_noninf(project):

    fp = GSA_DIR / "bio.mask.without_noninf.pickle"

    if fp.exists():
        mask = read_pickle(fp)
    else:
        bd.projects.set_current(project)
        ei = bd.Database("ecoinvent 3.8 cutoff").datapackage()
        bei = ei.filter_by_attribute('matrix', 'biosphere_matrix')
        bindices = bei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.indices')[0]
        bindices_wo_noninf = get_bindices_wo_noninf(project)
        mask = get_mask(bindices, bindices_wo_noninf)

        write_pickle(mask, fp)

    return mask


def get_cmask_wo_noninf(project):

    fp = GSA_DIR / "cf.mask.without_noninf.pickle"

    if fp.exists():
        mask = read_pickle(fp)
    else:
        bd.projects.set_current(project)
        cf = bd.Method(("IPCC 2013", "climate change", "GWP 100a", "uncertain")).datapackage()
        cindices = cf.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.indices')[0]
        cindices_wo_noninf = get_cindices_wo_noninf(project)
        mask = get_mask(cindices, cindices_wo_noninf)

        write_pickle(mask, fp)

    return mask
