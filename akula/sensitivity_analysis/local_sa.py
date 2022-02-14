import numpy as np
import bw2calc as bc
import bw_processing as bwp


def get_mask(all_indices, use_indices):
    """Creates a `mask` such that `all_indices[mask]=use_indices`."""
    use_indices = np.array(use_indices, dtype=[('row', '<i4'), ('col', '<i4')])
    mask = np.zeros(len(all_indices), dtype=bool)
    for inds in use_indices:
        mask_current = all_indices == inds
        mask = mask | mask_current
    assert sum(mask) == len(use_indices)
    return mask


def get_inds_tech_without_noninf(lca, cutoff, max_calc=1e4):
    """Find datapackage indices with lowest contribution scores obtained with Supply chain traversal"""
    res = bc.GraphTraversal().calculate(
        lca, cutoff=cutoff, max_calc=max_calc
    )
    use_indices = []
    use_indices_dict = {}
    for edge in res['edges']:
        if edge['to'] != -1:
            if abs(edge['impact']) > abs(lca.score * cutoff):
                row, col = edge['from'], edge['to']
                i, j = lca.dicts.activity.reversed[row], lca.dicts.activity.reversed[col]
                use_indices.append((i, j))
                use_indices_dict[(i, j)] = edge['impact']
    return use_indices


def get_inds_bio_without_noninf(lca, cutoff):
    """Find datapackage indices that correspond to B*Ainv*f, where contributions are higher than cutoff"""
    inv = lca.characterized_inventory
    finv = inv.multiply(abs(inv) > abs(lca.score * cutoff))
    # Find row and column in B*Ainv*f
    biosphere_row_col = list(zip(*finv.nonzero()))
    # Translate row and column to datapackage indices
    biosphere_reversed = lca.dicts.biosphere.reversed
    activity_reversed = lca.dicts.activity.reversed
    use_indices = []
    for row, col in biosphere_row_col:
        i, j = biosphere_reversed[row], activity_reversed[col]
        use_indices.append((i, j))
    return use_indices


def get_inds_cf_without_noninf(lca, cutoff):
    """Find datapackage indices that correspond to C*B*Ainv*f, where contributions are higher than cutoff"""
    inv_sum = np.array(np.sum(lca.characterized_inventory, axis=1)).squeeze()
    # print('Characterized inventory:', inv.shape, inv.nnz)
    finv_sum = inv_sum * abs(inv_sum) > abs(lca.score * cutoff)
    characterization_row = list(finv_sum.nonzero()[0])
    # Translate row to datapackage indices
    biosphere_reversed = lca.dicts.biosphere.reversed
    use_indices = [(biosphere_reversed[row], 1) for row in characterization_row]
    return use_indices


class LocalSAInterface:
    def __init__(self, indices, data, distributions, mask, factor=10, cutoff=1e-3):
        self.indices = indices
        self.data = data
        self.distributions = distributions
        self.has_uncertainty = self.get_uncertainty_bool(self.distributions)
        self.factor = factor
        self.cutoff = cutoff
        self.mask = mask  # indices with high enough contributions

        assert self.indices.shape[0] == self.data.shape[0] == self.distributions.shape[0]

        self.masked_indices = self.indices[self.mask]
        self.masked_data = self.data[self.mask]
        self.masked_has_uncertainty = self.has_uncertainty[self.mask]

        self.size = len(self.masked_indices)
        self.index = None  # To indicate we haven't consumed first value yet
        self.mask_where = np.where(self.mask)[0]

    def __next__(self):
        if self.index is None:
            self.index = 0
        else:
            self.index += 1

        if self.index < self.size:
            # 0 and 1 are `no` and `unknown` uncertainty
            while self.masked_has_uncertainty[self.index]:
                self.index += 1
                if self.index >= self.size:
                    raise StopIteration
        else:
            raise StopIteration

        data = self.data.copy()
        data[self.mask_where[self.index]] *= self.factor
        return data

    @staticmethod
    def get_uncertainty_bool(distributions):
        try:
            arr = distributions['uncertainty_type'] >= 2
        except:
            arr = distributions > 0
        return arr

    @property
    def coordinates(self):
        return self.masked_indices[self.index]


def run_local_sa(
        matrix_type,
        fu_mapped,
        packages,
        indices_array,
        data_array,
        distributions_array,
        mask_without_noninf,
        flip_array=None,  # only needed for technosphere
        const_factor=10,
        cutoff=1e-3,
):

    interface = LocalSAInterface(
        indices_array,
        data_array,
        distributions_array,
        mask_without_noninf,
        const_factor,
        cutoff,
    )

    dp = bwp.create_datapackage()
    dp.add_dynamic_vector(
        matrix=f"{matrix_type}_matrix",
        interface=interface,
        indices_array=indices_array,
        flip_array=flip_array,
    )
    if matrix_type == "characterization":
        [d.update({"global_index": 1}) for d in dp.metadata['resources']]  # TODO Chris, is this correct?

    lca_local_sa = bc.LCA(demand=fu_mapped, data_objs=packages + [dp])
    lca_local_sa.lci()
    lca_local_sa.lcia()

    interface.index = None  # there should be a better way to discount the first __next__
    indices_local_sa_scores = {}

    count = 0
    try:
        while True:
            next(lca_local_sa)
            count += 1
            indices_local_sa_scores[tuple(interface.coordinates)] = lca_local_sa.score
    except StopIteration:
        pass

    assert count <= sum(interface.mask)

    return indices_local_sa_scores



