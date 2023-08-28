import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from pathlib import Path

PROJECT = "GSA with correlations"
PROJECT_DIR = Path(__file__).parent.parent.resolve()


if __name__ == "__main__":
    bd.projects.set_current(PROJECT)
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")

    fu, data_objs, _ = bd.prepare_lca_inputs({ch_low: 1}, method=ipcc, remapping=False)
    lca = bc.LCA(
        demand=fu,
        data_objs=(
                data_objs
                + [bwp.load_datapackage(ZipFS("../akula/data/entso-timeseries.zip"))]
        ),
        use_arrays=True,
        use_distributions=True
    )
    lca.lci()
    lca.lcia()


    entsoe_timeseries = PROJECT_DIR / "entsoe_timeseries"