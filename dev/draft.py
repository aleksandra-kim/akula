import numpy as np
from pathlib import Path
import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from fs.zipfs import ZipFS

from akula.parameterized_exchanges import DATA_DIR
# from akula.electricity.create_datapackages import create_average_entso_datapackages


if __name__ == "__main__":

    project = "GSA for archetypes"
    bd.projects.set_current(project)

    co = bd.Database('swiss consumption 1.0')
    fu = [act for act in co if "ch hh average consumption aggregated, years 151617" == act['name']][0]
    demand = {fu: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")

    fu_mapped, packages, _ = bd.prepare_lca_inputs(demand=demand, method=method, remapping=False)

    fp_flocal_sa = DATA_DIR / "local-sa-1e+01-liquid-fuels-kilogram.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp_flocal_sa)))

    lca = bc.LCA(demand=fu_mapped, data_objs=packages, use_distributions=False)
    lca.lci()
    lca.lcia()
    print("\n--> data_objs=packages, use_distributions=False")
    print([(lca.score, next(lca)) for _ in range(5)])

    lca = bc.LCA(demand=fu_mapped, data_objs=packages+[dp], use_distributions=False, use_arrays=True)
    lca.lci()
    lca.lcia()
    print("\n--> data_objs=packages+[dp], use_distributions=False")
    print([(lca.score, next(lca)) for _ in range(5)])

    lca = bc.LCA(demand=fu_mapped, data_objs=packages+[dp], use_distributions=True)
    lca.lci()
    lca.lcia()
    print("\n--> data_objs=packages+[dp], use_distributions=True")
    print([(lca.score, next(lca)) for _ in range(5)])

    me = bd.Method(method).datapackage()
    bs = bd.Database("biosphere3").datapackage()
    ei = bd.Database("ecoinvent 3.8 cutoff").datapackage()
    co = bd.Database("swiss consumption 1.0").datapackage()
    re = bd.Database("swiss residual electricity mix").datapackage()

    seed = 42

    class SimpleInterface:
        def __init__(self, data):
            self.data = data

        def __next__(self):
            return self.data


    class Simple2DInterface:
        def __init__(self, data):
            self.data = data
            self.size = data.shape[1]

        def __next__(self):
            if self.index is None:
                self.index = 0
            else:
                self.index += 1
            if self.index >= self.size:
                raise StopIteration
            else:
                return self.data[:, self.index]


    name = "static-uncertain-gwp"
    me_static = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}.zip"), write=True),
        name=name,
        # set seed to have reproducible (though not sequential) sampling
        seed=seed,
        sequential=True,
    )
    interface = SimpleInterface(me.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.data')[0])
    me_static.add_dynamic_vector(
        matrix="characterization_matrix",
        interface=interface,
        # Resource group name that will show up in provenance
        name=name,
        indices_array=me.get_resource('IPCC_2013_climate_change_GWP_100a_uncertain_matrix_data.indices')[0],
    )
    me_static.finalize_serialization()

    name = "static-ecoinvent-tech"
    ei_tech_static = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}.zip"), write=True),
        name=name,
        # set seed to have reproducible (though not sequential) sampling
        seed=seed,
        sequential=True,
    )
    interface = SimpleInterface(ei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.data')[0])
    ei_tech_static.add_dynamic_vector(
        matrix="technosphere_matrix",
        interface=interface,
        # Resource group name that will show up in provenance
        name=name,
        indices_array=ei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.indices')[0],
        flip_array=ei.get_resource('ecoinvent_3.8_cutoff_technosphere_matrix.flip')[0],
    )
    ei_tech_static.finalize_serialization()

    name = "static-ecoinvent-bio"
    ei_bio_static = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}.zip"), write=True),
        name=name,
        # set seed to have reproducible (though not sequential) sampling
        seed=seed,
        sequential=True,
    )
    interface = SimpleInterface(ei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.data')[0])
    ei_bio_static.add_dynamic_vector(
        matrix="biosphere_matrix",
        interface=interface,
        # Resource group name that will show up in provenance
        name=name,
        indices_array=ei.get_resource('ecoinvent_3.8_cutoff_biosphere_matrix.indices')[0],
    )
    ei_bio_static.finalize_serialization()

    name = "static-swiss-consumption"
    co_static = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}.zip"), write=True),
        name=name,
        # set seed to have reproducible (though not sequential) sampling
        seed=seed,
        sequential=True,
    )
    interface = SimpleInterface(co.get_resource('swiss_consumption_1.0_technosphere_matrix.data')[0])
    co_static.add_dynamic_vector(
        matrix="technosphere_matrix",
        interface=interface,
        # Resource group name that will show up in provenance
        name=name,
        indices_array=co.get_resource('swiss_consumption_1.0_technosphere_matrix.indices')[0],
        flip_array=co.get_resource('swiss_consumption_1.0_technosphere_matrix.flip')[0],
    )
    co_static.finalize_serialization()

    name = "static-swiss-residual-mix"
    re_static = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}.zip"), write=True),
        name=name,
        # set seed to have reproducible (though not sequential) sampling
        seed=seed,
        sequential=True,
    )
    interface = SimpleInterface(re.get_resource('swiss_residual_electricity_mix_technosphere_matrix.data')[0])
    re_static.add_dynamic_vector(
        matrix="technosphere_matrix",
        interface=interface,
        # Resource group name that will show up in provenance
        name=name,
        indices_array=re.get_resource('swiss_residual_electricity_mix_technosphere_matrix.indices')[0],
        flip_array=re.get_resource('swiss_residual_electricity_mix_technosphere_matrix.flip')[0],
    )
    re_static.finalize_serialization()

    name = "local-sa-combustion-tech"
    dp_localsa_tech = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}.zip"), write=True),
        name=name,
        # set seed to have reproducible (though not sequential) sampling
        seed=seed,
        sequential=True,
    )
    interface = Simple2DInterface(dp.get_resource('local-sa-liquid-fuels-kilogram-tech.data')[0])
    dp_localsa_tech.add_dynamic_vector(
        matrix="technosphere_matrix",
        interface=interface,
        # Resource group name that will show up in provenance
        name=name,
        indices_array=dp.get_resource('local-sa-liquid-fuels-kilogram-tech.indices')[0],
        flip_array=dp.get_resource('local-sa-liquid-fuels-kilogram-tech.flip')[0],
    )
    dp_localsa_tech.finalize_serialization()

    name = "local-sa-combustion-tech"
    dp_localsa_bio = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}.zip"), write=True),
        name=name,
        # set seed to have reproducible (though not sequential) sampling
        seed=seed,
        sequential=True,
    )
    interface = Simple2DInterface(dp.get_resource('local-sa-liquid-fuels-kilogram-bio.data')[0])
    dp_localsa_bio.add_dynamic_vector(
        matrix="biosphere_matrix",
        interface=interface,
        # Resource group name that will show up in provenance
        name=name,
        indices_array=dp.get_resource('local-sa-liquid-fuels-kilogram-bio.indices')[0],
    )
    dp_localsa_bio.finalize_serialization()

    static_packages = [
        ei_bio_static,
        ei_tech_static,
        me_static,
        co_static,
        re_static,
    ]

    dp_localsa_bio.rehydrate_interface()
    dp_localsa_tech.rehydrate_interface()

    lca = bc.LCA(demand=fu_mapped, data_objs=static_packages, use_distributions=True)
    lca.lci()
    lca.lcia()
    print("\n--> data_objs=static_packages, use_distributions=True")
    print([(lca.score, next(lca)) for _ in range(5)])

    lca = bc.LCA(demand=fu_mapped, data_objs=static_packages + [dp], use_distributions=True)
    lca.lci()
    lca.lcia()
    print("\n--> data_objs=static_packages + [dp], use_distributions=True")
    print([(lca.score, next(lca)) for _ in range(5)])

    lca = bc.LCA(demand=fu_mapped, data_objs=static_packages + [dp_localsa_tech, dp_localsa_bio], use_distributions=False)
    lca.lci()
    lca.lcia()
    print("\n--> data_objs=static_packages + [dp], use_distributions=False")
    print([(lca.score, next(lca)) for _ in range(5)])

    a = 5

    print("jj")



