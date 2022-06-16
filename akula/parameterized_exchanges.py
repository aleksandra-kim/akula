import re
import traceback
from numbers import Number
from pathlib import Path
from copy import deepcopy
import bw2data as bd
import bw2io as bi
import bw2parameters as bwpa
import bw_processing as bwp
from bw_processing.merging import merge_datapackages_with_mask
import numpy as np
from asteval import Interpreter
from bw2data.backends.schema import ActivityDataset as AD
from bw2parameters.errors import BroadcastingError
from fs.zipfs import ZipFS
from stats_arrays import uncertainty_choices
from tqdm import tqdm
from gsa_framework.utils import read_pickle, write_pickle

assert bi.__version__ >= (0, 9, "DEV7")


SAMPLES = 25000
PARAMS_DTYPE = [('row', '<U40'), ('col', '<i4')]

DATA_DIR = Path(__file__).parent.resolve() / "data"
# FILEPATH = "/Users/cmutel/Documents/lca/Ecoinvent/3.8/cutoff/datasets"
FILEPATH = "/Users/akim/Documents/LCA_files/ecoinvent_38_cutoff/datasets"
FILEPATH_PARAMETERS = DATA_DIR / "ecoinvent-parameters.pickle"

MC_ERROR_TEXT = """Formula returned array of wrong shape:
Name: {}
Formula: {}
Expected shape: {}
Returned shape: {}"""

substitutions = {
    "yield": "yield_",
    "import": "import_",
    "load": "load_",
}


class PatchedParameterSet(bwpa.ParameterSet):
    def evaluate_monte_carlo(self, iterations=SAMPLES, stochastic=True, seed=42):
        """Evaluate each formula using Monte Carlo and variable uncertainty data, if present.
        Formulas **must** return a one-dimensional array, or ``BroadcastingError`` is raised.
        Returns dictionary of ``{parameter name: numpy array}``."""
        interpreter = Interpreter()
        result = {}
        for key in self.order:
            if key in self.global_params:
                if stochastic:
                    interpreter.symtable[key] = result[key] = self.get_rng_sample(
                        self.global_params[key], iterations, seed,
                    )
                else:
                    interpreter.symtable[key] = result[key] = self.get_static_sample(self.global_params[key])
            elif self.params[key].get("formula"):
                sample = self.fix_shape(interpreter(self.params[key]["formula"]), iterations)
                if sample.shape != (iterations,):
                    raise BroadcastingError(
                        MC_ERROR_TEXT.format(
                            key,
                            self.params[key]["formula"],
                            (iterations,),
                            sample.shape,
                        )
                    )
                interpreter.symtable[key] = result[key] = sample
            else:
                if stochastic:
                    interpreter.symtable[key] = result[key] = self.get_rng_sample(self.params[key], iterations, seed)
                else:
                    interpreter.symtable[key] = result[key] = self.get_static_sample(self.params[key])
        return result

    @staticmethod
    def get_rng_sample(obj, iterations, seed=42):
        if isinstance(obj, np.ndarray):
            # Already a Monte Carlo sample
            return obj
        if "uncertainty_type" not in obj:
            if "uncertainty type" not in obj:
                obj = obj.copy()
                obj["uncertainty_type"] = 0
                obj["loc"] = obj["amount"]
            else:
                obj["uncertainty_type"] = obj["uncertainty type"]
        kls = uncertainty_choices[obj["uncertainty_type"]]
        np.random.seed(seed)
        seeded_random = np.random
        return kls.bounded_random_variables(kls.from_dicts(obj), iterations, seeded_random=seeded_random).ravel()

    @staticmethod
    def get_static_sample(obj):
        return obj['amount']

    @staticmethod
    def fix_shape(array, iterations):
        # This is new
        if array is None:
            return np.zeros((iterations,))
        elif isinstance(array, Number):
            return np.ones((iterations,)) * array
        elif not isinstance(array, np.ndarray):
            return np.zeros((iterations,))
        # End new section
        elif array.shape in {(1, iterations), (iterations, 1)}:
            return array.reshape((iterations,))
        else:
            return array


def drop_pedigree_uncertainty(dct):
    if "scale" in dct and "scale with pedigree" in dct:
        dct["scale with pedigree"] = dct.pop("scale")
        dct["scale"] = dct.pop("scale without pedigree")
    return dct


def clean_formula(string):
    string = (
        string.strip()
        .replace("%", " / 100")
        .replace("^", " ** ")
        .replace("\r\n", " ")
        .replace("\n", "")
    )

    for k, v in substitutions.items():
        string = string.replace(k, v)

    string = re.sub(r"(\d)\,(\d)", r"\1.\2", string)
    return string


def clean_dct(dct):
    if dct.get("formula"):
        dct["formula"] = clean_formula(dct["formula"])
    if dct.get("name") in substitutions:
        dct["name"] = substitutions[dct["name"]]
    return dct


def reformat_parameters(act):
    parameters = {
        substitutions.get(dct["name"], dct["name"]): clean_dct(
            drop_pedigree_uncertainty(dct)
        )
        for dct in act["parameters"]
        if "name" in dct
    }

    for index, exc in enumerate(act["exchanges"]):
        if exc.get("formula"):
            pn = f"__exchange_{index}"
            exc["parameter_name"] = pn
            parameters[pn] = {"formula": clean_formula(exc["formula"])}

    return parameters


def parameter_set_for_activity(act, iterations=250, stochastic=True, seed=42):
    ps = PatchedParameterSet(reformat_parameters(act))
    return ps.evaluate_monte_carlo(iterations=iterations, stochastic=stochastic, seed=seed)


def check_that_parameters_are_reasonable(act, results, rtol=0.1):
    for exc in act["exchanges"]:
        if exc.get("formula"):
            arr = results[exc["parameter_name"]]
            if not np.isclose(exc["amount"], np.median(arr), rtol=rtol):
                return False
    return True


def get_ecoinvent_raw_data(filepath=FILEPATH):
    fp_ei = DATA_DIR / "ecoinvent.pickle"
    if fp_ei.exists():
        eii = read_pickle(fp_ei)
    else:
        eii = bi.SingleOutputEcospold2Importer(filepath, "ecoinvent 3.8 cutoff")
        eii.apply_strategies()
        write_pickle(eii, fp_ei)
    return eii.data


def get_lookup_cache():
    return {
        (x, y): z
        for x, y, z in AD.select(AD.database, AD.code, AD.id)
        .where(AD.database << ("biosphere3", "ecoinvent 3.8 cutoff"))
        .tuples()
    }


def add_tech_bio_data(
        technosphere_data, biosphere_data, lookup_cache, exchange, activity, parameters, parameters_static=None,
):
    if parameters_static is None:
        parameters_static = {}
    if exchange["input"][0] == "ecoinvent 3.8 cutoff":
        technosphere_data.append(
            (
                (
                    lookup_cache[exchange["input"]],
                    lookup_cache[(activity["database"], activity["code"])],
                ),
                parameters[exchange["parameter_name"]],
                exchange["type"] != "production",  # flip
                parameters_static.get(exchange["parameter_name"], None),
            )
        )
        if parameters_static is not None:
            biosphere_data.append((None, None, None, None))
    else:
        biosphere_data.append(
            (
                (
                    lookup_cache[exchange["input"]],
                    lookup_cache[(activity["database"], activity["code"])],
                ),
                parameters[exchange["parameter_name"]],
                False,
                parameters_static.get(exchange["parameter_name"], None),
            )
        )
        if parameters_static is not None:
            technosphere_data.append((None, None, None, None))
    return technosphere_data, biosphere_data


def get_parameterized_values(input_data, mask=None, num_samples=SAMPLES, seed=42):
    tech_data, bio_data = [], []
    params_data = {}

    lookup_cache = get_lookup_cache()

    parameters_list = get_parameters(input_data)
    if mask is not None:
        parameters_list = parameters_list[mask]

    for element in tqdm(parameters_list):

        act = element['activity']
        id_ = lookup_cache[(act['database'], act['code'])]

        use_exchanges = list(element['exchanges'])
        use_exchanges = [int(exc.replace("__exchange_", "")) for exc in use_exchanges]

        params = parameter_set_for_activity(act, iterations=num_samples, stochastic=True, seed=seed)

        # ps = PatchedParameterSet(reformat_parameters(act))
        # use_params = remove_unused_parameters(params)
        # mc_params = ps.evaluate_monte_carlo()

        for p, v in element['parameters'].items():
            if ("__exchange_" not in p) and (len(set(v)) > 1):
                params_data[(p, id_)] = params[p]

        for iexc in use_exchanges:
            exc = act['exchanges'][iexc]
            tech_data, bio_data = add_tech_bio_data(
                tech_data, bio_data, lookup_cache, exc, act, params,
            )
    tech_data = [t for t in tech_data if t[0] is not None]
    bio_data = [b for b in bio_data if b[0] is not None]
    # return sorted(tech_data), sorted(bio_data)
    return tech_data, bio_data, params_data


def get_dp_arrays(dict_):
    indices, sample, flip, static, positions = [], [], [], [], []
    i, j, k = 0, 0, 0
    for params in dict_.values():
        param = list(params.values())[0]
        indices += [p[0] for p in param if p[0] is not None]
        flip += [p[2] for p in param if p[2] is not None]
        static += [p[3] for p in param if p[3] is not None]
        for param in params.values():
            sample += [p[1] for p in param if p[1] is not None]
            k = 0
            for element in param:
                if element[0] is not None:
                    positions.append((i+k, j))
                    k += 1
            j += 1
        i += k
    indices = np.array(indices, dtype=bwp.INDICES_DTYPE)
    sample = np.array(sample).flatten()
    flip = np.array(flip, dtype=bool).flatten()
    static = np.array(static).flatten()
    data = np.tile(static, (j, 1)).T
    for p, pos in enumerate(positions):
        data[pos[0], pos[1]] = sample[p]
    return indices, data, flip


def get_parameters(input_data=None):
    if FILEPATH_PARAMETERS.exists():
        parameters_list = read_pickle(FILEPATH_PARAMETERS)
    else:
        found, errors, unreasonable, missing = 0, 0, 0, 0

        error_log = open("error.log", "w")
        missing_reference_log = open("undefined_reference.log", "w")

        parameters_list = []

        for act in tqdm(input_data):
            if any(exc.get("formula") for exc in act["exchanges"]):
                try:
                    ps = PatchedParameterSet(reformat_parameters(act))
                    params_exchanges = ps.evaluate_monte_carlo(iterations=1, stochastic=False)
                    if check_that_parameters_are_reasonable(act, params_exchanges):
                        found += 1

                        params = remove_unused_parameters(ps.params)
                        params_wo_exchanges = {p: v for p, v in params.items() if "__exchange_" not in p}

                        if len(params_wo_exchanges) > 0:
                            excs = remove_unused_exchanges(ps.params)
                            parameters_list.append(
                                {
                                    "parameters": params_wo_exchanges,
                                    "activity": act,
                                    "exchanges": excs,
                                }
                            )
                    else:
                        unreasonable += 1
                except (ValueError, SyntaxError, bwpa.errors.DuplicateName):
                    error_log.write(act["filename"] + "\n")
                    traceback.print_exc(file=error_log)
                    errors += 1
                    pass
                except bwpa.errors.ParameterError:
                    missing_reference_log.write(act["filename"] + "\n")
                traceback.print_exc(file=missing_reference_log)
                missing += 1

        error_log.close()
        missing_reference_log.close()

        print(
            f"""Activity statistics:
        Parameterized activities: {found}
        Activities whose formulas we can't parse: {errors}
        Activities whose formulas produce unreasonable values: {unreasonable}
        Activities whose formulas contain missing references: {missing}"""
        )

        write_pickle(parameters_list, FILEPATH_PARAMETERS)
    return parameters_list


def remove_unused_parameters(params_dict):
    params = list(params_dict.keys())
    formulas = {v['formula'] for p, v in params_dict.items() if "__exchange" in p}
    formulas_str = "|".join(formulas)
    for param in params:
        unct = params_dict[param].get("uncertainty type", 0)
        # if (param not in formulas_str) and ("__exchange_" not in param):
        if ((param not in formulas_str) or (unct < 2)) and ("__exchange_" not in param):
            params_dict.pop(param)
    return params_dict


def remove_unused_exchanges(params_dict):
    exchanges = {p: v for p, v in params_dict.items() if "__exchange_" in p}
    params = [p for p, v in params_dict.items() if "__exchange_" not in p]
    exchanges_copy = deepcopy(exchanges)
    for exc, dict_ in exchanges_copy.items():
        flag = False
        for p in params:
            if p in dict_['formula']:
                flag = True
                continue
        if not flag:
            exchanges.pop(exc)
    return exchanges


def get_parameterized_local_sa_values(input_data, const_factor=10.0):

    technosphere, biosphere = {}, {}

    lookup_cache = get_lookup_cache()

    parameters_list = get_parameters(input_data)

    for element in tqdm(parameters_list):

        act = element['activity']

        use_exchanges = list(element['exchanges'])
        use_exchanges = [int(exc.replace("__exchange_", "")) for exc in use_exchanges]

        use_parameters = [p for p in element['parameters'] if "__exchange_" not in p]
        num_params = len(act['parameters'])

        tech_dict, bio_dict = {}, {}

        for iparam in range(num_params):

            if act['parameters'][iparam]['name'] in use_parameters:

                tech_data, bio_data = [], []

                act_copy = deepcopy(act)
                act_copy['parameters'][iparam]['amount'] *= const_factor
                params = parameter_set_for_activity(act_copy, iterations=1, stochastic=False)
                params_static = parameter_set_for_activity(act, iterations=1, stochastic=False)

                for iexc in use_exchanges:
                    exc = act['exchanges'][iexc]
                    tech_data, bio_data = add_tech_bio_data(
                        tech_data, bio_data, lookup_cache, exc, act_copy, params, params_static
                    )

                if len(tech_data) > 0:
                    tech_dict.update({act['parameters'][iparam]['name']: tech_data})
                if len(bio_data) > 0:
                    bio_dict.update({act['parameters'][iparam]['name']: bio_data})

        if len(tech_dict) > 0:
            technosphere[act['code']] = tech_dict
        if len(bio_dict) > 0:
            biosphere[act['code']] = bio_dict

    return technosphere, biosphere


def generate_local_sa_datapackage(input_data, const_factor=10.0):

    technosphere, biosphere = get_parameterized_local_sa_values(input_data, const_factor)

    tindices, tdata, tflip = get_dp_arrays(technosphere)
    bindices, bdata, bflip = get_dp_arrays(biosphere)

    name = f"local-sa-{const_factor:.0e}-ecoinvent-parameterization"
    name_short = "local-sa-ecoinvent-parameterization"

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}.zip"), write=True),
        name=f"{name}",
        sequential=True,
    )

    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=tdata,
        name=f"{name_short}-tech",
        indices_array=tindices,
        flip_array=tflip,
    )

    dp.add_persistent_array(
        matrix="biosphere_matrix",
        data_array=bdata,
        name=f"{name_short}-bio",
        indices_array=bindices,
        flip_array=bflip,
    )

    dp.finalize_serialization()


def generate_parameterized_exchanges_datapackage(name, num_samples=SAMPLES, seed=42):

    ei_raw_data = get_ecoinvent_raw_data(FILEPATH)
    tech_data, bio_data, params_data = get_parameterized_values(ei_raw_data, num_samples=num_samples, seed=seed)

    # Create datapackage with parameters values
    pdp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}-parameters-{seed}.zip"), write=True),
        name=f"parameters-{name}",
        seed=seed,
        sequential=True,
    )

    pdp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=np.vstack(list(params_data.values())),
        name=f"{name}-parameters",
        indices_array=np.array(list(params_data), dtype=PARAMS_DTYPE),
        flip_array=np.ones(len(params_data), dtype=bool),
    )

    # Create datapackage with parameterized exchanges
    edp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}-exchanges-{seed}.zip"), write=True),
        name=f"{name}",
        seed=seed,
        sequential=True,
    )

    indices = np.empty(len(tech_data), dtype=bwp.INDICES_DTYPE)
    indices[:] = [x for x, y, z, _ in tech_data]

    edp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=np.vstack([y for x, y, z, _ in tech_data]),
        name=f"{name}-tech",
        indices_array=indices,
        flip_array=np.hstack([z for x, y, z, _ in tech_data]),
    )

    indices = np.empty(len(bio_data), dtype=bwp.INDICES_DTYPE)
    indices[:] = [x for x, y, z, _ in bio_data]

    edp.add_persistent_array(
        matrix="biosphere_matrix",
        data_array=np.vstack([y for x, y, z, _ in bio_data]),
        name=f"exchanges-{name}-bio",
        indices_array=indices,
        flip_array=np.hstack([z for x, y, z, _ in bio_data]),
    )

    return pdp, edp


def create_static_datapackage(
        name, indices_tech=None, data_tech=None, flip_tech=None, indices_bio=None, data_bio=None, seed=42,
):

    dp = bwp.create_datapackage(
        name=f"validation.{name}.static",
        seed=seed,
        sequential=True
    )

    if indices_tech is not None:
        dp.add_persistent_vector(
            matrix="technosphere_matrix",
            data_array=data_tech,
            # Resource group name that will show up in provenance
            name=f"{name}-tech",
            indices_array=indices_tech,
            flip_array=flip_tech,
        )

    if indices_bio is not None:
        dp.add_persistent_vector(
            matrix="biosphere_matrix",
            data_array=data_bio,
            # Resource group name that will show up in provenance (?)
            name=f"{name}-bio",
            indices_array=indices_bio,
        )

    return dp


def create_static_data(indices):

    acts = get_activities_from_indices(indices)

    data = []
    for i, inds in enumerate(indices):
        act = bd.get_activity(int(inds['col']))
        row = inds['row']
        for exc in acts[act]:
            if exc.input.id == row:
                # TODO remove hardcoded values from here
                # problem: these two activities have
                if not ((exc.output.id == exc.input.id in [8127, 19669]) and exc['type'] == 'production'):
                    data.append(exc.amount)
                    continue
    data = np.array(data)
    assert len(data) == len(indices)
    return data


def get_activities_from_indices(indices):

    bd.projects.set_current("GSA for archetypes")
    activities = {}

    if indices is not None:

        cols = sorted(set(indices['col']))
        for col in cols:

            rows = sorted(indices[indices['col'] == col]['row'])
            act = bd.get_activity(int(col))

            exchanges = []
            for exc in act.exchanges():
                if exc.input.id in rows:
                    exchanges.append(exc)

            if len(exchanges) > 0:
                activities[act] = exchanges

    return activities


def get_exchanges_mask_from_parameters(indices_params, mask, indices_tech, indices_bio):

    use_cols = indices_params[mask]['col']

    mask_tech = np.zeros(len(indices_tech), dtype=bool)
    mask_bio = np.zeros(len(indices_bio), dtype=bool)

    for col in use_cols:

        mask_tech_current = indices_tech['col'] == col
        mask_tech = np.logical_or(mask_tech, mask_tech_current)

        mask_bio_current = indices_bio['col'] == col
        mask_bio = np.logical_or(mask_bio, mask_bio_current)

    return mask_tech, mask_bio


def collect_tech_and_bio_datapackages(name, dp_tech, dp_bio):

    seed_tech = dp_tech.metadata['seed']
    seed_bio = dp_tech.metadata['seed']
    assert seed_tech == seed_bio

    seed = seed_tech

    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}-{seed}.zip"), write=True),
        name=f"validation.{name}.static",
        seed=seed,
        sequential=True,
    )

    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=dp_tech.data[1],
        # Resource group name that will show up in provenance
        name=f"{name}-tech",
        indices_array=dp_tech.data[0],
        flip_array=dp_tech.data[2],
    )

    dp.add_persistent_array(
        matrix="biosphere_matrix",
        data_array=dp_bio.data[1],
        # Resource group name that will show up in provenance (?)
        name=f"{name}-bio",
        indices_array=dp_bio.data[0],
    )

    return dp


def generate_validation_datapackages(indices, mask, num_samples, seed=42):

    name = "ecoinvent-parameterization"

    bd.projects.set_current("GSA for archetypes")

    dp_validation_all = generate_parameterized_exchanges_datapackage(
        name=f"validation.{name}.all", num_samples=num_samples, seed=seed,
    )

    dp_dynamic_tech = dp_validation_all.groups[f'validation.{name}.all-tech']
    dp_dynamic_bio = dp_validation_all.groups[f'validation.{name}.all-bio']

    indices_tech = dp_dynamic_tech.data[0]
    indices_bio = dp_dynamic_bio.data[0]

    data_tech = create_static_data(indices_tech)
    data_bio = create_static_data(indices_bio)

    mask_tech, mask_bio = get_exchanges_mask_from_parameters(indices, mask, indices_tech, indices_bio)

    dp_static = create_static_datapackage(
        name=name,
        indices_tech=indices_tech,
        data_tech=data_tech,
        flip_tech=dp_dynamic_tech.data[2],
        indices_bio=dp_dynamic_bio.data[0],
        data_bio=data_bio,
    )

    dp_validation_inf_tech = merge_datapackages_with_mask(
        first_dp=dp_dynamic_tech,
        first_resource_group_label=f'{name}-tech',
        second_dp=dp_static,
        second_resource_group_label=f'validation-{name}-static',
        mask_array=mask_tech,
    )

    dp_validation_inf_bio = merge_datapackages_with_mask(
        first_dp=dp_dynamic_bio,
        first_resource_group_label=f'{name}-bio',
        second_dp=dp_static,
        second_resource_group_label=f'validation-{name}-static',
        mask_array=mask_bio,
    )

    dp_validation_inf = collect_tech_and_bio_datapackages(
        f"validation.{name}.inf", dp_validation_inf_tech, dp_validation_inf_bio
    )

    return dp_validation_all, dp_validation_inf


if __name__ == "__main__":

    bd.projects.set_current("GSA for archetypes")

    # Generate datapackages for high-dimensional screening
    random_seeds = [85, 86]
    num_samples = 15000
    for random_seed in random_seeds:
        print(f"Random seed {random_seed}")
        parameters_dp, exchanges_dp = generate_parameterized_exchanges_datapackage(
            "ecoinvent-parameterization", num_samples, random_seed
        )
        parameters_dp.finalize_serialization()
        exchanges_dp.finalize_serialization()

    # print("Generating local SA datapackage")
    # generate_local_sa_datapackage(ei_raw_data, const_factor=10.0)
    # generate_local_sa_datapackage(ei_raw_data, const_factor=0.1)

    # fp_parameters = DATA_DIR / "ecoinvent-parameters.pickle"
    # parameters = read_pickle(fp_parameters)

    # write_dir = Path("/Users/akim/PycharmProjects/akula/dev/write_files/gsa_for_archetypes/"
    #                  "ch_hh_average_consumption_aggregated_years_151617")
    # fp_local_sa = write_dir / "local_sa.ecoinvent-parameterization.pickle"
    # local_sa = read_pickle(fp_local_sa)
    # pindices = np.array(list(local_sa), dtype=PARAMS_DTYPE)
    # pmask = np.random.randint(0, 2, len(pindices), dtype=bool)
    # dpvall, dpvinf = generate_validation_datapackages(pindices, pmask, 10, seed=42)

    print("")
