import re
import traceback
from numbers import Number
from pathlib import Path
from copy import deepcopy
import bw2data as bd
import bw2io as bi
import bw2parameters as bwpa
import bw_processing as bwp
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
    def evaluate_monte_carlo(self, iterations=1000, stochastic=True):
        """Evaluate each formula using Monte Carlo and variable uncertainty data, if present.
        Formulas **must** return a one-dimensional array, or ``BroadcastingError`` is raised.
        Returns dictionary of ``{parameter name: numpy array}``."""
        interpreter = Interpreter()
        result = {}
        for key in self.order:
            if key in self.global_params:
                if stochastic:
                    interpreter.symtable[key] = result[key] = self.get_rng_sample(self.global_params[key], iterations)
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
                    interpreter.symtable[key] = result[key] = self.get_rng_sample(self.params[key], iterations)
                else:
                    interpreter.symtable[key] = result[key] = self.get_static_sample(self.params[key])
        return result

    @staticmethod
    def get_rng_sample(obj, iterations):
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
        return kls.bounded_random_variables(kls.from_dicts(obj), iterations).ravel()

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


def parameter_set_for_activity(act, iterations=250, stochastic=True,):
    ps = PatchedParameterSet(reformat_parameters(act))
    return ps.evaluate_monte_carlo(iterations=iterations, stochastic=stochastic)


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
                exchange["type"] != "production",
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


def get_parameterized_values(input_data, num_samples=25000):
    tech_data, bio_data = [], []
    found, errors, unreasonable, missing = 0, 0, 0, 0

    lookup_cache = get_lookup_cache()

    error_log = open("error.log", "w")
    missing_reference_log = open("undefined_reference.log", "w")

    for act in tqdm(input_data):
        if any(exc.get("formula") for exc in act["exchanges"]):
            try:
                params = parameter_set_for_activity(act, iterations=num_samples, stochastic=True)
                if check_that_parameters_are_reasonable(act, params):
                    found += 1

                    for exc in act["exchanges"]:
                        if not exc.get("formula"):
                            continue
                        tech_data, bio_data = add_tech_bio_data(
                            tech_data, bio_data, lookup_cache, exc, act, params,
                        )
                else:
                    unreasonable += 1
            except (ValueError, SyntaxError, bwpa.errors.DuplicateName):
                error_log.write(act["filename"] + "\n")
                traceback.print_exc(file=error_log)
                errors += 1
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

    print(
        """Parameterized exchanges statistics:
    {} technosphere exchanges
    {} biosphere exchanges""".format(
            len(tech_data), len(bio_data)
        )
    )

    return sorted(tech_data), sorted(bio_data)


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


def get_parameters(data=None):
    if FILEPATH_PARAMETERS.exists():
        parameters_list = read_pickle(FILEPATH_PARAMETERS)
    else:
        parameters_list = []
        for act in tqdm(data):
            if any(exc.get("formula") for exc in act["exchanges"]):
                try:
                    ps = PatchedParameterSet(reformat_parameters(act))
                    params_exchanges = ps.evaluate_monte_carlo(iterations=1, stochastic=False)
                    if check_that_parameters_are_reasonable(act, params_exchanges):
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
                except (ValueError, SyntaxError, bwpa.errors.DuplicateName):
                    pass
                except bwpa.errors.ParameterError:
                    pass
        write_pickle(parameters_list, FILEPATH_PARAMETERS)
    return parameters_list


def remove_unused_parameters(params_dict):
    params = list(params_dict.keys())
    formulas = {v['formula'] for p, v in params_dict.items() if "__exchange" in p}
    formulas_str = "|".join(formulas)
    for param in params:
        unct = params_dict[param].get("uncertainty type", 0)
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
        seed=42,
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


def generate_parameterized_exchanges_datapackage(tech_data, bio_data):
    name = "ecoinvent-parameterization"
    dp = bwp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / f"{name}.zip"), write=True),
        name=f"{name}",
        seed=42,
    )

    indices = np.empty(len(tech_data), dtype=bwp.INDICES_DTYPE)
    indices[:] = [x for x, y, z in tech_data]

    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=np.vstack([y for x, y, z in tech_data]),
        name=f"{name}-tech",
        indices_array=indices,
        flip_array=np.hstack([z for x, y, z in tech_data]),
    )

    indices = np.empty(len(bio_data), dtype=bwp.INDICES_DTYPE)
    indices[:] = [x for x, y, z in bio_data]

    dp.add_persistent_array(
        matrix="biosphere_matrix",
        data_array=np.vstack([y for x, y, z in bio_data]),
        name=f"{name}-bio",
        indices_array=indices,
        flip_array=np.hstack([z for x, y, z in bio_data]),
    )

    dp.finalize_serialization()


# def generate_parameters_datapackage(input_data):
#     parameters = get_parameters(input_data)
#     dp = bwp.create_datapackage(
#         fs=ZipFS(str(FILEPATH_PARAMETERS), write=True),
#         name="ecoinvent-parameters",
#         seed=42,
#     )
#     num_parameters = sum(len(p['parameters']) for p in parameters)
#     data = np.array([v['amount'] for element in parameters for v in element['parameters'].values()])
#     indices = np.empty(num_parameters, dtype=bwp.INDICES_DTYPE)
#     indices[:] = [(x, 0) for x in np.arange(1e6, 1e6+num_parameters)]
#     flip = np.ones(num_parameters, dtype=bool)
#     dp.add_persistent_vector(
#         matrix="technosphere_matrix",
#         data_array=data,
#         name="ecoinvent-parameters",
#         indices_array=indices,
#         flip_array=flip,
#     )
#     max_index = indices[-1][0]
#     data = np.array([(p['activity']["code"], len(p['parameters'])) for p in parameters])
#     num_activities = len(data)
#     indices = np.empty(num_activities, dtype=bwp.INDICES_DTYPE)
#     indices[:] = [(x, 0) for x in np.arange(max_index, max_index+num_activities)]
#     flip = np.ones(num_activities, dtype=bool)
#     dp.add_persistent_array(
#         matrix="technosphere_matrix",
#         data_array=data,
#         name="ecoinvent-parameterized-activities",
#         indices_array=indices,
#         flip_array=flip,
#     )
#     dp.finalize_serialization()


if __name__ == "__main__":
    bd.projects.set_current("GSA for archetypes")

    print("Importing ecoinvent to get exchange parameterization data")
    ei_raw_data = get_ecoinvent_raw_data(FILEPATH)

    # dp_name = "local-sa-1e+01-ecoinvent-parameterization"
    # dp = bwp.load_datapackage(ZipFS(str(DATA_DIR / f"{dp_name}.zip")))
    plist = get_parameters(ei_raw_data)

    print("Generating local SA datapackage")
    generate_local_sa_datapackage(ei_raw_data, const_factor=10.0)
    generate_local_sa_datapackage(ei_raw_data, const_factor=0.1)

    # print("Generating parameterized values")
    # td, bd = get_parameterized_values(ei_raw_data, num_samples=SAMPLES)
    # print("Writing datapackage")
    # generate_parameterized_exchanges_datapackage(td, bd)
