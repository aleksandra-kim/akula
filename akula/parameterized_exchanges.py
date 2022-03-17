import re
import traceback
from numbers import Number
from pathlib import Path

import bw2data as bd
import bw2io as bi
import bw2parameters as bwp
import bw_processing as bp
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
# FILEPATH = "/Users/cmutel/Documents/lca/Ecoinvent/3.8/cutoff/datasets"
FILEPATH = "/Users/akim/Documents/LCA_files/ecoinvent_38_cutoff/datasets"

DATA_DIR = Path(__file__).parent.resolve() / "data"
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


class PatchedParameterSet(bwp.ParameterSet):
    def evaluate_monte_carlo(self, iterations=1000):
        """Evaluate each formula using Monte Carlo and variable uncertainty data, if present.

        Formulas **must** return a one-dimensional array, or ``BroadcastingError`` is raised.

        Returns dictionary of ``{parameter name: numpy array}``."""
        interpreter = Interpreter()
        result = {}

        def get_rng_sample(obj):
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

        def fix_shape(array):
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

        for key in self.order:
            if key in self.global_params:
                interpreter.symtable[key] = result[key] = get_rng_sample(
                    self.global_params[key]
                )
            elif self.params[key].get("formula"):
                sample = fix_shape(interpreter(self.params[key]["formula"]))
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
                interpreter.symtable[key] = result[key] = get_rng_sample(
                    self.params[key]
                )
        return result


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


def stochastic_parameter_set_for_activity(act, iterations=250):
    ps = PatchedParameterSet(reformat_parameters(act))
    return ps.evaluate_monte_carlo(iterations=iterations)


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


def get_parameterized_values(input_data, num_samples=25000):
    tech_data, bio_data = [], []
    found, errors, unreasonable, missing = 0, 0, 0, 0

    lookup_cache = {
        (x, y): z
        for x, y, z in AD.select(AD.database, AD.code, AD.id)
        .where(AD.database << ("biosphere3", "ecoinvent 3.8 cutoff"))
        .tuples()
    }

    error_log = open("error.log", "w")
    missing_reference_log = open("undefined_reference.log", "w")

    for act in tqdm(input_data):
        if any(exc.get("formula") for exc in act["exchanges"]):
            try:
                params = stochastic_parameter_set_for_activity(
                    act, iterations=num_samples
                )
                if check_that_parameters_are_reasonable(act, params):
                    found += 1

                    for exc in act["exchanges"]:
                        if not exc.get("formula"):
                            continue
                        if exc["input"][0] == "ecoinvent 3.8 cutoff":
                            tech_data.append(
                                (
                                    (
                                        lookup_cache[exc["input"]],
                                        lookup_cache[(act["database"], act["code"])],
                                    ),
                                    params[exc["parameter_name"]],
                                    exc["type"] != "production",
                                )
                            )
                        else:
                            bio_data.append(
                                (
                                    (
                                        lookup_cache[exc["input"]],
                                        lookup_cache[(act["database"], act["code"])],
                                    ),
                                    params[exc["parameter_name"]],
                                    False,
                                )
                            )
                else:
                    unreasonable += 1
            except (ValueError, SyntaxError, bwp.errors.DuplicateName):
                error_log.write(act["filename"] + "\n")
                traceback.print_exc(file=error_log)
                errors += 1
            except bwp.errors.ParameterError:
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


def get_parameterized_local_sa_values(input_data):
    tech_data, bio_data = [], []

    lookup_cache = {
        (x, y): z
        for x, y, z in AD.select(AD.database, AD.code, AD.id)
        .where(AD.database << ("biosphere3", "ecoinvent 3.8 cutoff"))
        .tuples()
    }

    for act in tqdm(input_data):
        if any(exc.get("formula") for exc in act["exchanges"]):
            try:
                params = stochastic_parameter_set_for_activity(
                    act, iterations=1
                )
                if check_that_parameters_are_reasonable(act, params):

                    for exc in act["exchanges"]:
                        if not exc.get("formula"):
                            continue
                        if exc["input"][0] == "ecoinvent 3.8 cutoff":
                            tech_data.append(
                                (
                                    (
                                        lookup_cache[exc["input"]],
                                        lookup_cache[(act["database"], act["code"])],
                                    ),
                                    params[exc["parameter_name"]],
                                    exc["type"] != "production",
                                )
                            )
                        else:
                            bio_data.append(
                                (
                                    (
                                        lookup_cache[exc["input"]],
                                        lookup_cache[(act["database"], act["code"])],
                                    ),
                                    params[exc["parameter_name"]],
                                    False,
                                )
                            )
            except (ValueError, SyntaxError, bwp.errors.DuplicateName):
                pass
            except bwp.errors.ParameterError:
                pass

    print(
        """Parameterized exchanges statistics:
    {} technosphere exchanges
    {} biosphere exchanges""".format(
            len(tech_data), len(bio_data)
        )
    )

    return sorted(tech_data), sorted(bio_data)


def generate_parameterized_exchanges_datapackage(tech_data, bio_data):
    dp = bp.create_datapackage(
        fs=ZipFS(str(DATA_DIR / "ecoinvent-parameterization.zip"), write=True),
        name="ecoinvent-parameterization",
        seed=42,
    )

    indices = np.empty(len(tech_data), dtype=bp.INDICES_DTYPE)
    indices[:] = [x for x, y, z in tech_data]

    dp.add_persistent_array(
        matrix="technosphere_matrix",
        data_array=np.vstack([y for x, y, z in tech_data]),
        name="ecoinvent-parameterization-tech",
        indices_array=indices,
        flip_array=np.hstack([z for x, y, z in tech_data]),
    )

    indices = np.empty(len(bio_data), dtype=bp.INDICES_DTYPE)
    indices[:] = [x for x, y, z in bio_data]

    dp.add_persistent_array(
        matrix="biosphere_matrix",
        data_array=np.vstack([y for x, y, z in bio_data]),
        name="ecoinvent-parameterization-bio",
        indices_array=indices,
        flip_array=np.hstack([z for x, y, z in bio_data]),
    )

    dp.finalize_serialization()


if __name__ == "__main__":
    bd.projects.set_current("GSA for archetypes")

    print("Importing ecoinvent to get exchange parameterization data")
    ei_raw_data = get_ecoinvent_raw_data(FILEPATH)

    td, bd = get_parameterized_local_sa_values(ei_raw_data)

    # print("Generating parameterized values")
    # td, bd = get_parameterized_values(ei_raw_data, num_samples=SAMPLES)
    #
    # print("Writing datapackage")
    # generate_parameterized_exchanges_datapackage(td, bd)
