import numpy as np


def add_variances(list_, lcia_score):
    for dictionary in list_:
        values = np.vstack(list(dictionary.values()))
        values = np.hstack([values, np.ones((len(values), 1))*lcia_score])
        variances = np.var(values, axis=1)
        for i, k in enumerate(dictionary.keys()):
            dictionary[k] = {
                "arr": values[i, :],
                "var": variances[i],
            }


def get_variance_threshold(list_, num_parameters):
    # Collect all variances
    vars = np.array([value['var'] for dictionary in list_ for key, value in dictionary.items()])
    vars = np.sort(vars)[-1::-1]
    vars_threshold = vars[:num_parameters][-1]
    return vars_threshold


# Remove lowly influential
def get_indices_high_variance(dictionary, variances_threshold):
    return [key for key in dictionary if dictionary[key]['var'] >= variances_threshold]
