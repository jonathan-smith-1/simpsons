from itertools import product


def generate_configs(param_dict):
    """
    Generate exhaustive configuration dictionaries of parameters.

    Args:
        param_dict: Dictionary of parameters of the form key:value where
                    value can be a list of parameter values to try.

    Returns:
        List of dictionaries, covering all possible combination of input
        parameters.

        For example:

        generate_configs({'b': [1, 2], 'a': ['x', 'y']})

        returns:

        [{'b': 1, 'a': 'x'},
         {'b': 1, 'a': 'y'},
         {'b': 2, 'a': 'x'},
         {'b': 2, 'a': 'y'},]
    """

    values = list(product(*param_dict.values()))
    keys = param_dict.keys()

    return [dict(zip(keys, v)) for v in values]



