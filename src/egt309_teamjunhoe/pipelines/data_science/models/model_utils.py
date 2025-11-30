from skopt.space import Integer, Categorical
from typing import Any

def read_bs_search_space(search_dict: dict[str, list]) -> dict[str, Any]:
    # Read and sort categories into respective skopt spaces
    search_space = dict()
    for key in search_dict:
        # item list should all have the same dtypes for items
        item = search_dict[key]
        assert len(set(type(x) for x in item)) == 1
        if isinstance(item[0], int):
            # Length of item list should be 2 if integer and index 0 < index 1
            assert len(item) == 2 and item[0] < item[1]
            search_space[key] = Integer(item[0], item[1])
        elif isinstance(item[0], str):
            search_space[key] = Categorical(item)

    return search_space