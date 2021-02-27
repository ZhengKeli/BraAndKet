def structured_iter(structure):
    try:
        structure_iter = iter(structure)
    except TypeError:
        structure_iter = None
    if structure_iter is not None:
        for structure_item in structure_iter:
            for sub_item in structured_iter(structure_item):
                yield sub_item
    else:
        yield structure


def structured_map(structure, map_func):
    if isinstance(structure, list):
        return [structured_map(item, map_func) for item in structure]
    elif isinstance(structure, tuple):
        return tuple(structured_map(item, map_func) for item in structure)
    else:
        return map_func(structure)
