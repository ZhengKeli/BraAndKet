def iter_structure(structure):
    if structure is None:
        return
    try:
        structure_iter = iter(structure)
    except TypeError:
        structure_iter = None
    if structure_iter is not None:
        for structure_item in structure_iter:
            for sub_item in iter_structure(structure_item):
                yield sub_item
    else:
        yield structure


def map_structure(structure, map_func):
    if structure is None:
        return None
    if isinstance(structure, list):
        return [map_structure(item, map_func) for item in structure]
    elif isinstance(structure, tuple):
        return tuple(map_structure(item, map_func) for item in structure)
    else:
        return map_func(structure)
