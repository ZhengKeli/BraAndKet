def structured_iter(structure):
    if isinstance(structure, (list, tuple)):
        for item in structure:
            for sub_item in structured_iter(item):
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