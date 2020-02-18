from functools import partial


def excluder(to_exclude_set, name, item):
    for k, v in item.attrs.items():
        if k == 'bounds':
            to_exclude_set.add(v.decode())
        elif k == "REFERENCE_LIST":
            to_exclude_set.add(name)
        elif k == "CLASS" and v.decode() == "DIMENSION_SCALE":
            to_exclude_set.add(name)
        elif k == "coordinates":
            for n in v.decode().split(" "):
                to_exclude_set.add(n)


def get_data_datasets_and_others(open_file):
    exclude = set()
    all_items = set()

    open_file.visititems(partial(excluder, exclude))
    open_file.visit(lambda x: all_items.add(x))

    exclude.add('rotated_latitude_longitude')  # Special case I've not figured out what to do with...

    return (all_items - exclude, exclude)
