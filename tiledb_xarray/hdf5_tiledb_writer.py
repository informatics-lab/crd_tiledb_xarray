import os
import h5py

import json
import numpy as np
import tiledb
import re

from .misc import DIMENSION_KEY


def get_tiledb_name(ds):
    attrs = ['cell_methods', 'um_stash_source']
    name_items = [ds.name.encode('utf-8')]
    name_items += [ds.attrs.get(attr, b'') for attr in attrs]
    coords = ds.attrs.get('coordinates', b'').decode()
    if coords.find('pressure') >= 0:
        name_items += [b"at-pressure"]
    if coords.find('height') >= 0:
        name_items += [b"at-height"]
    return re.sub("[^-A-z0-9_]", "", b'__'.join(name_items).decode().replace(' ', '-'))


class HDF5AttrsEncoder(json.JSONEncoder):
    def __init__(self, file):
        self.hd5file = file

    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, h5py.Reference):
            return self.hd5file[obj].name
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
            return obj
        if isinstance(obj, (np.bytes_, bytes)):
            try:
                return obj.decode('utf-8')
            except UnicodeDecodeError:
                print(f'couldn\'t decode {obj}')
                pass

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class TileDBDataSetBuilder():
    def __init__(self, tiledb_root, hdf5_file):
        self.groups = set()
        self.root = tiledb_root
        if isinstance(hdf5_file, str):
            # TODO: worry about closing?
            hdf5_file = h5py.File(hdf5_file, 'r')
        self.file = hdf5_file

    def build(self):
        self.handle_group('/', self.file)
        self.file.visititems(self.visit)

    def visit(self, member_name, obj):
        # TODO Root attrs?
        if isinstance(obj, h5py.Group):
            self.handle_group(member_name, obj)
        elif isinstance(obj, h5py.Dataset):
            self.handle_ds(member_name, obj)
        else:
            raise RuntimeError(f"What is this {obj}")

    def handle_ds(self, ds_name, ds):
        # TODO - what if ds visited before group?
        ds_path = ds_name.rsplit('/', 1)
        if len(ds_path) == 1:
            group = None
            ds_base_name = ds_path[0]
        elif ds_path[0].strip() == '':
            group = None
            ds_base_name = ds_path[1]
        else:
            group = ds_path[0]
            ds_base_name = ds_path[1]

        print(f'make ds {ds_base_name} in group {group}')

        if len(ds.shape) == 0:
            self.create_scaler(group, ds_name, ds)
        else:
            self.create_array(group, ds_name, ds)

    def handel_var_attrs(self, ds_name, ds):
        for k, v in flux.attrs.items():

            print(f"{k}:{v}")

    def handle_group(self, group_name, group):
        print(f"group {group} type {type(group)}")
        # TODO group attrs?
        print("group name", group_name)
        path = os.path.join(self.root, group_name[1:] if group_name[0] == '/' else group_name)
        os.makedirs(path, exist_ok=False)
        tiledb.group_create(path)
        print(f"made_group {group_name} at {path}")
        with open(os.path.join(path, 'attrs.json'), 'w') as fp:
            json.dump({k: v for k, v in group.attrs.items()}, fp, default=HDF5AttrsEncoder(self.file).default)
        print(f'wrote group attrs for {group_name}')

    def __get_path(self, group, ds):
        tiledb_name = get_tiledb_name(ds)
        location = os.path.join(self.root, tiledb_name)

        if group and group is not "/":
            location = os.path.join(self.root, group, tiledb_name)
        return location

    def create_scaler(self, group, ds_name, ds):
        # TODO: this is doggy!
        location = self.__get_path(group, ds)
        os.mkdir(location)
        with open(os.path.join(location, "value"), 'w') as fp:
            val = ds[()]
            fp.write(str(val.item()))

        with open(os.path.join(location, 'attrs.json'), 'w') as fp:
            json.dump({k: v for k, v in ds.attrs.items()}, fp,
                      default=HDF5AttrsEncoder(self.file).default)

    # def __chunk_suggest(self, dtype, shape, target_size_MB=50):
    #     digit_size = {
    #         "float32": 32/4,
    #         "float64": 63/4,
    #         "int16": 16/4,
    #         "int32": 32/4,
    #         "int8": 8/4,
    #     }[dtype.name]

    #     size_bytes_MB = digit_size * np.prod(shape)/ (1000*1000)
    #     number_splits = round(size_bytes_MB / target_size_MB)

    #     if number_splits < 2:
    #         return shape

    def create_array(self, group, ds_name, ds):
        location = self.__get_path(group, ds)
        os.mkdir(location)

        dom_width = np.iinfo(np.int64).max
        dom_width = dom_width if dom_width % 2 == 0 else dom_width - 1

        tile = 100
        domain_indexs = [tiledb.Dim(name=f"d{i}", domain=(-dom_width/2,  dom_width-tile), tile=tile, dtype=np.int64) for i in range(len(ds.shape))]

        data_type = {
            "float32": np.float32,
            "float64": np.float64,
            "int16": np.int16,
            "int32": np.int32,
            "int8": np.int8,
        }[ds.dtype.name]

        # The array will be 4x4 with dimensions "d1" and "d2", with domain [1,4].
        dom = tiledb.Domain(*domain_indexs)

        # The array will be dense with a single attribute "a" so each (i,j) cell can store an integer.
        schema = tiledb.ArraySchema(domain=dom, sparse=False,
                                    attrs=[tiledb.Attr(name=ds_name, dtype=data_type)])

        # Create the (empty) array on disk.
        tiledb.DenseArray.create(location, schema)

        with tiledb.open(location, 'w') as A:

            for k, v in ds.attrs.items():
                encoded_v = json.dumps(v, default=HDF5AttrsEncoder(self.file).default)
                A.meta[k] = encoded_v

            if 'DIMENSION_LIST' not in ds.attrs.keys():
                if len(ds.shape) != 1:
                    raise RuntimeError(f'No "DIMENSION_LIST" but shape!= 1 {ds_name}, {ds.shape}')
                A.meta['DIMENSION_LIST'] = json.dumps([[ds_name]])

            data_domain = tuple([slice(0, i, None) for i in ds.shape])
            data = ds[()]

            A[data_domain] = data

    def close(self):
        self.file.close()


def extend_dim(arr_path, ds_to_insert, guess_indexes=False, from_indexes=None, at_indexes=None):
    if guess_indexes or at_indexes:
        raise NotImplementedError('from_index and guess_indexes not implemented yet')

    assert len(ds_to_insert.shape) == len(from_indexes)

    # TODO check meta data - at lest units?
    selection = tuple(
        slice(i, i+length, None) for i, length in zip(from_indexes, ds_to_insert.shape)
    )
    with tiledb.open(arr_path, 'w') as A:
        A[selection] = ds_to_insert[()]


def _get_indexes_of_values(ds, tiledb_array):
    print(f"get vals {ds[()]} from {tiledb_array}")
    assert len(ds.shape) == 1, f"Can only process 1D datasets here. Shape is {ds.shape} for {ds}"

    try:
        ds_units = ds.attrs['units'].tostring().decode()
    except (KeyError, UnicodeDecodeError, AttributeError):
        raise ValueError("Could not access units of dataset {ds} does not have units. not safe...")
    with tiledb.open(tiledb_array, 'r') as A:
        try:
            tiledb_units = json.loads(A.meta['units'])
        except (KeyError, json.decoder.JSONDecodeError):
            raise ValueError("Could not access units of tiledb {tiledb_array} does not have units. not safe...")

        if not ds_units == tiledb_units:
            raise ValueError("Could not access units of tiledb {tiledb_array} does not have units. not safe...")

        start, stop = A.nonempty_domain()[0]
        found_first_index = None
        cur = start
        step = 3000
        find = ds[()]
        ds_name = ds.name.split('/')[-1]
        while cur <= stop:
            cur_stop = cur+step if cur+step <= stop else stop
            search = A[(slice(cur, cur+step, None),)]

            # Shouldn't need to do this each loop but can't find how to get info from schema about what attr names are.
            if len(search.keys()) == 1:
                tiledb_key = next(iter(search))
            elif ds_name in search.keys():
                tiledb_key = ds_name
            else:
                raise RuntimeError(f'Dont know which bit of data should work with for {ds} in tiledb attrs {search.keys()})')

            search_arr = search[tiledb_key]
            found_first_item = np.where(search_arr == find[0])[0]
            if len(found_first_item) >= 1:
                # TODO: what about multiple finds.
                print(f'found at {found_first_item[0]} at offset {cur}')
                found_first_index = found_first_item[0] + cur
                break
            else:
                print(f"not found at offset {cur}:{cur+step}")

        if found_first_index is None:
            raise ValueError("Could not find the data {find[0:10]}... in {tiledb_array}")

        test_slice = slice(found_first_index, found_first_index + len(find))
        test_data = A[(test_slice,)][tiledb_key]

        if not (test_data == find).all():
            raise ValueError("could not find the data. Found first item but following items didn't match")

    print('done get index')
    return test_slice


def insert_dataset(ds, tiledb_root):
    index_slices = []
    for dim in ds.attrs[DIMENSION_KEY]:
        if len(dim) != 1:
            raise ValueError(f"Can only work with data sets with 1d dims. Got:{ds.attrs['_DIMENSION_KEY']}")
        dim = dim[0]
        dim_ds = ds.file[dim]

        print('Should check units... on dime but not going to...', dim_ds.attrs.get('units', None))

        dim_tiledb_arr = os.path.join(tiledb_root, get_tiledb_name(dim_ds))
        index_slices.append(_get_indexes_of_values(dim_ds, dim_tiledb_arr))
    print(index_slices)

    tiledb_arr = os.path.join(tiledb_root, get_tiledb_name(ds))
    with tiledb.open(tiledb_arr, 'w') as A:
        A[tuple(index_slices)] = ds[()]
