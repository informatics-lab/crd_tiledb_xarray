import os
import h5py

import json
import numpy as np
import tiledb
import re

from .utils import get_data_datasets_and_others

DIM_KEY = "DIMENSION_LIST"


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


class HDF5DSEncoder(HDF5AttrsEncoder):
    def __init__(self, file):
        self.hd5file = file

    def default(self, obj):
        if isinstance(obj, h5py.Dataset):
            return {
                "name": obj.name,
                "attrs": {k: v for k, v in obj.attrs.items() if not k == "REFERENCE_LIST"},
                "data": obj[()]
            }

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, h5py.Reference):
            return self.hd5file[obj]
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


class TileDBDataSetBuilderBase():
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

    def _get_path(self, group, ds_name):
        location = os.path.join(self.root, ds_name)

        if group and group is not "/":
            location = os.path.join(self.root, group, ds_name)
        return location

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

    def create_scaler(self, group, ds_name, ds):
        # TODO: this is doggy!
        location = self._get_path(group, ds_name)
        os.mkdir(location)
        with open(os.path.join(location, "value"), 'w') as fp:
            val = ds[()]
            fp.write(str(val.item()))

        with open(os.path.join(location, 'attrs.json'), 'w') as fp:
            json.dump({k: v for k, v in ds.attrs.items()}, fp,
                      default=HDF5AttrsEncoder(self.file).default)

    def close(self):
        self.file.close()


# TODO: a better way where you specify the dim valuse up front.

class MultiPhonomBuilder(TileDBDataSetBuilderBase):

    def __init__(self, tiledb_root, hdf5_file, dynamic_coords=None):
        super().__init__(tiledb_root, hdf5_file)
        self.dynamic_coords = dynamic_coords if dynamic_coords else []

    def safe_dom_name(self, name):
        return re.sub(r"[^A-z0-9_-]", "", name)

    def build(self):
        [self.data_ds, self.other_ds] = get_data_datasets_and_others(self.file)

        domains = {}
        for ds_name in self.data_ds:
            ds = self.file[ds_name]
            key = json.dumps(ds.attrs[DIM_KEY], default=HDF5AttrsEncoder(ds.file).default) + "__" + str(ds.shape)
            dom = domains.get(key, [])
            dom.append(ds_name)
            domains[key] = dom

        self.handle_group('/', self.file)

        for domain, data_sets in domains.items():
            domain_name = domain.rsplit('__', 1)[0]
            self.create_multi_array(self.safe_dom_name(domain_name), data_sets)

    def get_dynamic_dim_def(self, dim_ds, extra_bias=0):

        # TODO handel non int?

        dim_name = dim_ds.name[1:] if dim_ds.name[0] == '/' else dim_ds.name
        dim_vals = dim_ds[()]
        diffs = np.diff(dim_vals)

        all_same_distance = all([diffs[i] == diffs[i-1] for i in range(1, len(diffs))])
        assert all_same_distance, "dynamic coords must be equally spaced"

        bias = dim_vals[0] + extra_bias
        grad = diffs[0]

        print(f"coord {dim_name} == {grad} x i + {bias}")
        return (bias, grad)

    def create_multi_array(self, domain_name, data_sets):

        def get_data_type(ds):
            return {
                "float32": np.float32,
                "float64": np.float64,
                "int16": np.int16,
                "int32": np.int32,
                "int8": np.int8,
            }[ds.dtype.name]

        location = self._get_path("", domain_name)
        print(f"create multi for {domain_name} at {location}")
        os.mkdir(location)

        sample_ds = self.file[data_sets[0]]

        tile = 100  # TODO: better tiles.
        half_max = np.iinfo(np.int64).max / 2

        all_dims_attrs = []
        for i, dim in enumerate(sample_ds.attrs[DIM_KEY]):
            assert (len(dim)) == 1, "Can only deal with length one dims..."
            dim = dim[0]
            dim_ds = self.file[dim]
            dim_name = dim_ds.name[1:] if dim_ds.name[0] == '/' else dim_ds.name
            print(dim_name, self.dynamic_coords)
            print(dim_name)

            # Dynamic
            if dim_name in self.dynamic_coords:
                print('is dynamic')
                domain_indexs = [tiledb.Dim(name=dim_name, domain=(-half_max,  half_max-tile), tile=tile, dtype=np.int64) for i in range(len(sample_ds.shape))]
                bias, grad = self.get_dynamic_dim_def(dim_ds, extra_bias=half_max)

                dim_attrs = dict(dim_ds.attrs.items())
                dim_attrs.update({
                    "bias": bias,
                    "grad": grad,
                    "func": "y=mx+c"
                })

            # Non dynamic
            else:
                domain_indexs = [tiledb.Dim(name=dim_name, domain=(0,  np.iinfo(np.uint64).max-tile), tile=tile, dtype=np.uint64) for i in range(len(sample_ds.shape))]
                dim_attrs = f"Ref:{dim_name}"

            all_dims_attrs.append(dim_attrs)

        dom = tiledb.Domain(*domain_indexs)

        attrs = [tiledb.Attr(name=ds_name, dtype=get_data_type(self.file[ds_name])) for ds_name in data_sets]

        schema = tiledb.ArraySchema(domain=dom, sparse=False,
                                    attrs=attrs)

        # Create the (empty) array on disk.
        tiledb.DenseArray.create(location, schema)

        with tiledb.open(location, 'w') as A:

            #     for k, v in ds.attrs.items():
            #         encoded_v = json.dumps(v, default=HDF5DSEncoder(self.file).default)
            #         A.meta[k] = encoded_v

            #     # if 'DIMENSION_LIST' not in ds.attrs.keys():
            #     #     if len(ds.shape) != 1:
            #     #         raise RuntimeError(f'No "DIMENSION_LIST" but shape!= 1 {ds_name}, {ds.shape}')
            #     #     A.meta['DIMENSION_LIST'] = json.dumps([[ds_name]])

            data_domain = tuple([slice(0, i, None) for i in sample_ds.shape])
        #     data = ds[()]

            A[data_domain] = {ds_name: self.file[ds_name][()] for ds_name in data_sets}


class TileDBDataSetBuilderDimsInAttrs(TileDBDataSetBuilderBase):

    def build(self):
        # Build a list of things to exclude
        exclude = set()

        def excluder(name, item):
            for k, v in item.attrs.items():
                if k == 'bounds':
                    exclude.add(v.decode())
                elif k == "REFERENCE_LIST":
                    exclude.add(name)
                elif k == "CLASS" and v.decode() == "DIMENSION_SCALE":
                    exclude.add(name)
                elif k == "coordinates":
                    for n in v.decode().split(" "):
                        exclude.add(n)

        self.file.visititems(excluder)

        # Visit the rest;
        print("Exclude:", exclude)

        def visit(name, obj):
            if name not in exclude:
                print(f"{name} not in {exclude}")
                self.visit(name, obj)

        self.handle_group('/', self.file)
        self.file.visititems(visit)

    def create_array(self, group, ds_name, ds):
        location = self._get_path(group, ds_name)
        os.mkdir(location)

        tile = 100
        domain_indexs = [tiledb.Dim(name=f"d{i}", domain=(0,  np.iinfo(np.uint64).max-tile), tile=tile, dtype=np.uint64) for i in range(len(ds.shape))]

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
                encoded_v = json.dumps(v, default=HDF5DSEncoder(self.file).default)
                A.meta[k] = encoded_v

            # if 'DIMENSION_LIST' not in ds.attrs.keys():
            #     if len(ds.shape) != 1:
            #         raise RuntimeError(f'No "DIMENSION_LIST" but shape!= 1 {ds_name}, {ds.shape}')
            #     A.meta['DIMENSION_LIST'] = json.dumps([[ds_name]])

            data_domain = tuple([slice(0, i, None) for i in ds.shape])
            data = ds[()]

            A[data_domain] = data


class TileDBDataSetBuilderClassic(TileDBDataSetBuilderBase):

    def create_array(self, group, ds_name, ds):
        location = self._get_path(group, ds_name)
        os.mkdir(location)

        tile = 100
        domain_indexs = [tiledb.Dim(name=f"d{i}", domain=(0,  np.iinfo(np.uint64).max-tile), tile=tile, dtype=np.uint64) for i in range(len(ds.shape))]

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
