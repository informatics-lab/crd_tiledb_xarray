import os
import h5py

import json
import numpy as np
import tiledb


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

    def __get_path(self, group, ds_name):
        location = os.path.join(self.root, ds_name)

        if group and group is not "/":
            location = os.path.join(self.root, group, ds_name)
        return location

    def create_scaler(self, group, ds_name, ds):
        # TODO: this is doggy!
        location = self.__get_path(group, ds_name)
        os.mkdir(location)
        with open(os.path.join(location, "value"), 'w') as fp:
            val = ds[()]
            fp.write(str(val.item()))

        with open(os.path.join(location, 'attrs.json'), 'w') as fp:
            json.dump({k: v for k, v in ds.attrs.items()}, fp,
                      default=HDF5AttrsEncoder(self.file).default)

    def create_array(self, group, ds_name, ds):
        location = self.__get_path(group, ds_name)
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

    def close(self):
        self.file.close()
