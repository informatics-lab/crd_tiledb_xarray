
import datetime
import glob
import shutil
import iris
import os
import numpy as np
import tiledb

import json


def cubes_into_datasets(cubes):
    datasets = []
    for cube in cubes:
        for ds in datasets:
            if ds.domain.in_domain(cube):
                ds.add_cube(cube)
                break
        else:
            ds = Dataset([cube])
            datasets.append(ds)

    return datasets


def coords_same(coords1, coords2):
    if not len(coords2) == len(coords1):
        return False

    for i in range(len(coords1)):
        if not coords1[i] == coords2[i]:
            return False
    else:
        return True


class Domain():
    def __init__(self, coords):
        self.coords = coords
        super().__init__()

    def in_domain(self, cube):
        return coords_same(self.coords, cube.dim_coords)

    @property
    def name(self):
        return '_'.join([f"{coord.name()}_{str(coord.shape)}" for coord in self.coords])


class Dataset():
    def __init__(self, cubes):
        super().__init__()
        eg_cube = cubes[0]
        assert eg_cube.attributes.get('STASH', None) is not None

        self._cubes = [eg_cube]
        self.domain = Domain(eg_cube.dim_coords)

        for cube in cubes[1:]:
            self.add_cube(cube)

    def add_cube(self, cube):
        assert self.domain.in_domain(cube)
        assert cube.attributes.get('STASH', None) is not None
        self._cubes.append(cube)

    @property
    def cubes(self):
        return list(sorted(self._cubes, key=lambda c: str(c.attributes['STASH'])))

    @property
    def key(self):
        return "_".join(str(c.attributes['STASH']) for c in self.cubes)


def stash(cube):
    return str(cube.attributes['STASH'])


class TileDBBuilder():
    def __init__(self, datasets, tiledb_root):
        super().__init__()
        self.datasets = datasets
        self.root = tiledb_root

        dom_width = np.iinfo(np.int64).max
        self._dom_width = dom_width if dom_width % 2 == 0 else dom_width - 1

    def build(self):
        tiledb.group_create(self.root)

        for ds in datasets:
            group_path = os.path.join(self.root, ds.key)
            tiledb.group_create(group_path)

            for coord in ds.domain.coords:
                self.create_coord_array(group_path, coord)

            self.create_data_array(group_path, ds)

    def create_data_array(self, group_path, ds):
        data = {stash(c): c.data for c in ds.cubes}

        path = os.path.join(group_path, "data")

        all_attrs = {}
        for c in ds.cubes:
            attrs = {k: v for k, v in c.attributes.items()}
            attrs['STASH'] = str(attrs['STASH'])
            attrs['units'] = str(c.units)
            attrs['name'] = c.name()
            all_attrs[stash(c)] = attrs

        all_attrs['dimensions'] = [coord.name() for coord in ds.domain.coords]

        log(f"insert into {path.split('/')[-2]} shape {c.shape}")
        self.create_array(path, data, all_attrs)

    def create_coord_array(self, group, coord):
        name = coord.name()
        path = os.path.join(self.root, group, name)
        self.create_array(path, {name: coord.points}, {'units': str(coord.units)})

    def create_array(self, path, data, meta=None):
        domain_indexes = []
        eg_data = next(iter(data.values()))
        for i, i_len in enumerate(eg_data.shape):
            dim = tiledb.Dim(
                name=f"dim_{i}",
                domain=(-self._dom_width/2,  self._dom_width-i_len),
                tile=i_len,
                dtype=np.int64
            )
            domain_indexes.append(dim)

        dom = tiledb.Domain(*domain_indexes)

        attrs = []
        for name, data_arr in data.items():
            attrs.append(tiledb.Attr(name=name, dtype=data_arr.dtype))

        schema = tiledb.ArraySchema(domain=dom, sparse=False,
                                    attrs=attrs)

        tiledb.DenseArray.create(path, schema)
        with tiledb.open(path, 'w') as A:
            data_domain = tuple([slice(0, i, None) for i in eg_data.shape])
            A[data_domain] = data

            for k, v in meta.items():
                A.meta[k] = json.dumps(v)


def get_dynamic_dim_def(self, dim_ds, extra_bias=0):

    dim_name = dim_ds.name[1:] if dim_ds.name[0] == '/' else dim_ds.name
    dim_vals = dim_ds[()]
    diffs = np.diff(dim_vals)

    all_same_distance = all([diffs[i] == diffs[i-1] for i in range(1, len(diffs))])
    assert all_same_distance, "dynamic coords must be equally spaced"

    bias = dim_vals[0] + extra_bias
    grad = diffs[0]

    print(f"coord {dim_name} == {grad} x i + {bias}")
    return (bias, grad)


def linear_arr_extend(tiledb_path, amount):
    with tiledb.open(tiledb_path, 'r') as A:
        nonempty = A.nonempty_domain()
        assert len(nonempty) == 1, "can only extend 1d datasets"
        selection = tuple(slice(start, stop) for start, stop in nonempty)

        data_set = A[selection]
        attr = next(iter(data_set.keys()))
        data = data_set[attr]

    gradient = set(np.diff(data))
    assert len(gradient) == 1, f"Non linear arr {gradient}"
    gradient = next(iter(gradient))
    bias = data[0]

    new_data = np.array([i*gradient + bias for i in range(amount)], dtype=data.dtype)

    assert np.all(new_data[0:len(data)] == data), "Data equality safety check failed..."

    with tiledb.open(tiledb_path, 'w') as A:
        selection = (slice(nonempty[0][0], nonempty[0][0]+amount),)
        A[selection] = {attr: new_data}


class TileDBAppender():
    def __init__(self, dataset, tiledb_root):
        super().__init__()

        self.root = os.path.join(tiledb_root, dataset.key)
        assert os.path.exists(self.root), f"The dataset must already exist in tiledb_root {tiledb_root}"
        self.dataset = dataset

    def build(self):
        self.offsets = self._offsets()
        self._insert()

    def _offsets(self):
        offsets = []
        for coord in self.dataset.domain.coords:
            arr = os.path.join(self.root, coord.name())
            coord_offset = self._get_offset_of_coord(coord, arr)
            offsets.append(coord_offset)
        return offsets

    def _insert(self):
        data = {stash(c): c.data for c in self.dataset.cubes}
        data_shape = next(iter(data.values())).shape
        log(f"insert into {self.root.split('/')[-1]} shape {data_shape}")
        selection = tuple(
            slice(offset, offset + data_shape[i]) for i, offset in enumerate(self.offsets)
        )
        with tiledb.open(os.path.join(self.root, 'data',), 'w') as A:
            A[selection] = data

    def _get_offset_of_coord(self, coord, tiledb_array):
        assert len(coord.shape) == 1, f"Can only process 1D datasets here. Shape is {ds.shape} for {ds}"

        with tiledb.open(tiledb_array, 'r') as A:

            start, stop = A.nonempty_domain()[0]
            found_first_index = None
            cur = start
            step = 3000
            find = coord.points
            ds_name = coord.name()
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
                    found_first_index = found_first_item[0] + cur
                    break
                else:
                    cur += step

            if found_first_index is None:
                raise ValueError("Could not find the data {find[0:10]}... in {tiledb_array}")

            test_slice = slice(found_first_index, found_first_index + len(find))
            test_data = A[(test_slice,)][tiledb_key]

            if not (test_data == find).all():
                raise ValueError("could not find the data. Found first item but following items didn't match")

        return found_first_index


if __name__ == "__main__":
    def log(*args):
        print(datetime.datetime.now(), *args)

    NC_SOURCE_DIR = "/Users/theo/repos/crd_tiledb_xarray/data/6hrly/"
    files = [os.path.join(NC_SOURCE_DIR, f) for f in sorted(os.listdir(NC_SOURCE_DIR))]
    # part1_files, part2_files = files[:1], files[1:]
    # file_groups = [files[i:i+2] for i in range(0, len(files), 2)]

    TILEDB_PATH = '/Users/theo/repos/crd_tiledb_xarray/data/tmp.tiledb'
    shutil.rmtree(TILEDB_PATH, ignore_errors=True)

    log('load in iris')
    first_file = files.pop(0)
    cubes = iris.load(first_file)
    log('sort into domains')
    datasets = cubes_into_datasets(cubes)
    log('Build tiledb')
    builder = TileDBBuilder(datasets, TILEDB_PATH)
    builder.build()
    log('Built')

    #  Extend in time....
    log('expand times')
    time_arrs = glob.glob(TILEDB_PATH+'/**/time')
    for arr in time_arrs:
        linear_arr_extend(arr, 1000)
        log('Done', os.path.basename(arr))

    # Append the rest
    # for file in part2_files:
    log('load rest of the cubes ')
    for i, file in enumerate(files):
        log(f"add file {i}")
        cubes = iris.load(file)
        datasets = cubes_into_datasets(cubes)
        for dataset in datasets:
            log(f'   add ds {dataset.key}')

            appender = TileDBAppender(dataset, TILEDB_PATH)
            appender.build()

    log("Fin")
