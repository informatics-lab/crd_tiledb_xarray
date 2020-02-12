from xarray import coding, conventions
from xarray.core import indexing
from xarray.core.utils import FrozenDict, HiddenKeyDict
from xarray.core.variable import Variable
from xarray.core.utils import NdimSizeLenMixin
from xarray.backends.common import AbstractDataStore
import os.path
import json

from .hdf5_tiledb_writer import TileDBDataSetBuilder

_DIMENSION_KEY = "DIMENSION_LIST"


def _get_tiledb_dims_and_attrs(tilebd_obj, dimension_key):
    # # Zarr arrays do not have dimenions. To get around this problem, we add
    # # an attribute that specifies the dimension. We have to hide this attribute
    # # when we send the attributes to the user.
    # # zarr_obj can be either a zarr group or zarr array
    # try:
    #     dimensions = zarr_obj.attrs[dimension_key]
    # except KeyError:
    #     raise KeyError(
    #         "Zarr object is missing the attribute `%s`, which is "
    #         "required for xarray to determine variable dimensions." % (dimension_key)
    #     )

    # TODO: have assumed array but could be group...
    import tiledb
    try:
        with tiledb.open(tilebd_obj, 'r') as A:
            encoded_meta = A.meta.items()
    except:
        raise RuntimeError(f"Handel groups here? {tiledb_obj}")

    meta = {k: json.loads(v) for k, v in encoded_meta}

    dimensions = meta.get(dimension_key, [])
    flat_dims = []
    for dim in dimensions:
        dim_name = None
        if isinstance(dim, (str,)):
            dim_name = dim
        elif len(dim) == 1:
            dim_name = dim[0]
        else:
            raise RuntimeError(f"Don't know how to handel len(1) dim: {dim}")

        dim_name = dim_name[1:] if dim_name[0] == '/' else dim_name
        flat_dims.append(dim_name)
    attributes = HiddenKeyDict(meta, [dimension_key])
    return flat_dims, attributes


class LazyTileDB(NdimSizeLenMixin, indexing.ExplicitlyIndexed):
    __slots__ = ['_arr', 'shape', 'dtype', 'attr']

    # This needs more thinking about:
    # * Do we want a custom indexer.
    # * optimising and more...
    def __init__(self, arr):
        import tiledb

        self._arr = arr
        # TODO: currently assume that attrs is the basename but this needs to be explained
        self.attr = os.path.basename(arr)

        with tiledb.open(arr, 'r') as A:
            self.shape = tuple(i+1 for _, i in A.nonempty_domain())
            self.dtype = A.dtype

    def __getitem__(self, key):
        # TODO: need to think more about indexing.
        import tiledb
        if not isinstance(key, indexing.BasicIndexer):
            raise NotImplementedError(f"Only know how to deal with indexing.BasicIndexer got {key}")
        slices = key.tuple

        bounded_slices = []
        for i, s in enumerate(slices):
            if not isinstance(s, slice):
                bounded_slices.append(s)
            else:
                bounded_slices.append(
                    slice(
                        0 if s.start is None else s.start,
                        self.shape[i] if s.stop is None else s.stop,
                        s.step
                    ))

        with tiledb.open(self._arr, 'r') as A:

            return A[tuple(bounded_slices)][self.attr]


class TileDBStore(AbstractDataStore):
    """Store for reading and writing data via zarr
    """

    __slots__ = (
        "group_location", "_items"
    )

    @classmethod
    def open_group(
        cls,
        group_location
    ):

        return cls(group_location)

    def __init__(self, group_location):
        import tiledb
        self.group_location = group_location

        items = {}

        def walker(path, obj_type):
            collection = items.get(obj_type, {})
            # TODO: not handling nesting in groups as basename would ignore group.
            collection.update({os.path.basename(path): path})
            items[obj_type] = collection

        tiledb.walk(group_location, walker)

        if not items.get('array', False):
            raise RuntimeError(f'There was no tiledb arrays at/in group {group_location}')
        self._items = items

    def open_store_variable(self, name, tiledb_array):
        # TODO: What / why was LazilyOuterIndexedArray being used?
        # data = indexing.LazilyOuterIndexedArray(ZarrArrayWrapper(name, self))
        dimensions, attributes = _get_tiledb_dims_and_attrs(tiledb_array, _DIMENSION_KEY)
        attributes = dict(attributes)
        # encoding = {
        #     "chunks": zarr_array.chunks,
        #     "compressor": zarr_array.compressor,
        #     "filters": zarr_array.filters,
        # }
        # _FillValue needs to be in attributes, not encoding, so it will get
        # picked up by decode_cf

        # TODO: fill value???
        # if getattr(attributes, "fill_value") is not None:
        #     attributes["_FillValue"] = zarr_array.fill_value
        data = indexing.LazilyOuterIndexedArray(LazyTileDB(tiledb_array))
        return Variable(dimensions, data, attributes)

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self._items['array'].items()
        )

    def get_attrs(self):
        try:
            with open(os.path.join(self.group_location, 'attrs.json'), 'r') as fp:
                attrs = json.load(fp)
            return dict(attrs)
        except (OSError, IOError):
            # TODO: currently assuming this means no attrs. Maybe it's just an error....
            return dict()

    def get_dimensions(self):
        dimensions = {}
        for k, v in self.ds.arrays():
            try:
                for d, s in zip(v.attrs[_DIMENSION_KEY], v.shape):
                    if d in dimensions and dimensions[d] != s:
                        raise ValueError(
                            "found conflicting lengths for dimension %s "
                            "(%d != %d)" % (d, s, dimensions[d])
                        )
                    dimensions[d] = s

            except KeyError:
                raise KeyError(
                    "TileDB object is missing the attribute `%s`, "
                    "which is required for xarray to determine "
                    "variable dimensions." % (_DIMENSION_KEY)
                )
        return dimensions

    def sync(self):
        pass


def open_tiledb(tiledb_group):

    def maybe_decode_store(store, lock=False):
        # TODO: THINK ON ALL THESE OPTIONS. ESPECIALLY FILL
        ds = conventions.decode_cf(
            store
            # mask_and_scale=mask_and_scale,
            # decode_times=decode_times,
            # concat_characters=concat_characters,
            # decode_coords=decode_coords,
            # drop_variables=drop_variables
        )

        # TODO: this is where we would apply caching

        return ds

    tiledb_store = TileDBStore.open_group(
        tiledb_group
    )

    ds = maybe_decode_store(tiledb_store)

    return ds
