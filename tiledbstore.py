from xarray.backends.common import AbstractWritableDataStore

class TileDBStore(AbstractWritableDataStore):
    """Store for reading and writing data via zarr
    """

    __slots__ = ("append_dim",
        "ds",
        "_consolidate_on_close",
        "_group",
        "_read_only",
        "_synchronizer",
    )

    @classmethod
    def open_group(
        cls,
        store,
        mode="r",
        synchronizer=None,
        group=None,
        consolidated=False,
        consolidate_on_close=False,
    ):
        import zarr

        open_kwargs = dict(mode=mode, synchronizer=synchronizer, path=group)
        if consolidated:
            # TODO: an option to pass the metadata_key keyword
            zarr_group = zarr.open_consolidated(store, **open_kwargs)
        else:
            zarr_group = zarr.open_group(store, **open_kwargs)
        return cls(zarr_group, consolidate_on_close)

    def __init__(self, zarr_group, consolidate_on_close=False):
        self.ds = zarr_group
        self._read_only = self.ds.read_only
        self._synchronizer = self.ds.synchronizer
        self._group = self.ds.path
        self._consolidate_on_close = consolidate_on_close
        self.append_dim = None

    def open_store_variable(self, name, zarr_array):
        data = indexing.LazilyOuterIndexedArray(ZarrArrayWrapper(name, self))
        dimensions, attributes = _get_zarr_dims_and_attrs(zarr_array, _DIMENSION_KEY)
        attributes = dict(attributes)
        encoding = {
            "chunks": zarr_array.chunks,
            "compressor": zarr_array.compressor,
            "filters": zarr_array.filters,
        }
        # _FillValue needs to be in attributes, not encoding, so it will get
        # picked up by decode_cf
        if getattr(zarr_array, "fill_value") is not None:
            attributes["_FillValue"] = zarr_array.fill_value

        return Variable(dimensions, data, attributes, encoding)

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.arrays()
        )

    def get_attrs(self):
        attributes = dict(self.ds.attrs.asdict())
        return attributes

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
                    "Zarr object is missing the attribute `%s`, "
                    "which is required for xarray to determine "
                    "variable dimensions." % (_DIMENSION_KEY)
                )
        return dimensions

    def set_dimensions(self, variables, unlimited_dims=None):
        if unlimited_dims is not None:
            raise NotImplementedError(
                "Zarr backend doesn't know how to handle unlimited dimensions"
            )

    def set_attributes(self, attributes):
        self.ds.attrs.put(attributes)

    def encode_variable(self, variable):
        variable = encode_zarr_variable(variable)
        return variable

    def encode_attribute(self, a):
        return _encode_zarr_attr_value(a)

    def store(
        self,
        variables,
        attributes,
        check_encoding_set=frozenset(),
        writer=None,
        unlimited_dims=None,
    ):
        """
        Top level method for putting data on this store, this method:
          - encodes variables/attributes
          - sets dimensions
          - sets variables
        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        attributes : dict-like
            Dictionary of key/value (attribute name / attribute) pairs
        check_encoding_set : list-like
            List of variables that should be checked for invalid encoding
            values
        writer : ArrayWriter
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
            dimension on which the zarray will be appended
            only needed in append mode
        """

        existing_variables = {
            vn for vn in variables if _encode_variable_name(vn) in self.ds
        }
        new_variables = set(variables) - existing_variables
        variables_without_encoding = {vn: variables[vn] for vn in new_variables}
        variables_encoded, attributes = self.encode(
            variables_without_encoding, attributes
        )

        if len(existing_variables) > 0:
            # there are variables to append
            # their encoding must be the same as in the store
            ds = open_zarr(self.ds.store, chunks=None)
            variables_with_encoding = {}
            for vn in existing_variables:
                variables_with_encoding[vn] = variables[vn].copy(deep=False)
                variables_with_encoding[vn].encoding = ds[vn].encoding
            variables_with_encoding, _ = self.encode(variables_with_encoding, {})
            variables_encoded.update(variables_with_encoding)

        self.set_attributes(attributes)
        self.set_dimensions(variables_encoded, unlimited_dims=unlimited_dims)
        self.set_variables(
            variables_encoded, check_encoding_set, writer, unlimited_dims=unlimited_dims
        )

    def sync(self):
        pass

    def set_variables(self, variables, check_encoding_set, writer, unlimited_dims=None):
        """
        This provides a centralized method to set the variables on the data
        store.
        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        check_encoding_set : list-like
            List of variables that should be checked for invalid encoding
            values
        writer :
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
        """

        for vn, v in variables.items():
            name = _encode_variable_name(vn)
            check = vn in check_encoding_set
            attrs = v.attrs.copy()
            dims = v.dims
            dtype = v.dtype
            shape = v.shape

            fill_value = attrs.pop("_FillValue", None)
            if v.encoding == {"_FillValue": None} and fill_value is None:
                v.encoding = {}
            if name in self.ds:
                zarr_array = self.ds[name]
                if self.append_dim in dims:
                    # this is the DataArray that has append_dim as a
                    # dimension
                    append_axis = dims.index(self.append_dim)
                    new_shape = list(zarr_array.shape)
                    new_shape[append_axis] += v.shape[append_axis]
                    new_region = [slice(None)] * len(new_shape)
                    new_region[append_axis] = slice(zarr_array.shape[append_axis], None)
                    zarr_array.resize(new_shape)
                    writer.add(v.data, zarr_array, region=tuple(new_region))
            else:
                # new variable
                encoding = _extract_zarr_variable_encoding(v, raise_on_invalid=check)
                encoded_attrs = {}
                # the magic for storing the hidden dimension data
                encoded_attrs[_DIMENSION_KEY] = dims
                for k2, v2 in attrs.items():
                    encoded_attrs[k2] = self.encode_attribute(v2)

                if coding.strings.check_vlen_dtype(dtype) == str:
                    dtype = str
                zarr_array = self.ds.create(
                    name, shape=shape, dtype=dtype, fill_value=fill_value, **encoding
                )
                zarr_array.attrs.put(encoded_attrs)
                writer.add(v.data, zarr_array)

    def close(self):
        if self._consolidate_on_close:
            import zarr

            zarr.consolidate_metadata(self.ds.store)


def open_zarr(
    store,
    group=None,
    synchronizer=None,
    chunks="auto",
    decode_cf=True,
    mask_and_scale=True,
    decode_times=True,
    concat_characters=True,
    decode_coords=True,
    drop_variables=None,
    consolidated=False,
    overwrite_encoded_chunks=False,
    **kwargs,
):
    """Load and decode a dataset from a Zarr store.
    .. note:: Experimental
              The Zarr backend is new and experimental. Please report any
              unexpected behavior via github issues.
    The `store` object should be a valid store for a Zarr group. `store`
    variables must contain dimension metadata encoded in the
    `_ARRAY_DIMENSIONS` attribute.
    Parameters
    ----------
    store : MutableMapping or str
        A MutableMapping where a Zarr Group has been stored or a path to a
        directory in file system where a Zarr DirectoryStore has been stored.
    synchronizer : object, optional
        Array synchronizer provided to zarr
    group : str, obtional
        Group path. (a.k.a. `path` in zarr terminology.)
    chunks : int or dict or tuple or {None, 'auto'}, optional
        Chunk sizes along each dimension, e.g., ``5`` or
        ``{'x': 5, 'y': 5}``. If `chunks='auto'`, dask chunks are created
        based on the variable's zarr chunks. If `chunks=None`, zarr array
        data will lazily convert to numpy arrays upon access. This accepts
        all the chunk specifications as Dask does.
    overwrite_encoded_chunks: bool, optional
        Whether to drop the zarr chunks encoded for each variable when a
        dataset is loaded with specified chunk sizes (default: False)
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    drop_variables : string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    consolidated : bool, optional
        Whether to open the store using zarr's consolidated metadata
        capability. Only works for stores that have already been consolidated.
    Returns
    -------
    dataset : Dataset
        The newly created dataset.
    See Also
    --------
    open_dataset
    References
    ----------
    http://zarr.readthedocs.io/
    """
    if "auto_chunk" in kwargs:
        auto_chunk = kwargs.pop("auto_chunk")
        if auto_chunk:
            chunks = "auto"  # maintain backwards compatibility
        else:
            chunks = None

        warnings.warn(
            "auto_chunk is deprecated. Use chunks='auto' instead.",
            FutureWarning,
            stacklevel=2,
        )

    if kwargs:
        raise TypeError(
            "open_zarr() got unexpected keyword arguments " + ",".join(kwargs.keys())
        )

    if not isinstance(chunks, (int, dict)):
        if chunks != "auto" and chunks is not None:
            raise ValueError(
                "chunks must be an int, dict, 'auto', or None. "
                "Instead found %s. " % chunks
            )

    if not decode_cf:
        mask_and_scale = False
        decode_times = False
        concat_characters = False
        decode_coords = False

    def maybe_decode_store(store, lock=False):
        ds = conventions.decode_cf(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
        )

        # TODO: this is where we would apply caching

        return ds

    # Zarr supports a wide range of access modes, but for now xarray either
    # reads or writes from a store, never both. For open_zarr, we only read
    mode = "r"
    zarr_store = ZarrStore.open_group(
        store,
        mode=mode,
        synchronizer=synchronizer,
        group=group,
        consolidated=consolidated,
    )
    ds = maybe_decode_store(zarr_store)

    # auto chunking needs to be here and not in ZarrStore because variable
    # chunks do not survive decode_cf
    # return trivial case
    if not chunks:
        return ds

    # adapted from Dataset.Chunk()
    if isinstance(chunks, int):
        chunks = dict.fromkeys(ds.dims, chunks)

    if isinstance(chunks, tuple) and len(chunks) == len(ds.dims):
        chunks = dict(zip(ds.dims, chunks))

    def get_chunk(name, var, chunks):
        chunk_spec = dict(zip(var.dims, var.encoding.get("chunks")))

        # Coordinate labels aren't chunked
        if var.ndim == 1 and var.dims[0] == name:
            return chunk_spec

        if chunks == "auto":
            return chunk_spec

        for dim in var.dims:
            if dim in chunks:
                spec = chunks[dim]
                if isinstance(spec, int):
                    spec = (spec,)
                if isinstance(spec, (tuple, list)) and chunk_spec[dim]:
                    if any(s % chunk_spec[dim] for s in spec):
                        warnings.warn(
                            "Specified Dask chunks %r would "
                            "separate Zarr chunk shape %r for "
                            "dimension %r. This significantly "
                            "degrades performance. Consider "
                            "rechunking after loading instead."
                            % (chunks[dim], chunk_spec[dim], dim),
                            stacklevel=2,
                        )
                chunk_spec[dim] = chunks[dim]
        return chunk_spec

    def maybe_chunk(name, var, chunks):
        from dask.base import tokenize

        chunk_spec = get_chunk(name, var, chunks)

        if (var.ndim > 0) and (chunk_spec is not None):
            # does this cause any data to be read?
            token2 = tokenize(name, var._data)
            name2 = "zarr-%s" % token2
            var = var.chunk(chunk_spec, name=name2, lock=None)
            if overwrite_encoded_chunks and var.chunks is not None:
                var.encoding["chunks"] = tuple(x[0] for x in var.chunks)
            return var
        else:
            return var

    variables = {k: maybe_chunk(k, v, chunks) for k, v in ds.variables.items()}
    return ds._replace_vars_and_dims(variables)