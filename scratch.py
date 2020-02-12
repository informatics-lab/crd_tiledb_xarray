from tiledb_xarray import TileDBStore, open_tiledb
import h5py
import xarray as xr

# ds = xr.open_dataset('local.nc')
# ds.to_zarr('local.zarr')

# ds = xr.open_zarr('local.zarr')
# print(ds)


store = TileDBStore.open_group("./myds")
print(store)

# vars = store.get_variables()
# print(vars)

ds = open_tiledb("./myds")
print(ds)

print("/\\ tiledb /\\")

print("\\/ netcdf \\/")
ds = xr.open_dataset('local.nc')
print(ds)
