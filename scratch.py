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

t_ds = open_tiledb("./myds")
print(t_ds)

print("/\\ tiledb /\\")

print("\\/ netcdf \\/")
l_ds = xr.open_dataset('local.nc')
print(l_ds)

print('end')
