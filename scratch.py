from tiledb_xarray import TileDBStore, open_tiledb
import h5py
import xarray as xr
import numpy as np

TB_DS = "./data/local.tiledb"

# store = TileDBStore.open_group(TB_DS)
# print(store)

# vars = store.get_variables()
# print(vars)

t_ds = open_tiledb(TB_DS)
print(t_ds)

print("/\\ tiledb /\\")
print("--------------")
print("\\/ netcdf \\/")
l_ds = xr.open_dataset('./data/local.nc')
print(l_ds)

local_data = l_ds['precipitation_flux'][0:5, 0:10, 10:20].data
tiledb_data = t_ds['precipitation_flux'][0:5, 0:10, 10:20].data

assert np.all(local_data == tiledb_data)
print("*** Yay data match! ***")
