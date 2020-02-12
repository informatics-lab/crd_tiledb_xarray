from tiledb_xarray import TileDBStore, open_tiledb
import h5py
import xarray as xr

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
