from tiledb_xarray import TileDBStore, open_tiledb, hdf5_tiledb_writer, iris_writer
import h5py
import xarray as xr
import numpy as np
import shutil
import tiledb
import os

TB_DS = "./data/local.tiledb"
NC_DIR = "./data/monthly"
IN_NC = [os.path.join(NC_DIR, f) for f in sorted(os.listdir(NC_DIR))]
print(IN_NC)

try:
    shutil.rmtree(TB_DS)
except (IOError, OSError):
    pass


first_nc, other_ncs = IN_NC[0], IN_NC[1:]


# Build the first data set
builder = iris_writer.TileDBBuilder(TB_DS, first_nc)
builder.build()
builder.close()


# linear_arr_extend
# data = np.array([i*1000 for i in range(1000)])
# # hdf5_tiledb_writer.extend_dim(TB_DS + '/time', ds_to_insert=data, from_indexes=[0])

# # Grow dims


# ds = h5file['precipitation_flux']
# hdf5_tiledb_writer.insert_dataset(ds, TB_DS)

# with tiledb.open(TB_DS + '/time____', 'r') as A:
#     print(A.nonempty_domain())
# # store = TileDBStore.open_group(TB_DS)
# # print(store)

# # vars = store.get_variables()
# # print(vars)

# # t_ds = open_tiledb(TB_DS)
# # print(t_ds)

# # print("/\\ tiledb /\\")
# # print("--------------")
# # print("\\/ netcdf \\/")
# # l_ds = xr.open_dataset(IN_NC)
# # print(l_ds)

# # local_data = l_ds['precipitation_flux'][0:5, 0:10, 10:20].data
# # tiledb_data = t_ds['precipitation_flux'][0:5, 0:10, 10:20].data

# # assert np.all(local_data == tiledb_data)
# # print("*** Yay data match! ***")
