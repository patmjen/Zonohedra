import ctypes
import platform
import os
import numpy as np
import numpy.ctypeslib as ctl

# Load the shared library
if platform.system() == 'Windows':
	__GORPHO_LIB__ = ctl.load_library('pygorpho.dll', os.getenv('PYGORPHO_PATH','./'))
else:
	__GORPHO_LIB__ = ctl.load_library('libpygorpho.so', os.getenv('PYGORPHO_PATH','./'))

# Make thin wrappers for internal use

_gen_dilate_erode_impl = __GORPHO_LIB__.genDilateErode
_gen_dilate_erode_impl.argtypes = [
	ctypes.c_void_p, # vol
	ctypes.c_void_p, # strel
	ctypes.c_void_p, # res
	ctypes.c_int,    # doDilate
	ctypes.c_int,    # volsizx
	ctypes.c_int,    # volsizy
	ctypes.c_int,    # volsizz
	ctypes.c_int,    # strelsizx
	ctypes.c_int,    # strelsizy
	ctypes.c_int,    # strelsizz
	ctypes.c_int,    # blocksizx
	ctypes.c_int,    # blocksizy
	ctypes.c_int,    # blocksizz
	ctypes.c_int,    # type
	ctypes.c_int     # doPrint
]

_flat_linear_dilate_erode_impl = __GORPHO_LIB__.flatLinearDilateErode
_flat_linear_dilate_erode_impl.argtypes = [
	ctypes.c_void_p,               # vol
	ctl.ndpointer(dtype=np.int32), # strelSizes
	ctl.ndpointer(dtype=np.int32), # stepx
	ctl.ndpointer(dtype=np.int32), # stepy
	ctl.ndpointer(dtype=np.int32), # stepz
	ctypes.c_int,                  # nstrel
	ctypes.c_int,                  # doDilate
	ctypes.c_void_p,               # res
	ctypes.c_int,                  # volsizx
	ctypes.c_int,                  # volsizy
	ctypes.c_int,                  # volsizz
	ctypes.c_int,                  # blocksizx
	ctypes.c_int,                  # blocksizy
	ctypes.c_int,                  # blocksizz
	ctypes.c_int,                  # type
	ctypes.c_int,                  # doPrint
]

# Make friendly wrappers

def gen_dilate_erode(vol, strel, doDilate, blockSize=[256,256,256],
	                 doPrint=False):
	# Recast inputs to correct datatype
	vol = np.atleast_3d(np.asarray(vol))
	strel = np.asarray(strel, dtype=vol.dtype)
	assert vol.dtype == strel.dtype

	# Prepare output volume
	volsiz = vol.shape
	res = np.empty(volsiz, dtype=vol.dtype)

	code = _gen_dilate_erode_impl(vol.ctypes.data, strel.ctypes.data,
		res.ctypes.data, doDilate, 
		volsiz[2], volsiz[1], volsiz[0],
		strel.shape[2], strel.shape[1], strel.shape[0], 
		blockSize[2], blockSize[1], blockSize[0], 
		vol.dtype.num, doPrint)
	if code != 0:
		raise Exception()
	return res

def gen_dilate(vol, strel, blockSize=[256,256,256], doPrint=False):
	return gen_dilate_erode(vol, strel, True, blockSize, doPrint)

def gen_erode(vol, strel, blockSize=[256,256,256], doPrint=False):
	return gen_dilate_erode(vol, strel, False, blockSize, doPrint)

def flat_linear_dilate_erode(vol, strelSizes, strelSteps, doDilate,
							 blockSize=(256,256,256), doPrint=False):
	# Recast inputs to correct datatype
	vol = np.atleast_3d(np.asarray(vol))
	stepsInt = np.atleast_2d(np.asarray(strelSteps, dtype=np.int32))
	strelSizesInt = np.atleast_1d(np.asarray(strelSizes, dtype=np.int32))

	# Validate input
	assert stepsInt.shape[1] == 3
	assert len(strelSizesInt) == stepsInt.shape[0]

	# Prepare output volume
	volsiz = vol.shape
	res = np.empty(volsiz, dtype=vol.dtype)

	# Need to copy so we can use regular indexing in C++
	stepsIntX = stepsInt[:,2].copy()
	stepsIntY = stepsInt[:,1].copy()
	stepsIntZ = stepsInt[:,0].copy()

	code = _flat_linear_dilate_erode_impl(vol.ctypes.data, 
		strelSizesInt, stepsIntX, stepsIntY, stepsIntZ, 
		len(strelSizesInt), doDilate, res.ctypes.data, 
		volsiz[2], volsiz[1], volsiz[0], 
		blockSize[2], blockSize[1], blockSize[0], 
		vol.dtype.num, doPrint)
	if code != 0:
		raise Exception()
	return res

def flat_linear_dilate(vol, strelSizes, strelSteps, blockSize=[256,256,256],
                       doPrint=False):
    return flat_linear_dilate_erode(vol, strelSizes, strelSteps, True,
		blockSize, doPrint)
		
def flat_linear_erode(vol, strelSizes, strelSteps, blockSize=[256,256,256],
                       doPrint=False):
    return flat_linear_dilate_erode(vol, strelSizes, strelSteps, False,
		blockSize, doPrint)