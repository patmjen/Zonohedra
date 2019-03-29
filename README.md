# Zonohedral Approximation of Spherical Structuring Element for Volumetric Morphology

Code for the SCIA19 paper on zonohedral approximations of spherical structuring elements.

## What is included

* MATLAB code for computing zonohedral approximations of spheres.
* CUDA code for doing morhological operations.
* MATLAB mex files which call out to the CUDA code (tested on Windows + Linux).
* A Python library which calls out to the CUDA code (tested on Windows + Linux).

**Precompiled libraries and mex-files** have been included for Windows and Linux. 
They can be downloaded with the source code under Releases.
Thus, you may not have to build anything. Instead just:

* Place the .mexa64 and .mexw64 in the matlab folder (or somewhere MATLAB can find them).
* Place the (lib)pygorpho files in the python folder (or somewhere Python can find them).
  Alternatively, the library will look for them at the path given by the `PYGORPHO_PATH` environment variable.

It that doesn't work, then...

## How to build

**Python**: A CMakeLists.txt has been provided. To build:

1. Navigate to the `python` folder
2. Make a new folder for building (e.g. called build)
3. Navigate to new folder
4. Call `cmake ..`.
5. Call `make` (or build Visual Studio project depending on the cmake generator).
6. Copy the shared library to the `python` folder (or set `PYGORPHO_PATH`)

**MATLAB**: A script for compilation has been provided: To build:

1. Open MATLAB and navigate to the `matlab` folder.
2. Run `compile_mex.m`.

**C/C++**: An API has not been made, but the python bindings should be a starting point.