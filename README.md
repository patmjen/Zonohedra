# Zonohedral Approximation of Spherical Structuring Element for Volumetric Morphology

Code for the SCIA19 paper on zonohedral approximations of spherical structuring elements. Paper can be found [here](http://orbit.dtu.dk/files/172879029/SCIA19_Zonohedra.pdf).

**UPDATE:** This code has now been published as a Python and MATLAB package.

**Python:**
* **PyPI**: [pypi.org/project/pygorpho](https://pypi.org/project/pygorpho/)
* **GitHub**: [github.com/patmjen/pygorpho](https://github.com/patmjen/pygorpho)
* **Documentation**: [pygorpho.readthedocs.io](https://pygorpho.readthedocs.io)

**MATLAB:**
* **GitHub**: [github.com/patmjen/mexgorpho](https://github.com/patmjen/mexgorpho)

**CUDA backend:**
* **GitHub**: [github.com/patmjen/gorpho](https://github.com/patmjen/gorpho)

The packages includes the GPU morphology code for dilation/erosion and ways to easily use the sperical structuing element approximation.

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

NOTE: The python code was built with CUDA 10.0 and the MATLAB code was built with CUDA 9.0.

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
