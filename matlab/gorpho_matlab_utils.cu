#include "gorpho_matlab_utils.cuh"

void assertPOD(mxGPUArray const *arr, char const * const errId)
{
    switch (mxGPUGetClassID(arr)) {
        case mxDOUBLE_CLASS:
        case mxSINGLE_CLASS:
        case mxINT8_CLASS:
        case mxUINT8_CLASS:
        case mxINT16_CLASS:
        case mxUINT16_CLASS:
        case mxINT32_CLASS:
        case mxUINT32_CLASS:
        case mxINT64_CLASS:
        case mxUINT64_CLASS:
            // Do nothing
            break;
        default:
            mexErrMsgIdAndTxt(errId, "Inputs must be of class double, single, int8, uint8, int16, uint16, int32, uint32, int64. uint64");
            break;
    }
}

int3 getVolSize(const mxGPUArray *vol)
{
    int3 out = make_int3(1);
    mwSize volndim = mxGPUGetNumberOfDimensions(vol);
    const mwSize *volsiz = mxGPUGetDimensions(vol);
    if (volndim >= 3) {
        out.z = volsiz[2];
    }
    if (volndim >= 2) {
        out.y = volsiz[1];
    }
    if (volndim >= 1) {
        out.x = volsiz[0];
    }
    return out;
}


int3 getVolSize(const mxArray *vol)
{
    int3 out = make_int3(1);
    mwSize volndim = mxGetNumberOfDimensions(vol);
    const mwSize *volsiz = mxGetDimensions(vol);
    if (volndim >= 3) {
        out.z = volsiz[2];
    }
    if (volndim >= 2) {
        out.y = volsiz[1];
    }
    if (volndim >= 1) {
        out.x = volsiz[0];
    }
    return out;
}

int3 mxArrayToInt3(const mxArray *arr, std::string msgPrefix) {
    mxAssert(mxIsNumeric(arr) && !mxIsComplex(arr), msgPrefix + "must be real and double");
    mxAssert(mxGetNumberOfElements(arr), msgPrefix + "must have 3 elements");
    const void *data = mxGetData(arr);
    return make_int3(
        void_ptr_cast_and_deref<int>(data, mxGetClassID(arr), 0),
        void_ptr_cast_and_deref<int>(data, mxGetClassID(arr), 1),
        void_ptr_cast_and_deref<int>(data, mxGetClassID(arr), 2)
    );
}