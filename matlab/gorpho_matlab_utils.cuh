#ifndef __GORPHO_MATLAB_UTILS_H__
#define __GORPHO_MATLAB_UTILS_H__

#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include "helper_math.cuh"

void assertPOD(mxGPUArray const *arr, char const * const errId);

int3 getVolSize(const mxGPUArray *vol);
int3 getVolSize(const mxArray *vol);

int3 mxArrayToInt3(const mxArray *arr, std::string msgPrefix = "Input ");

template <typename TyOut, typename TyPtr>
inline TyOut void_ptr_cast_and_deref(const void *dat, const size_t offset=0)
{
    return static_cast<TyOut>(static_cast<const TyPtr*>(dat)[offset]);
}

template <typename TyOut>
TyOut void_ptr_cast_and_deref(const void *dat, mxClassID cid, const size_t offset=0)
{
    switch (cid) {
        case mxDOUBLE_CLASS:
            return void_ptr_cast_and_deref<TyOut, double>(dat, offset);
        case mxSINGLE_CLASS:
            return void_ptr_cast_and_deref<TyOut, float>(dat, offset);
        case mxINT8_CLASS:
            return void_ptr_cast_and_deref<TyOut, int8_t>(dat, offset);
        case mxUINT8_CLASS:
            return void_ptr_cast_and_deref<TyOut, uint8_t>(dat, offset);
        case mxINT16_CLASS:
            return void_ptr_cast_and_deref<TyOut, int16_t>(dat, offset);
        case mxUINT16_CLASS:
            return void_ptr_cast_and_deref<TyOut, uint16_t>(dat, offset);
        case mxINT32_CLASS:
            return void_ptr_cast_and_deref<TyOut, int32_t>(dat, offset);
        case mxUINT32_CLASS:
            return void_ptr_cast_and_deref<TyOut, uint32_t>(dat, offset);
        case mxINT64_CLASS:
            return void_ptr_cast_and_deref<TyOut, int64_t>(dat, offset);
        case mxUINT64_CLASS:
            return void_ptr_cast_and_deref<TyOut, uint64_t>(dat, offset);
        default:
            mexErrMsgIdAndTxt("gorpho:internal", "Inputs must be of class double, single, int8, uint8, int16, uint16, int32, uint32, int64. uint64");
            // The below will *never* run as mexErrMsgIdAndTxt aborts execution
            return TyOut();
    }
}

template <typename Ty>
Ty get_nth_element(const mxArray *arr, const size_t n)
{
    return void_ptr_cast_and_deref<Ty>(mxGetData(arr), mxGetClassID(arr), n);
}

template <typename Ty>
void copy_and_cast_mxarray(Ty *dst, const mxArray *src)
{
    const mxClassID cid = mxGetClassID(src);
    const size_t num_elem = mxGetNumberOfElements(src);
    const void *dat = mxGetData(src);
    for (size_t i = 0; i < num_elem; i++) {
        dst[i] = void_ptr_cast_and_deref<Ty>(dat, cid, i);
    }
}

// TODO: If possible, find a way to make this into a proper template function
#define podTypeDispatch(cid, func, ...) do { \
    switch (cid) { \
    case mxDOUBLE_CLASS: \
        func<double>(__VA_ARGS__); \
        break; \
    case mxSINGLE_CLASS: \
        func<float>(__VA_ARGS__); \
        break; \
    case mxINT8_CLASS: \
        func<int8_t>(__VA_ARGS__); \
        break; \
    case mxUINT8_CLASS: \
        func<uint8_t>(__VA_ARGS__); \
        break; \
    case mxINT16_CLASS: \
        func<int16_t>(__VA_ARGS__); \
        break; \
    case mxUINT16_CLASS: \
        func<uint16_t>(__VA_ARGS__); \
        break; \
    case mxINT32_CLASS: \
        func<int32_t>(__VA_ARGS__); \
        break; \
    case mxUINT32_CLASS: \
        func<uint32_t>(__VA_ARGS__); \
        break; \
    case mxINT64_CLASS: \
        func<int64_t>(__VA_ARGS__); \
        break; \
    case mxUINT64_CLASS: \
        func<uint64_t>(__VA_ARGS__); \
        break; \
    default: \
        mexErrMsgIdAndTxt("gorpho:internal", "Inputs must be of class double, single, int8, uint8, int16, uint16, int32, uint32, int64. uint64"); \
        break; \
    } \
} while (false)

#endif /* __GORPHO_MATLAB_UTILS_H__ */