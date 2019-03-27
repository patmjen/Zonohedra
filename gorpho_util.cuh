#ifndef __GORPHO_UTIL_H__
#define __GORPHO_UTIL_H__

#include <limits>
#include <cuda_runtime.h>
#include <math_constants.h>
#include "helper_math.cuh"
#include "gorpho_consts.cuh"

#define returnOnCudaError(e) do { \
    if (e != cudaSuccess) { \
        return GORPHO_FAILURE; \
    } \
} while(false)

#define returnOnCublasError(e) do { \
    if (e != CUBLAS_STATUS_SUCCESS) { \
        return GORPHO_FAILURE; \
    } \
} while(false)

inline size_t gridLineBlocks(size_t nthr, const int siz)
{
    return siz/nthr + ((siz % nthr != 0) ? 1 : 0);
}

inline dim3 gridBlocks(const dim3 thrConfig, const int3 siz)
{
    return dim3(
        gridLineBlocks(thrConfig.x, siz.x),
        gridLineBlocks(thrConfig.y, siz.y),
        gridLineBlocks(thrConfig.z, siz.z)
    );
}

inline __host__ __device__ size_t numel(const int3& x)
{
    return ((size_t)x.x) * ((size_t)x.y) * ((size_t)x.z);
}

inline __host__ __device__ size_t getIdx(const int x, const int y, const int z, const int3 siz)
{
    return x + y * siz.x + z * siz.x * siz.y;
}
inline __host__ __device__ size_t getIdx(const int x, const int y, const int z, const dim3 siz)
{
    return x + y * siz.x + z * siz.x * siz.y;
}

inline __host__ __device__ size_t getIdx(const int3 pos, const int3 siz)
{
    return getIdx(pos.x, pos.y, pos.z, siz);
}
inline __host__ __device__ size_t getIdx(const int3 pos, const dim3 siz)
{
    return getIdx(pos.x, pos.y, pos.z, siz);
}
inline __host__ __device__ size_t getIdx(const dim3 pos, const dim3 siz)
{
    return getIdx((const int)pos.x, (const int)pos.y, (const int)pos.z, siz);
}

inline __device__ int3 getGlobalPos_3D()
{
    return make_int3(
        threadIdx.x + blockDim.x*blockIdx.x,
        threadIdx.y + blockDim.y*blockIdx.y,
        threadIdx.z + blockDim.z*blockIdx.z
    );
}

template <typename T>
__global__ void copyData(T *dst, const T *src, const size_t n)
{
    const size_t i = getIdx(getGlobalPos_3D(), blockDim*gridDim);
    if (i < n) {
        dst[i] = src[i];
    }
}
template __global__ void copyData<float>(float *dst, const float *src, size_t n);
template __global__ void copyData<double>(double *dst, const double *src, size_t n);

template <typename T>
inline __device__ T infOrMax()
{
    // TODO: Maybe do specialization for long double?
    return std::numeric_limits<T>::max();
}
template<>
inline __device__ float infOrMax<float>()
{
    return CUDART_INF_F;
}
template<>
inline __device__ double infOrMax<double>()
{
    return CUDART_INF;
}

template <typename T>
inline __device__ T minusInfOrMin()
{
    // TODO: Maybe do specialization for long double?
    return std::numeric_limits<T>::min();
}
template<>
inline __device__ float minusInfOrMin<float>()
{
    return -CUDART_INF_F;
}
template<>
inline __device__ double minusInfOrMin<double>()
{
    return -CUDART_INF;
}

#endif /* __GORPHO_UTIL_H__ */