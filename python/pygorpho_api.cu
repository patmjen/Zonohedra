#include <iostream>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

#include "cudablockproc.cuh"

#include "gorpho_kernels.cuh"
#include "gorpho_util.cuh"

#if defined _WIN32
    #define PYGORPHO_API __declspec(dllexport)
#else
    #define PYGORPHO_API
#endif

// Copied from: numpy/core/include/numpy/ndarraytypes.h
enum NPY_TYPES : int {
	NPY_BOOL=0,
    NPY_BYTE, NPY_UBYTE,
    NPY_SHORT, NPY_USHORT,
    NPY_INT, NPY_UINT,
    NPY_LONG, NPY_ULONG,
    NPY_LONGLONG, NPY_ULONGLONG,
    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
};

#define podTypeDispatch(type, func, ...) do { \
	switch (type) { \
	case NPY_BOOL: \
		func<bool>(__VA_ARGS__); \
		break; \
	case NPY_BYTE: \
		func<std::int8_t>(__VA_ARGS__); \
		break; \
	case NPY_UBYTE: \
		func<std::uint8_t>(__VA_ARGS__); \
		break; \
	case NPY_SHORT: \
		func<short>(__VA_ARGS__); \
		break; \
	case NPY_USHORT: \
		func<unsigned short>(__VA_ARGS__); \
		break; \
	case NPY_INT: \
		func<int>(__VA_ARGS__); \
		break; \
	case NPY_UINT: \
		func<unsigned int>(__VA_ARGS__); \
		break; \
	case NPY_LONG: \
		func<long>(__VA_ARGS__); \
		break; \
	case NPY_ULONG: \
		func<unsigned long>(__VA_ARGS__); \
		break; \
	case NPY_LONGLONG: \
		func<long long>(__VA_ARGS__); \
		break; \
	case NPY_ULONGLONG: \
		func<unsigned long long>(__VA_ARGS__); \
		break; \
	case NPY_FLOAT: \
		func<float>(__VA_ARGS__); \
		break; \
	case NPY_DOUBLE: \
		func<double>(__VA_ARGS__); \
		break; \
	} \
} while(false)

void ensureCudaSuccess(cudaError_t res)
{
	if (res != cudaSuccess) {
        std::string errMsg("CUDA error: ");
		errMsg += cudaGetErrorString(res);
		throw std::runtime_error(errMsg);
    }
}

template <class Ty>
void doGenDilateErode3d(void *__restrict__ vol, void *__restrict__ res, bool doDilate,
	const void *__restrict__ strel, const int3 blockSize, const bool doPrint, int3 volsiz, int3 strelsiz)
{
    Ty *volPtr = static_cast<Ty *>(vol);
    Ty *resPtr = static_cast<Ty *>(res);
    const Ty *strelPtr = static_cast<const Ty *>(strel);

	// Allocate device memory for strel and copy it to device
	Ty *d_strelPtr = nullptr;
	ensureCudaSuccess(cudaMalloc(&d_strelPtr, numel(strelsiz) * sizeof(Ty)));
	auto cudaDeleter = [](auto ptr) { cudaFree(ptr); };
	std::unique_ptr<Ty, decltype(cudaDeleter)> d_strelSPtr(d_strelPtr, cudaDeleter); // Put in smart pointer to ensure free
	ensureCudaSuccess(cudaMemcpy(d_strelPtr, strelPtr, numel(strelsiz) * sizeof(Ty), cudaMemcpyHostToDevice));

    const int3 borderSize = strelsiz / 2;
    const dim3 blockDim = dim3(32,8,4); // These usually give good performance

	// Choose kernel
    const auto morphKernel = doDilate ?
        genDilateErode3dKernel<Ty, GORPHO_DILATE> : genDilateErode3dKernel<Ty, GORPHO_ERODE>;

	// Make function to process each block
    cbp::BlockIndexIterator blockIter(volsiz, blockSize, borderSize);
    int blki = 1;
    int numBlocks = blockIter.maxLinearIndex() + 1;
    auto processBlock = [&](auto block, auto stream, auto volVec, auto resVec, void *buf)
    {
        const int3 siz = block.blockSizeBorder();
        const dim3 gridDim = gridBlocks(blockDim, siz);
        Ty *volBlk = volVec[0];
        Ty *resBlk = resVec[0];
        if (doPrint) {
            std::printf("Block %d / %d \n", blki, numBlocks);
            std::cout.flush(); // Ensure above is printed immediately (for a small perf. penalty)
        }
        morphKernel<<<gridDim, blockDim, 0, stream>>>(volBlk, d_strelPtr, resBlk, siz, strelsiz);
        blki++;
    };
    cbp::CbpResult bpres = cbp::blockProc(processBlock, volPtr, resPtr, blockIter);
    ensureCudaSuccess(cudaDeviceSynchronize());
    if (bpres != cbp::CBP_SUCCESS) {
        std::string errMsg("block processing failed (code: ");
		errMsg += bpres;
		errMsg += ")";
		throw std::runtime_error(errMsg);
    }
}

template <class Ty>
void doFlatLinearDilateErode3d(void *__restrict__ vol, void *__restrict__ res, bool doDilate,
	const int *strelSizes, const std::vector<int3>& steps, const int3 volsiz, const int3 blockSize, bool doPrint)
{
    Ty *volPtr = static_cast<Ty *>(vol);
    Ty *resPtr = static_cast<Ty *>(res);
	const int numStrels = steps.size();

	// Compute size of block borders and buffer to hold the R and S arrays
	int bufsiz = 0;
    int3 hasDirection = make_int3(0);
    int3 stepSumPos = make_int3(0);
    int3 stepSumNeg = make_int3(0);
	for (int i = 0; i < numStrels; i++) {
        hasDirection.x |= (steps[i].x != 0);
        hasDirection.y |= (steps[i].y != 0);
        hasDirection.z |= (steps[i].z != 0);

        if (steps[i].x > 0) {
            stepSumPos.x += steps[i].x * strelSizes[i];
        } else {
            stepSumNeg.x -= steps[i].x * strelSizes[i];
        }
        if (steps[i].y > 0) {
            stepSumPos.y += steps[i].y * strelSizes[i];
        } else {
            stepSumNeg.y -= steps[i].y * strelSizes[i];
        }
        if (steps[i].z > 0) {
            stepSumPos.z += steps[i].z * strelSizes[i];
        } else {
            stepSumNeg.z -= steps[i].z * strelSizes[i];
        }
    }
	int3 reqBufSiz = max(stepSumNeg, stepSumPos);
	int3 borderSize = blockSize >= volsiz ? make_int3(0) : reqBufSiz; // Don't use a border unless needed
	if (hasDirection.x != 0) {
        bufsiz = (blockSize.y + 2*borderSize.y)*(blockSize.z + 2*borderSize.z)*reqBufSiz.x;
    }
    if (hasDirection.y != 0) {
        bufsiz = max(bufsiz, (blockSize.x + 2*borderSize.x)*(blockSize.z + 2*borderSize.z)*reqBufSiz.y);
    }
    if (hasDirection.z != 0) {
        bufsiz = max(bufsiz, (blockSize.x + 2*borderSize.x)*(blockSize.y + 2*borderSize.y)*reqBufSiz.z);
    }

	// Choose kernel
    const auto morphKernel = doDilate ?
        flatLinearDilateErode3dKernel<Ty, GORPHO_DILATE> : flatLinearDilateErode3dKernel<Ty, GORPHO_ERODE>;

	// Make function to process each block
	cbp::BlockIndexIterator blockIter(volsiz, blockSize, borderSize);
    int blki = 1;
    const dim3 blockDim = dim3(2,16,16);
    int numBlocks = blockIter.maxLinearIndex() + 1;
	auto processBlock = [&](auto block, auto stream, auto volVec, auto resVec, void *buf)
    {
        const int3 siz = block.blockSizeBorder();

        Ty *volBlk = volVec[0];
        Ty *resBlk = resVec[0];
        Ty *rBlk = (Ty *)buf;
        Ty *sBlk = rBlk + bufsiz;
        Ty *tmpBlk = sBlk + bufsiz; // Only needed if numStrels > 1

        Ty *crntInBlk = volBlk;
        Ty *crntOutBlk = (numStrels % 2 == 0) ? tmpBlk : resBlk; // Make sure we end on resBlk
        if (doPrint) {
            std::printf("Block %d / %d \n", blki, numBlocks);
            std::cout.flush(); // Ensure above is printed immediately (for a small perf. penalty)
        }
        for (int i = 0; i < numStrels; i++) {
			const int3 step = steps[i];
			const int strelsiz = strelSizes[i];
            if (step.x != 0) {
                const dim3 gridDim = dim3(
                    1,
                    gridLineBlocks(blockDim.y, siz.y),
                    gridLineBlocks(blockDim.z, siz.z)
                );
                morphKernel<<<gridDim,blockDim,0,stream>>>(crntInBlk, strelsiz, GORPHO_DIR_X, step,
                    rBlk, sBlk, crntOutBlk, siz);
            }
            if (step.y != 0) {
                const dim3 gridDim = dim3(
                    1,
                    gridLineBlocks(blockDim.x, siz.x),
                    gridLineBlocks(blockDim.z, siz.z)
                );
                morphKernel<<<gridDim,blockDim,0,stream>>>(crntInBlk, strelsiz, GORPHO_DIR_Y, step,
                    rBlk, sBlk, crntOutBlk, siz);
            }
            if (step.z != 0) {
                const dim3 gridDim = dim3(
                    1,
                    gridLineBlocks(blockDim.x, siz.x),
                    gridLineBlocks(blockDim.y, siz.y)
                );
                morphKernel<<<gridDim,blockDim,0,stream>>>(crntInBlk, strelsiz, GORPHO_DIR_Z, step,
                    rBlk, sBlk, crntOutBlk, siz);
            }
            crntInBlk = crntOutBlk;
            crntOutBlk = (crntOutBlk == resBlk) ? tmpBlk : resBlk;
        }
        blki++;
    };

	size_t tmpSize = 2*bufsiz*sizeof(Ty);
	if (numStrels > 1) {
        // If there is more than one strel we need an extra output volume to put intermediary results into
        tmpSize += numel(blockSize + 2*borderSize)*sizeof(Ty);
    }
	cbp::CbpResult bpres = cbp::blockProc(processBlock, volPtr, resPtr, blockIter, tmpSize);
    ensureCudaSuccess(cudaDeviceSynchronize());
    if (bpres != cbp::CBP_SUCCESS) {
        std::string errMsg("block processing failed (code: ");
		errMsg += bpres;
		errMsg += ")";
		throw std::runtime_error(errMsg);
    }
}

#ifdef __cplusplus
extern "C" {
#endif

PYGORPHO_API int genDilateErode(void *__restrict__ vol, void *__restrict__ strel,
	void *__restrict__ res, int doDilate, int volsizx, int volsizy, int volsizz,
    int strelsizx, int strelsizy, int strelsizz, int blocksizx, int blocksizy, int blocksizz, int type, int doPrint)
{
	int3 volsiz = make_int3(volsizx, volsizy, volsizz);
	int3 strelsiz = make_int3(strelsizx, strelsizy, strelsizz);
	int3 blocksiz = make_int3(blocksizx, blocksizy, blocksizz);

	try {
		podTypeDispatch(type, doGenDilateErode3d, vol, res, doDilate != 0, strel, blocksiz, doPrint != 0,
			volsiz, strelsiz);
		return 0;
	} catch (const std::runtime_error& err) {
		std::printf("%s\n", err.what());
		return -1;
	}
}

PYGORPHO_API int flatLinearDilateErode(void *__restrict__ vol, const int *strelSizes,
	const int *stepx, const int *stepy, const int *stepz, int nstrel, int doDilate, void *__restrict__ res,
	int volsizx, int volsizy, int volsizz, int blocksizx, int blocksizy, int blocksizz, int type, int doPrint)
{
	std::vector<int3> steps;
	steps.reserve(nstrel);
	for (int i = 0; i < nstrel; ++i) {
		steps.push_back(make_int3(stepx[i], stepy[i], stepz[i]));
	}
	int3 volsiz = make_int3(volsizx, volsizy, volsizz);
	int3 blocksiz = make_int3(blocksizx, blocksizy, blocksizz);

	try {
		podTypeDispatch(type, doFlatLinearDilateErode3d, vol, res, doDilate != 0, strelSizes,
			steps, volsiz, blocksiz, doPrint != 0);
		return 0;
	} catch (const std::runtime_error& err) {
		std::printf("%s\n", err.what());
		return -1;
	}
}

#ifdef __cplusplus
}
#endif
