#include "gorpho_kernels.cuh"
#include "gorpho_util.cuh"
#include "gorpho_matlab_utils.cuh"
#include "helper_math.cuh"
#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include "cudablockproc.cuh"
#include <vector>

static char const * const errId = "parallel:gpu:mex_gorpho_genDilateErode3d_block";

template <typename Ty>
void doGenDilateErode3d(mxArray const *vol, mxArray *res, const bool doDilate, const mxGPUArray *strel,
	const int3 blockSize, const bool doPrint)
{
	Ty *volPtr = (Ty *)mxGetData(vol);
	Ty *resPtr = (Ty *)mxGetData(res);
	Ty *d_strel = (Ty *)mxGPUGetDataReadOnly(strel);

	const int3 volsiz = getVolSize(vol);
	const int3 strelsiz = getVolSize(strel);
	const int3 borderSize = strelsiz / 2;

	const dim3 blockDim = dim3(32, 8, 4);

	const auto morphKernel = doDilate ?
		genDilateErode3dKernel<Ty, GORPHO_DILATE> : genDilateErode3dKernel<Ty, GORPHO_ERODE>;

	cbp::BlockIndexIterator blockIter(volsiz, blockSize, borderSize);
	int blki = 1;
	int numBlocks = blockIter.maxLinearIndex() + 1;

	// TODO: Allocate all device memory using MATLAB's API
	auto processBlock = [&](const cbp::BlockIndex& block, cudaStream_t stream,
		const std::vector<Ty *>& volVec, const std::vector<Ty *> resVec, void *buf)
	{
		const int3 siz = block.blockSizeBorder();
		const dim3 gridDim = gridBlocks(blockDim, volsiz);
		Ty *volBlk = volVec[0];
		Ty *resBlk = resVec[0];
		if (doPrint) {
			mexPrintf("Block %d / %d \n", blki, numBlocks);
			mexEvalString("drawnow;"); // Ensure above is printed immediately (for a small perf. penalty)
		}
		morphKernel << <gridDim, blockDim, 0, stream >> > (volBlk, d_strel, resBlk, siz, strelsiz);
		blki++;
	};
	cbp::CbpResult bpres = cbp::blockProc(processBlock, volPtr, resPtr, blockIter);
	auto cudares = cudaDeviceSynchronize();
	if (cudares != cudaSuccess) {
		mexErrMsgIdAndTxt(errId, "CUDA error: %s\n", cudaGetErrorString(cudares));
	}
	if (bpres != cbp::CBP_SUCCESS) {
		mexErrMsgIdAndTxt(errId, "block processing failed (code: %d)\n", bpres);
	}
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray const *prhs[])
{
	// TODO: Maybe do better input validation?
	if (nrhs < 4 || nrhs > 5) {
		mexErrMsgIdAndTxt(errId, "Must provide 4 or 5 inputs");
	}
	if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS && mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) {
		mexErrMsgIdAndTxt(errId, "1st input must be single or double, was %s", mxGetClassName(prhs[0]));
	}
	if (!mxIsGPUArray(prhs[1])) {
		mexErrMsgIdAndTxt(errId, "2nd must be a gpu array");
	}
	const mxGPUArray *strel = mxGPUCreateFromMxArray(prhs[1]);
	if (mxGetClassID(prhs[0]) != mxGPUGetClassID(strel)) {
		mexErrMsgIdAndTxt(errId, "1st and 2nd input must have same underlying class");
	}
	if (!mxIsNumeric(prhs[3]) || mxGetNumberOfElements(prhs[3]) != 3) {
		mexErrMsgIdAndTxt(errId, "4rd input must give width, height, and depth of block size");
	}
	if (nrhs > 4 && !mxIsScalar(prhs[4])) {
		mexErrMsgIdAndTxt(errId, "5th input must be a scalar if given");
	}

	// Extract inputs
	const mxArray *vol = prhs[0];
	const mxClassID class_id = mxGetClassID(vol);
	const int3 blockSize = mxArrayToInt3(prhs[3], "4th input ");
	const bool doDilate = mxGetScalar(prhs[2]) != 0.0;
	const bool doPrint = (nrhs > 4) ? static_cast<bool>(mxGetScalar(prhs[4])) : false;

	// Allocate output
	mxArray *res = mxCreateNumericArray(
		mxGetNumberOfDimensions(vol),
		mxGetDimensions(vol),
		class_id,
		mxREAL);

	// Do operation
	podTypeDispatch(class_id, doGenDilateErode3d, vol, res, doDilate, strel, blockSize, doPrint);

	plhs[0] = res;
}
