#include "gorpho_kernels.cuh"
#include "gorpho_util.cuh"
#include "gorpho_matlab_utils.cuh"
#include "helper_math.cuh"
#include "mex.h"
#include "matrix.h"
#include "gpu/mxGPUArray.h"
#include "cudablockproc.cuh"
#include <vector>

static char const * const errId = "parallel:gpu:mex_gorpho_flatLinearDilateErode3d_block";

template <typename Ty>
void doFlatLinearDilateErode3d(mxArray const *vol, mxArray *res, const bool doDilate,
	const mxArray *strelsizes, const mxArray *steps, const int3 blockSize, const bool doPrint)
{
	Ty *volPtr = (Ty *)mxGetData(vol);
	Ty *resPtr = (Ty *)mxGetData(res);

	const int numStrels = mxGetNumberOfElements(strelsizes);
	const int3 volsiz = getVolSize(vol);
	const dim3 blockDim = dim3(2, 16, 16);

	// Compute buffer size based on max strel size and steps
	int bufsiz = 0;
	int3 hasDirection = make_int3(0);
	int3 stepSumPos = make_int3(0);
	int3 stepSumNeg = make_int3(0);
	for (int i = 0; i < numStrels; i++) {
		const int crntStrelSiz = get_nth_element<int>(strelsizes, i);
		const int3 step = make_int3(
			get_nth_element<int>(steps, i),
			get_nth_element<int>(steps, i + numStrels),
			get_nth_element<int>(steps, i + 2 * numStrels)
		);
		hasDirection.x |= step.x != 0;
		hasDirection.y |= step.y != 0;
		hasDirection.z |= step.z != 0;

		if (step.x > 0) {
			stepSumPos.x += step.x * crntStrelSiz;
		} else {
			stepSumNeg.x -= step.x * crntStrelSiz;
		}
		if (step.y > 0) {
			stepSumPos.y += step.y * crntStrelSiz;
		} else {
			stepSumNeg.y -= step.y * crntStrelSiz;
		}
		if (step.z > 0) {
			stepSumPos.z += step.z * crntStrelSiz;
		} else {
			stepSumNeg.z -= step.z * crntStrelSiz;
		}
	}
	int3 reqBufSiz = max(stepSumNeg, stepSumPos);
	int3 borderSize = blockSize >= volsiz ? make_int3(0) : reqBufSiz; // Don't use a border unless needed
	if (hasDirection.x != 0) {
		bufsiz = (blockSize.y + 2 * borderSize.y)*(blockSize.z + 2 * borderSize.z)*reqBufSiz.x;
	}
	if (hasDirection.y != 0) {
		bufsiz = max(bufsiz, (blockSize.x + 2 * borderSize.x)*(blockSize.z + 2 * borderSize.z)*reqBufSiz.y);
	}
	if (hasDirection.z != 0) {
		bufsiz = max(bufsiz, (blockSize.x + 2 * borderSize.x)*(blockSize.y + 2 * borderSize.y)*reqBufSiz.z);
	}

	const auto morphKernel = doDilate ?
		flatLinearDilateErode3dKernel<Ty, GORPHO_DILATE> : flatLinearDilateErode3dKernel<Ty, GORPHO_ERODE>;

	cbp::BlockIndexIterator blockIter(volsiz, blockSize, borderSize);
	int blki = 1;
	int numBlocks = blockIter.maxLinearIndex() + 1;

	// TODO: Allocate all device memory using MATLAB's API
	auto processBlock = [&](const cbp::BlockIndex& block, cudaStream_t stream,
		const std::vector<Ty *>& volVec, const std::vector<Ty *> resVec, void *buf)
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
			mexPrintf("Block %d / %d \n", blki, numBlocks);
			mexEvalString("drawnow;"); // Ensure above is printed immediately (for a small perf. penalty)
		}
		for (int i = 0; i < numStrels; i++) {
			// TODO: Figure out a better way to get these numbers
			const size_t strelsiz = get_nth_element<size_t>(strelsizes, i);
			const int3 step = make_int3(
				get_nth_element<int>(steps, i),
				get_nth_element<int>(steps, i + numStrels),
				get_nth_element<int>(steps, i + 2 * numStrels)
			);
			if (step.x != 0) {
				const dim3 gridDim = dim3(
					1,
					gridLineBlocks(blockDim.y, siz.y),
					gridLineBlocks(blockDim.z, siz.z)
				);
				morphKernel << <gridDim, blockDim, 0, stream >> > (crntInBlk, strelsiz, GORPHO_DIR_X, step,
					rBlk, sBlk, crntOutBlk, siz);
			}
			if (step.y != 0) {
				const dim3 gridDim = dim3(
					1,
					gridLineBlocks(blockDim.x, siz.x),
					gridLineBlocks(blockDim.z, siz.z)
				);
				morphKernel << <gridDim, blockDim, 0, stream >> > (crntInBlk, strelsiz, GORPHO_DIR_Y, step,
					rBlk, sBlk, crntOutBlk, siz);
			}
			if (step.z != 0) {
				const dim3 gridDim = dim3(
					1,
					gridLineBlocks(blockDim.x, siz.x),
					gridLineBlocks(blockDim.y, siz.y)
				);
				morphKernel << <gridDim, blockDim, 0, stream >> > (crntInBlk, strelsiz, GORPHO_DIR_Z, step,
					rBlk, sBlk, crntOutBlk, siz);
			}
			crntInBlk = crntOutBlk;
			crntOutBlk = (crntOutBlk == resBlk) ? tmpBlk : resBlk;
		}
		blki++;
	};
	size_t tmpSize = 2 * bufsiz * sizeof(Ty);
	if (numStrels > 1) {
		// If there is more than one strel we need an extra output volume to put intermediary results into
		tmpSize += numel(blockSize + 2 * borderSize) * sizeof(Ty);
	}
	cbp::CbpResult bpres = cbp::blockProc(processBlock, volPtr, resPtr, blockIter, tmpSize);
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
	if (nrhs < 4 || nrhs > 6) {
		mexErrMsgIdAndTxt(errId, "Must provide 4 to 6 inputs");
	}
	if (!mxIsNumeric(prhs[0])) {
		mexErrMsgIdAndTxt(errId, "1st input must be numeric");
	}
	if (!mxIsScalar(prhs[1])) {
		mexErrMsgIdAndTxt(errId, "2nd must be a scalar");
	}
	if (!mxIsNumeric(prhs[2])) {
		mexErrMsgIdAndTxt(errId, "3rd input must be numeric");
	}
	if (!mxIsNumeric(prhs[3]) || mxGetN(prhs[3]) != 3) {
		mexErrMsgIdAndTxt(errId, "4th input must have x-, y-, and z-step");
	}
	if (mxGetNumberOfElements(prhs[2]) != mxGetM(prhs[3])) {
		mexErrMsgIdAndTxt(errId, "Number of elements in 3rd input must match number of rows in 4th");
	}
	if (!mxIsNumeric(prhs[4]) || mxGetNumberOfElements(prhs[4]) != 3) {
		mexErrMsgIdAndTxt(errId, "5th input must give width, height, and depth of block size");
	}
	if (nrhs > 5 && !mxIsScalar(prhs[5])) {
		mexErrMsgIdAndTxt(errId, "6th input must be a scalar if given");
	}

	// Extract inputs
	const mxArray *vol = prhs[0];
	const int3 volsiz = getVolSize(vol);
	const mxClassID class_id = mxGetClassID(vol);
	const mxArray *strelsizes = prhs[2];
	const mxArray *steps = prhs[3];
	const int3 blockSize = mxArrayToInt3(prhs[4], "5th input ");
	const bool doDilate = mxGetScalar(prhs[1]) != 0.0;
	const bool doPrint = (nrhs > 5) ? static_cast<bool>(mxGetScalar(prhs[5])) : false;

	// Allocate output
	mxArray *res = mxCreateNumericArray(
		mxGetNumberOfDimensions(vol),
		mxGetDimensions(vol),
		class_id,
		mxREAL);

	// Do operation
	podTypeDispatch(class_id, doFlatLinearDilateErode3d,
		vol, res, doDilate, strelsizes, steps, blockSize, doPrint);

	plhs[0] = res;
}
