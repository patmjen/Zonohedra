#ifndef __GORPHO_KERNELS_H__
#define __GORPHO_KERNELS_H__

#include "gorpho_consts.cuh"
#include "gorpho_util.cuh"
#include "helper_math.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////
// General 3D grayscale dilation / erosion
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T, GorphoOp op>
__global__ void genDilateErode3dKernel(const T *__restrict__ vol, const T *__restrict__ strel,
	T *__restrict__ res, const int3 volsiz, const int3 strelsiz)
{
	const int3 pos = getGlobalPos_3D();

	// Make sure we are within array bounds
	if (pos < volsiz) {
		// Precompute start- and endpoints
		const int3 rstart = pos - strelsiz / 2;
		const int3 start = max(make_int3(0), rstart);
		const int3 end = min(volsiz - 1, pos + (strelsiz - 1) / 2);

		// Find and store value for this position
		T val;
		if (op == GORPHO_DILATE) {
			val = minusInfOrMin<T>();
		} else if (op == GORPHO_ERODE) {
			val = infOrMax<T>();
		}
		for (int iz = start.z; iz <= end.z; iz++) {
			for (int iy = start.y; iy <= end.y; iy++) {
				for (int ix = start.x; ix <= end.x; ix++) {
					const size_t vidx = getIdx(ix, iy, iz, volsiz);
					const size_t sidx = getIdx(ix - rstart.x, iy - rstart.y, iz - rstart.z, strelsiz);
					// TODO: These additions might overflow or underflow for integer data - replace with
					// saturating arithmetic
					if (op == GORPHO_DILATE) {
						val = max(val, vol[vidx] + strel[sidx]);
					} else if (op == GORPHO_ERODE) {
						val = min(val, vol[vidx] - strel[sidx]);
					}
				}
			}
		}
		res[getIdx(pos, volsiz)] = val;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Flat 3D grayscale linear dilation / erosion
////////////////////////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ int3 get_start_pos(const int3 grid_pos, const GorphoDir dir,
	const int3 step, const int3 volsiz)
{
	if (dir & GORPHO_DIR_X) {
		return make_int3((step.x > 0) ? 0 : volsiz.x - 1, grid_pos.y, grid_pos.z);
	} else if (dir & GORPHO_DIR_Y) {
		return make_int3(grid_pos.y, (step.y > 0) ? 0 : volsiz.y - 1, grid_pos.z);
	} else { // dir & GORPHO_DIR_Z
		return make_int3(grid_pos.y, grid_pos.z, (step.z > 0) ? 0 : volsiz.z - 1);
	}
}

inline __host__ __device__ size_t get_bufidx(const int3 grid_pos, const GorphoDir dir,
	const int3 volsiz)
{
	if (dir & GORPHO_DIR_X) {
		return (grid_pos.y + grid_pos.z*volsiz.y);
	} else if (dir & GORPHO_DIR_Y) {
		return (grid_pos.y + grid_pos.z*volsiz.x);
	} else { // dir & GORPHO_DIR_Z
		return (grid_pos.y + grid_pos.z*volsiz.x);
	}
}

inline __host__ __device__ size_t get_bufsiz(const GorphoDir dir, const int3 volsiz)
{
	if (dir & GORPHO_DIR_X) {
		return volsiz.y*volsiz.z;
	} else if (dir & GORPHO_DIR_Y) {
		return volsiz.x*volsiz.z;
	} else { // dir & GORPHO_DIR_Z
		return volsiz.x*volsiz.y;
	}
}

template <typename T, GorphoOp op>
__global__ void flatLinearDilateErode3dKernel(const T *__restrict__ vol, const int strelsiz,
	const GorphoDir dir, const int3 step, T *__restrict__ R, T *__restrict__ S, T *__restrict__ res,
	const int3 volsiz)
{
	const int hstrelsiz = strelsiz / 2;
	const int3 grid_pos = getGlobalPos_3D();
	const int3 start = get_start_pos(grid_pos, dir, step, volsiz);
	const int bufoffset = get_bufidx(grid_pos, dir, volsiz); // Offset into R and S
	const int bufstep = get_bufsiz(dir, volsiz);
	const T padval = (op == GORPHO_ERODE) ? infOrMax<T>() : minusInfOrMin<T>();

	T *const RS = (grid_pos.x == 0) ? S : R;
	const int sdir = (grid_pos.x == 0) ? 1 : -1;

	if (start >= 0 && start < volsiz) {
		// Initial boundary roll - only fill S buffer
		if (grid_pos.x == 0) {
			S[bufoffset] = vol[getIdx(start, volsiz)];
			for (int k = 1; k < strelsiz; k++) {
				const int3 posk = start + k * step;
				const T sv = (posk >= 0 && posk < volsiz) ? vol[getIdx(posk, volsiz)] : padval;
				if (op == GORPHO_ERODE) {
					S[bufoffset + k * bufstep] = min(sv, S[bufoffset + (k - 1)*bufstep]);
				} else if (op == GORPHO_DILATE) {
					S[bufoffset + k * bufstep] = max(sv, S[bufoffset + (k - 1)*bufstep]);
				}
			}
			for (int k = 0; k <= hstrelsiz; k++) {
				const int3 posk = start + step * (hstrelsiz - k);
				if (posk >= 0 && posk < volsiz) {
					const size_t ridx = getIdx(posk, volsiz);
					res[ridx] = S[bufoffset + (strelsiz - k - 1)*bufstep];
				}
			}
		}
		__syncthreads();

		// Normal van Herk Gil Werman roll
		int3 pos = start + strelsiz * step;
		for (; pos >= -step * strelsiz && pos < volsiz - step * strelsiz; pos += step * strelsiz) {
			RS[bufoffset] = vol[getIdx(pos, volsiz)];
			int k = 1;
			// Loop unrolled for speed. Using #pragma unroll does not seem to have any effect here.
			// NOTE: Loop unrolling only seems to have an effect when moving along the x-direction
			int posRs = bufoffset;
			for (; k + 3 < strelsiz; k += 4) {
				const T rsv1 = vol[getIdx(pos + sdir * (k + 0)*step, volsiz)];
				const T rsv2 = vol[getIdx(pos + sdir * (k + 1)*step, volsiz)];
				const T rsv3 = vol[getIdx(pos + sdir * (k + 2)*step, volsiz)];
				const T rsv4 = vol[getIdx(pos + sdir * (k + 3)*step, volsiz)];
				if (op == GORPHO_ERODE) {
					RS[posRs + 1 * bufstep] = min(rsv1, RS[posRs + 0 * bufstep]);
					RS[posRs + 2 * bufstep] = min(rsv2, RS[posRs + 1 * bufstep]);
					RS[posRs + 3 * bufstep] = min(rsv3, RS[posRs + 2 * bufstep]);
					RS[posRs + 4 * bufstep] = min(rsv4, RS[posRs + 3 * bufstep]);
				} else if (op == GORPHO_DILATE) {
					RS[posRs + 1 * bufstep] = max(rsv1, RS[posRs + 0 * bufstep]);
					RS[posRs + 2 * bufstep] = max(rsv2, RS[posRs + 1 * bufstep]);
					RS[posRs + 3 * bufstep] = max(rsv3, RS[posRs + 2 * bufstep]);
					RS[posRs + 4 * bufstep] = max(rsv4, RS[posRs + 3 * bufstep]);
				}
				posRs += 4 * bufstep;
			}
			// Handle leftovers
			for (; k < strelsiz; k++) {
				// TODO: Use posRs
				const T rsv1 = vol[getIdx(pos + sdir * k*step, volsiz)];
				if (op == GORPHO_ERODE) {
					RS[bufoffset + k * bufstep] = min(rsv1, RS[bufoffset + (k - 1)*bufstep]);
				} else if (op == GORPHO_DILATE) {
					RS[bufoffset + k * bufstep] = max(rsv1, RS[bufoffset + (k - 1)*bufstep]);
				}
			}
			__syncthreads();
			int posR = bufoffset + grid_pos.x*bufstep;
			int posS = bufoffset + (strelsiz - grid_pos.x - 1)*bufstep;
			for (int k = grid_pos.x; k < strelsiz; k += 2) {
				const int ridx1 = getIdx(pos + step * (hstrelsiz - k), volsiz);
				if (op == GORPHO_ERODE) {
					res[ridx1] = min(R[posR], S[posS]);
				} else if (op == GORPHO_DILATE) {
					res[ridx1] = max(R[posR], S[posS]);
				}
				posR += 2 * bufstep;
				posS -= 2 * bufstep;
			}
			__syncthreads();
		}

		// End boundary roll
		// This has a lot of branching code, so we want it in a seperate loop
		for (; pos >= step * strelsiz && pos <= volsiz + step * strelsiz; pos += step * strelsiz) {
			if (pos >= 0 && pos < volsiz) {
				RS[bufoffset] = vol[getIdx(pos, volsiz)];
			} else {
				RS[bufoffset] = padval;
			}
			for (int k = 1; k < strelsiz; k++) {
				const int3 posk = pos + sdir * k*step;
				const T rsv = (posk >= 0 && posk < volsiz) ? vol[getIdx(pos + sdir * k*step, volsiz)] : padval;
				if (op == GORPHO_ERODE) {
					RS[bufoffset + k * bufstep] = min(rsv, RS[bufoffset + (k - 1)*bufstep]);
				} else if (op == GORPHO_DILATE) {
					RS[bufoffset + k * bufstep] = max(rsv, RS[bufoffset + (k - 1)*bufstep]);
				}
			}
			__syncthreads();
			for (int k = hstrelsiz * grid_pos.x; k < hstrelsiz + (strelsiz - hstrelsiz)*grid_pos.x; k++) {
				const int3 posk = pos + step * (hstrelsiz - k);
				if (posk >= 0 && posk < volsiz) {
					const size_t ridx = getIdx(posk, volsiz);
					if (op == GORPHO_ERODE) {
						res[ridx] = min(R[bufoffset + k * bufstep], S[bufoffset + (strelsiz - k - 1)*bufstep]);
					} else if (op == GORPHO_DILATE) {
						res[ridx] = max(R[bufoffset + k * bufstep], S[bufoffset + (strelsiz - k - 1)*bufstep]);
					}
				}
			}
			__syncthreads();
		}
	}
}

#endif /* __GORPHO_KERNELS_H__ */
