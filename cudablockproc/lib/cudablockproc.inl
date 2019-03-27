#include "cudablockproc.cuh"

namespace cbp {

namespace detail {

template <class Ty>
struct typeSize : public std::integral_constant<size_t, sizeof(Ty)> {};
template <>
struct typeSize<void> : public std::integral_constant<size_t, 1> {};

// Code for zip class is from: https://gist.github.com/mortehu/373069390c75b02f98b655e3f7dbef9a
template <typename... T>
class zip_helper {
 public:
  class iterator
      : std::iterator<std::forward_iterator_tag,
                      std::tuple<decltype(*std::declval<T>().begin())...>> {
   private:
    std::tuple<decltype(std::declval<T>().begin())...> iters_;

    template <std::size_t... I>
    auto deref(std::index_sequence<I...>) const {
      return typename iterator::value_type{*std::get<I>(iters_)...};
    }

    template <std::size_t... I>
    void increment(std::index_sequence<I...>) {
      auto l = {(++std::get<I>(iters_), 0)...};
    }

   public:
    explicit iterator(decltype(iters_) iters) : iters_{std::move(iters)} {}

    iterator& operator++() {
      increment(std::index_sequence_for<T...>{});
      return *this;
    }

    iterator operator++(int) {
      auto saved{*this};
      increment(std::index_sequence_for<T...>{});
      return saved;
    }

    bool operator!=(const iterator& other) const {
      return iters_ != other.iters_;
    }

    auto operator*() const { return deref(std::index_sequence_for<T...>{}); }
  };

  zip_helper(T&... seqs)
      : begin_{std::make_tuple(seqs.begin()...)},
        end_{std::make_tuple(seqs.end()...)} {}

  iterator begin() const { return begin_; }
  iterator end() const { return end_; }

 private:
  iterator begin_;
  iterator end_;
};

// Sequences must be the same length.
template <typename... T>
auto zip(T&&... seqs) {
  return zip_helper<T...>{seqs...};
}

} // namespace detail

MemLocation getMemLocation(const void *ptr)
{
    cudaPointerAttributes attr;
    const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err == cudaSuccess) {
        if (attr.memoryType == cudaMemoryTypeHost) {
            return HOST_PINNED;
        } else {
            return DEVICE;
        }
    } else {
        cudaGetLastError(); // Pop error so we don't disturb future calls
        return HOST_NORMAL;
    }
}

template <MemLocation loc>
bool memLocationIs(const void *ptr)
{
    return cbp::getMemLocation(ptr) == loc;
}

inline CbpResult operator| (CbpResult lhs, CbpResult rhs)
{
    return static_cast<CbpResult>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline CbpResult operator& (CbpResult lhs, CbpResult rhs)
{
    return static_cast<CbpResult>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline CbpResult& operator|= (CbpResult& lhs, CbpResult rhs)
{
    lhs = lhs | rhs;
    return lhs;
}

inline CbpResult& operator&= (CbpResult& lhs, CbpResult rhs)
{
    lhs = lhs & rhs;
    return lhs;
}

template <class InTy, class OutTy, class Func>
CbpResult blockProc(Func func, InTy *inVol, OutTy *outVol,
    cbp::BlockIndexIterator blockIter, const size_t tmpSize)
{
    std::array<InTy *, 1> inVols = { inVol };
    std::array<OutTy *, 1> outVols = { outVol };
    return blockProcMultiple(func, inVols, outVols, blockIter, tmpSize);
}

template <class InArr, class OutArr, class Func>
CbpResult blockProcMultiple(Func func, const InArr& inVols, const OutArr& outVols,
    cbp::BlockIndexIterator blockIter, const size_t tmpSize)
{
    const int3 blockSize = blockIter.blockSize();
    const int3 borderSize = blockIter.borderSize();
    std::vector<typename InArr::value_type> inBlocks, d_inBlocks;
    std::vector<typename OutArr::value_type> outBlocks, d_outBlocks;
    void *d_tmpMem = nullptr;

    // TODO: Use a scope guard
    auto cleanUp = [&](){
        std::for_each(inBlocks.begin(), inBlocks.end(), cudaFreeHost);
        std::for_each(d_inBlocks.begin(), d_inBlocks.end(), cudaFree);
        std::for_each(outBlocks.begin(), outBlocks.end(), cudaFreeHost);
        std::for_each(d_outBlocks.begin(), d_outBlocks.end(), cudaFree);
        cudaFree(d_tmpMem);
    };

    CbpResult res = cbp::allocBlocks(inBlocks, inVols.size(), HOST_PINNED, blockSize, borderSize);
    res |= cbp::allocBlocks(d_inBlocks, inVols.size(), DEVICE, blockSize, borderSize);
    res |= cbp::allocBlocks(outBlocks, outVols.size(), HOST_PINNED, blockSize, borderSize);
    res |= cbp::allocBlocks(d_outBlocks, outVols.size(), DEVICE, blockSize, borderSize);
    if (tmpSize > 0) {
        if (cudaMalloc(&d_tmpMem, tmpSize) != cudaSuccess) {
            res |= CBP_DEVICE_MEM_ALLOC_FAIL;
        }
    }
    if (res != CBP_SUCCESS) {
        cleanUp();
        return res;
    }

    try {
        res = cbp::blockProcMultipleNoValidate(func, inVols, outVols, inBlocks, outBlocks,
                                       d_inBlocks, d_outBlocks, blockIter, d_tmpMem);
        cleanUp();
    } catch (...) {
        // In case an exception was thrown, ensure memory is freed and then rethrow
        cleanUp();
        std::rethrow_exception(std::current_exception());
    }
    return res;
}

template <class InArr, class OutArr, class InHBlkArr, class OutHBlkArr, class InDBlkArr, class OutDBlkArr,
    class Func>
CbpResult blockProcMultiple(Func func, const InArr& inVols, const OutArr& outVols,
    const InHBlkArr& inBlocks, const OutHBlkArr& outBlocks,
    const InDBlkArr& d_inBlocks, const OutDBlkArr& d_outBlocks,
    cbp::BlockIndexIterator blockIter, void *d_tmpMem)
{
    // Verify all blocks are pinned memory
    for (auto blockArray : { inBlocks, outBlocks }) {
        if (!std::all_of(blockArray.begin(), blockArray.end(), memLocationIs<HOST_PINNED>)) {
            return CBP_INVALID_MEM_LOC;
        }
    }
    // Verify all device blocks are on the device
    for (auto d_blockArray : { d_inBlocks, d_outBlocks }) {
        if (!std::all_of(d_blockArray.begin(), d_blockArray.end(), memLocationIs<DEVICE>)) {
            return CBP_INVALID_MEM_LOC;
        }
    }
    if (d_tmpMem != nullptr && !memLocationIs<DEVICE>(d_tmpMem)) {
        return CBP_INVALID_MEM_LOC;
    }

    return cbp::blockProcMultipleNoValidate(func, inVols, outVols,
        inBlocks, outBlocks, d_inBlocks, d_outBlocks, blockIter, d_tmpMem);
}

template <class InArr, class OutArr, class InHBlkArr, class OutHBlkArr, class InDBlkArr, class OutDBlkArr,
    class Func>
CbpResult blockProcMultipleNoValidate(Func func, const InArr& inVols, const OutArr& outVols,
    const InHBlkArr& inBlocks, const OutHBlkArr& outBlocks,
    const InDBlkArr& d_inBlocks, const OutDBlkArr& d_outBlocks,
    cbp::BlockIndexIterator blockIter, void *d_tmpMem)
{
    static_assert(std::is_same<typename InArr::value_type, typename InHBlkArr::value_type>::value,
        "Value types for input volumes and host input blocks must be equal");
    static_assert(std::is_same<typename InArr::value_type, typename InDBlkArr::value_type>::value,
        "Value types for input volumes and device input blocks must be equal");
    static_assert(std::is_same<typename OutArr::value_type, typename OutHBlkArr::value_type>::value,
        "Value types for output volumes and host output blocks must be equal");
    static_assert(std::is_same<typename OutArr::value_type, typename OutDBlkArr::value_type>::value,
        "Value types for output volumes and device output blocks must be equal");

    const int3 volSize = blockIter.volSize();
    size_t blockCount = blockIter.maxLinearIndex() + 1;
    std::vector<cudaStream_t> streams(blockCount);
    std::vector<cudaEvent_t> events(blockCount);
    // Create streams and events

    for (auto& s : streams) {
        cudaStreamCreate(&s);
    }
    for (auto& e : events) {
        cudaEventCreate(&e);
    }
    auto crntBlockIdx = blockIter[0];
    auto crntStream = streams[0];

    // Start transfer and copy for first block
    cbp::blockVolumeTransferAll(inVols, inBlocks, crntBlockIdx, volSize, VOL_TO_BLOCK, crntStream);
    cbp::hostDeviceTransferAll(d_inBlocks, inBlocks, crntBlockIdx, cudaMemcpyHostToDevice, crntStream);

    // Process remaining blocks
    auto zipped = detail::zip(events, streams, blockIter);
    auto prevBlockIdx = crntBlockIdx;
    auto prevStream = crntStream;
    cudaEvent_t crntEvent;
    // We skip the first block as we started it above
    for (auto crnt = ++(zipped.begin()); crnt != zipped.end(); ++crnt) {
        std::tie(crntEvent, crntStream, crntBlockIdx) = *crnt;

        cudaEventRecord(crntEvent);
        func(prevBlockIdx, prevStream, d_inBlocks, d_outBlocks, d_tmpMem);

        cudaStreamWaitEvent(crntStream, crntEvent, 0);
        cbp::blockVolumeTransferAll(inVols, inBlocks, crntBlockIdx, volSize, VOL_TO_BLOCK, crntStream);
        cbp::hostDeviceTransferAll(d_inBlocks, inBlocks, crntBlockIdx, cudaMemcpyHostToDevice, crntStream);
        cbp::hostDeviceTransferAll(outBlocks, d_outBlocks, prevBlockIdx, cudaMemcpyDeviceToHost, prevStream);
        cbp::blockVolumeTransferAll(outVols, outBlocks, prevBlockIdx, volSize, BLOCK_TO_VOL, prevStream);

        prevBlockIdx = crntBlockIdx;
        prevStream = crntStream;
    }
    func(prevBlockIdx, prevStream, d_inBlocks, d_outBlocks, d_tmpMem);
    cbp::hostDeviceTransferAll(outBlocks, d_outBlocks, prevBlockIdx, cudaMemcpyDeviceToHost, prevStream);
    cbp::blockVolumeTransferAll(outVols, outBlocks, prevBlockIdx, volSize, BLOCK_TO_VOL, prevStream);
    cudaStreamSynchronize(prevStream);

    for (auto& s : streams) {
        cudaStreamDestroy(s);
    }
    return CBP_SUCCESS;
}

template <typename DstArr, typename SrcArr>
void hostDeviceTransferAll(const DstArr& dstArray, const SrcArr& srcArray, const BlockIndex& blkIdx,
    cudaMemcpyKind kind, cudaStream_t stream)
{
    typename DstArr::value_type dstPtr;
    typename SrcArr::value_type srcPtr;
    static_assert(std::is_same<decltype(dstPtr), decltype(srcPtr)>::value,
        "Destination and source must have same type");
    static_assert(std::is_pointer_v<decltype(dstPtr)>, "dstArray must contain pointers");
    static_assert(std::is_pointer_v<decltype(srcPtr)>, "srcArray must contain pointers");
    const size_t sizeOfValueType = detail::typeSize<std::remove_pointer_t<decltype(dstPtr)>>();
    for (auto ptrs : detail::zip(dstArray, srcArray)) {
        std::tie(dstPtr, srcPtr) = ptrs;
        cudaMemcpyAsync(dstPtr, srcPtr, blkIdx.numel()*sizeOfValueType, kind, stream);
    }
}

template <typename VolArr, typename BlkArr>
void blockVolumeTransferAll(const VolArr& volArray, const BlkArr& blockArray, const BlockIndex& blkIdx,
    int3 volSize, BlockTransferKind kind, cudaStream_t stream)
{
    typename VolArr::value_type volPtr;
    typename BlkArr::value_type blkPtr;
    static_assert(std::is_same<decltype(volPtr), decltype(blkPtr)>::value,
        "Volume and block must have same type");
    static_assert(std::is_pointer_v<decltype(volPtr)>, "volArray must contain pointers");
    static_assert(std::is_pointer_v<decltype(blkPtr)>, "blockArray must contain pointers");
    for (auto ptrs : detail::zip(volArray, blockArray)) {
        std::tie(volPtr, blkPtr) = ptrs;
        cbp::blockVolumeTransfer(volPtr, blkPtr, blkIdx, volSize, kind, stream);
    }
}

template <typename Ty>
void blockVolumeTransfer(Ty *vol, Ty *block, const BlockIndex& bi, int3 volSize, BlockTransferKind kind,
    cudaStream_t stream)
{
    // TODO: Allow vol or block to be a const pointer - maybe use templates?
    // TODO: Allow caller to specify which axis corresponds to consecutive values.
    static const size_t sizeOfTy = detail::typeSize<Ty>();
    int3 start, bsize;
    const int3 blkSizeBdr = bi.blockSizeBorder();
    cudaMemcpy3DParms params = { 0 };
    const auto volPtr = make_cudaPitchedPtr(vol, volSize.x * sizeOfTy, volSize.x, volSize.y);
    const auto blockPtr = make_cudaPitchedPtr(block, blkSizeBdr.x * sizeOfTy, blkSizeBdr.x, blkSizeBdr.y);
    if (kind == VOL_TO_BLOCK) {
        start = bi.startIdxBorder;
        bsize = bi.blockSizeBorder();

        params.srcPtr = volPtr;
        params.dstPtr = blockPtr;
        params.srcPos = make_cudaPos(start.x * sizeOfTy, start.y, start.z);
        params.dstPos = make_cudaPos(0, 0, 0);
    } else {
        // If we are transferring back to the volume we need to discard the border
        start = bi.startIdx;
        bsize = bi.blockSize();
        int3 startBorder = bi.startBorder();

        params.dstPtr = volPtr;
        params.srcPtr = blockPtr;
        params.dstPos = make_cudaPos(start.x * sizeOfTy, start.y, start.z);
        params.srcPos = make_cudaPos(startBorder.x * sizeOfTy, startBorder.y, startBorder.z);
    }

    params.kind = cudaMemcpyHostToHost;
    params.extent = make_cudaExtent(bsize.x * sizeOfTy, bsize.y, bsize.z);

    cudaMemcpy3DAsync(&params, stream);
}

template <typename Ty>
CbpResult allocBlocks(std::vector<Ty *>& blocks, const size_t n, const MemLocation loc, const int3 blockSize,
    const int3 borderSize) noexcept
{
    const int3 totalSize = blockSize + 2*borderSize;
    const size_t nbytes = detail::typeSize<Ty>()*(totalSize.x * totalSize.y * totalSize.z);
    blocks.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Ty *ptr;
        if (loc == HOST_NORMAL) {
            ptr = static_cast<Ty *>(malloc(nbytes));
            if (ptr == nullptr) {
                return CBP_HOST_MEM_ALLOC_FAIL;
            }
        } else if (loc == HOST_PINNED) {
            if (cudaMallocHost(&ptr, nbytes) != cudaSuccess) {
                return CBP_HOST_MEM_ALLOC_FAIL;
            }
        } else if (loc == DEVICE) {
            if (cudaMalloc(&ptr, nbytes) != cudaSuccess) {
                return CBP_DEVICE_MEM_ALLOC_FAIL;
            }
        } else {
            return CBP_INVALID_VALUE;
        }
        blocks.push_back(ptr);
    }
    return CBP_SUCCESS;
}

} // namespace cbp