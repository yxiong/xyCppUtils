/**
 * Range class for CUDA.
 *
 * Author: Ying Xiong.
 * Created: Mar 20, 2015.
 */

#ifndef __XYUTILS_CUDA_RANGE_H__
#define __XYUTILS_CUDA_RANGE_H__

#include <type_traits>

#include "../Range.h"

namespace xyUtils {
namespace cuda {
// This class has the same semantic as xyUtils::Range, and member functions can
// be called on device as well.
template <typename T=int>
class Range {
 public:
  __host__ __device__ Range(T start, T stop, T step = 1)
      : start_(start), stop_(stop), step_(step) { }

  // Construct a xyUtils::cuda::Range from xyUtils::Range.
  __host__ Range(const xyUtils::Range<T>& r)
      : start_(r.start()), stop_(r.stop()), step_(r.step()) { }

  __host__ __device__ T start() const { return start_; }
  __host__ __device__ T stop() const { return stop_; }
  __host__ __device__ T step() const { return step_; }
  // The number of elements in the range.
  __host__ __device__ int size() const { return size_(std::is_integral<T>()); }
  // Get `i`-th element in the range (zero-based).
  __host__ __device__ T operator[](int i) const { return start_ + step_ * i; }
 private:
  // Compute size for integral type `T`.
  __host__ __device__ int size_(std::true_type) const {
    if (step_ > 0) {
      return (stop_ - start_ - 1) / step_ + 1;
    } else {
      return (start_ - stop_ - 1) / (-step_) + 1;
    }
  }
  // Compute size for floating point type `T`.
  __host__ __device__ int size_(std::false_type) const {
#ifndef __CUDACC__   // CUDA compiler has some issues on the following check.
    static_assert(std::is_floating_point<T>(),
                  "Range: 'T' has to be integral or floating point type.");
#endif
    return ceil((stop_ - start_) / step_);
  }
  T start_, stop_, step_;
};

}   // namespace cuda
}   // namespace xyUtils

#endif   // __XYUTILS_CUDA_RANGE_H__
