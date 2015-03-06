/**
  * Range class.
  *
  * Author: Ying Xiong.
  * Created: Jan 26, 2015.
  */

#ifndef __XYUTILS_RANGE_H__
#define __XYUTILS_RANGE_H__

#include <cmath>
#include <type_traits>

namespace xyUtils  {
// A range of integers from `start` (inclusive) to `stop` (exclusive) with a
// given `step`.
template <typename T=int>
class Range {
 public:
  Range(T start, T stop, T step = 1)
      : start_(start), stop_(stop), step_(step) { }
  T start() const { return start_; }
  T stop() const { return stop_; }
  T step() const { return step_; }
  // The number of elements in the range.
  int size() const {
    return size_(std::is_integral<T>());
  }
 private:
  // Compute size for integral type `T`.
  int size_(std::true_type) const {
    if (step_ > 0) {
      return (stop_ - start_ - 1) / step_ + 1;
    } else {
      return (start_ - stop_ - 1) / (-step_) + 1;
    }
  }
  // Compute size for floating point type `T`.
  int size_(std::false_type) const {
    static_assert(std::is_floating_point<T>(),
                  "Range: 'T' has to be integral or floating point type.");
    return ceil((stop_ - start_) / step_);
  }
  T start_, stop_, step_;
};
}   // namespace xyUtils

#endif   // __XYUTILS_RANGE_H__
