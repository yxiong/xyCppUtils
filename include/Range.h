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
// A range of numbers from `start` (inclusive) to `stop` (exclusive) with a
// given `step`. More specifically, the numbers in the range are
//     [start, start+step, start+2*step, ..., start+(size-1)*step]
// where
//     start + step * (size-1) < stop       if step > 0
//     start + step * (size-1) > stop       if step < 0
//
// The numbers in the range can be floating point, but note the rounding effect
// when doing so. The suggested way is to pad an `epsilon` at the `stop`:
//     Range(1.5, 1.8001, 0.3)   // 1.8 will be included.
//     Range(1.5, 1.7999, 0.3)   // 1.5 will be excluded.
//     Range(1.5, 1.8, 0.3)      // 1.8 should be excluded, but might not be
//                               // because of rounding effect. Avoid this.
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
