/**
  * Range class.
  *
  * Author: Ying Xiong.
  * Created: Jan 26, 2015.
  */

#ifndef __XYUTILS_RANGE_H__
#define __XYUTILS_RANGE_H__

namespace xyUtils  {
// A range of integers from `start` (inclusive) to `end` (exclusive) with a
// given `step`.
class Range {
 public:
  Range(int start, int end, int step = 1)
      : start_(start), end_(end), step_(step) { }
  int start() const { return start_; }
  int end() const { return end_; }
  int step() const { return step_; }
  // The number of elements in the range.
  int size() const {
    if (step_ > 0) {
      return (end_ - start_ - 1) / step_ + 1;
    } else {
      return (start_ - end_ - 1) / (-step_) + 1;
    }
  }
 private:
  int start_, end_, step_;
};
}   // namespace xyUtils

#endif   // __XYUTILS_RANGE_H__
