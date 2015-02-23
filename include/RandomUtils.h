/**
  * Some utility functions for generating random numbers.
  *
  * Author: Ying Xiong.
  * Created: Feb 23, 2015.
  */

#ifndef __XYUTILS_RANDOM_UTILS_H__
#define __XYUTILS_RANDOM_UTILS_H__

#include <vector>

namespace xyUtils  {

// Generate `num` different random numbers in the range [`low`, `high`).
std::vector<int> RandomIntegers(int low, int high, int num);

}   // namespace xyUtils

#endif   // __XYUTILS_RANDOM_UTILS_H__
