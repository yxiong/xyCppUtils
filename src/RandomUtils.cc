/**
  * Some utility functions for generating random numbers.
  *
  * Author: Ying Xiong.
  * Created: Feb 23, 2015.
  */

#include "RandomUtils.h"

#include <cstdlib>
#include <vector>

namespace xyUtils  {

std::vector<int> RandomIntegers(int low, int high, int num) {
  int size = high-low;
  std::vector<int> choices(size);
  for (int i = low; i < high; ++i) {
    choices[i-low] = i;
  }
  std::vector<int> result(num);
  for (int i = 0; i < num; ++i) {
    int idx = rand() % (size-i);
    result[i] = choices[idx];
    choices[idx] = choices[size-i-1];
  }
  return result;
}

}   // namespace xyUtils
