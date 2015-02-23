/**
  * Test on RandomUtils.
  *
  * Author: Ying Xiong.
  * Created: Feb 23, 2015.
  */

#include "RandomUtils.h"

#include <vector>

#include "LogAndCheck.h"
#include "Timer.h"

using namespace std;
using namespace xyUtils;

int main()  {
  Timer timer;
  LOG(INFO) << "Test on RandomUtils ...";

  vector<int> r = RandomIntegers(10, 100, 20);
  // Sanity check.
  CHECK_EQ(r.size(), 20);
  for (auto i : r) {
    CHECK_GE(i, 10);
    CHECK_LT(i, 100);
  }
  // Check that all numbers are different.
  for (int i = 0; i < r.size(); ++i) {
    for (int j = i+1; j < r.size(); ++j) {
      CHECK_NE(i, j);
    }
  }

  LOG(INFO) << "Passed. [" << timer.elapsed() << " seconds]";
  return 0;
}
