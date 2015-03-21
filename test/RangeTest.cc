/**
  * Test on Range class.
  *
  * Author: Ying Xiong.
  * Created: Jan 26, 2015.
  */

#include "Range.h"

#include "LogAndCheck.h"
#include "Timer.h"

using namespace xyUtils;

int main()  {
  Timer timer;
  LOG(INFO) << "Test on Range class...";

  Range<int> r(5, 7, 2);
  CHECK_EQ(r.start(), 5);
  CHECK_EQ(r.stop(), 7);
  CHECK_EQ(r.step(), 2);
  CHECK_EQ(r.size(), 1);
  CHECK_EQ(r[0], 5);

  r = Range<int>(9, 0, -2);
  CHECK_EQ(r.start(), 9);
  CHECK_EQ(r.stop(), 0);
  CHECK_EQ(r.step(), -2);
  CHECK_EQ(r.size(), 5);
  CHECK_EQ(r[0], 9);
  CHECK_EQ(r[2], 5);
  CHECK_EQ(r[4], 1);

  Range<float> rf(0.0f, 1.5001f, 0.3f);
  CHECK_EQ(rf.start(), 0.0f);
  CHECK_EQ(rf.stop(), 1.5001f);
  CHECK_EQ(rf.step(), 0.3f);
  CHECK_EQ(rf.size(), 6);
  CHECK_NEAR(rf[1], 0.3f, 0.001f);
  CHECK_NEAR(rf[3], 0.9f, 0.001f);

  rf = Range<float>(1.5f, 0.001f, -0.3f);
  CHECK_EQ(rf.start(), 1.5f);
  CHECK_EQ(rf.stop(), 0.001f);
  CHECK_EQ(rf.step(), -0.3f);
  CHECK_EQ(rf.size(), 5);
  CHECK_NEAR(rf[0], 1.5f, 0.001f);
  CHECK_NEAR(rf[2], 0.9f, 0.001f);

  LOG(INFO) << "Passed. [" << timer.elapsed() << " seconds]";
  return 0;
}
