/**
  * Description.
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
  LOG(INFO) << "Test on ...";

  Range<int> r(5, 7, 2);
  CHECK_EQ(r.start(), 5);
  CHECK_EQ(r.stop(), 7);
  CHECK_EQ(r.step(), 2);
  CHECK_EQ(r.size(), 1);

  r = Range<int>(9, 0, -2);
  CHECK_EQ(r.start(), 9);
  CHECK_EQ(r.stop(), 0);
  CHECK_EQ(r.step(), -2);
  CHECK_EQ(r.size(), 5);

  Range<float> rf(0.0f, 1.5001f, 0.3f);
  CHECK_EQ(rf.start(), 0.0f);
  CHECK_EQ(rf.stop(), 1.5001f);
  CHECK_EQ(rf.step(), 0.3f);
  CHECK_EQ(rf.size(), 6);

  rf = Range<float>(1.5f, 0.001f, -0.3f);
  CHECK_EQ(rf.start(), 1.5f);
  CHECK_EQ(rf.stop(), 0.001f);
  CHECK_EQ(rf.step(), -0.3f);
  CHECK_EQ(rf.size(), 5);

  LOG(INFO) << "Passed. [" << timer.elapsed() << " seconds]";
  return 0;
}
