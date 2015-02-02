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

  Range r(5, 7, 2);
  CHECK_EQ(r.start(), 5);
  CHECK_EQ(r.end(), 7);
  CHECK_EQ(r.step(), 2);
  CHECK_EQ(r.size(), 1);

  r = Range(9, 0, -2);
  CHECK_EQ(r.start(), 9);
  CHECK_EQ(r.end(), 0);
  CHECK_EQ(r.step(), -2);
  CHECK_EQ(r.size(), 5);

  LOG(INFO) << "Passed. [" << timer.elapsed() << " seconds]";
  return 0;
}
