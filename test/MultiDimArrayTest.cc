/**
  * Test on MultiDimArray.
  *
  * Author: Ying Xiong.
  * Created: Jan 14, 2015.
  */

#include "MultiDimArray.h"

#include "LogAndCheck.h"
#include "Timer.h"

using namespace xyUtils;

int main()  {
  Timer timer;
  LOG(INFO) << "Test on ...";

  int dims1[1] = {24};
  MultiDimArray<float,1> mdarray1(dims1, NULL);
  mdarray1(5) = 5.0;
  CHECK_NEAR(mdarray1(5), 5.0, 0.0001);

  int dims2[2] = {4,6};
  MultiDimArray<float,2> mdarray2(dims2, NULL);
  CHECK_EQ(mdarray2.GetDim(0), 4);
  CHECK_EQ(mdarray2.GetDim(1), 6);
  CHECK_EQ(mdarray2.GetStride(0), 6);
  CHECK_EQ(mdarray2.GetStride(1), 1);
  mdarray2(2,3) = 3.0;
  CHECK_NEAR(mdarray2(2,3), 3.0, 0.0001);

  dims2[0] = 2, dims2[1] = 3;
  int strides2[2] = {3, 1};
  float data2[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  mdarray2 = MultiDimArray<float,2>(dims2, strides2, data2);
  CHECK_NEAR(mdarray2(0,2), 3.0, 0.0001);
  CHECK_NEAR(mdarray2(1,1), 5.0, 0.0001);

  int dims3[3] = {2, 4, 6};
  int strides3[3] = {1, 2, 8};
  MultiDimArray<float,3> mdarray3(dims3, strides3);
  CHECK_EQ(mdarray3.GetNumElements(), 48);
  CHECK_EQ(mdarray3.GetStride(0), 1);
  CHECK_EQ(mdarray3.GetStride(1), 2);
  CHECK_EQ(mdarray3.GetStride(2), 8);
  mdarray3(1,3,2) = 9.0;
  CHECK_NEAR(mdarray3(1,3,2), 9.0, 0.0001);

  LOG(INFO) << "Passed. [" << timer.elapsed() << " seconds]";
  return 0;
}
