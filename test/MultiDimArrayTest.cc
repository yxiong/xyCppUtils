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

void IndexTest() {
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
}

void SliceTest() {
  // Slice dimension by index.
  int dims[4] = {3, 5, 2, 6};
  MultiDimArray<int,4> mda1(dims);  // 3x5x2x6.
  mda1(0,3,1,4) = 100;
  MultiDimArray<int,3> mda2 = mda1.SliceDim(1, 3);  // 3x2x6.
  CHECK_EQ(mda1(0,3,1,4), mda2(0,1,4));
  MultiDimArray<int,2> mda3 = mda2.SliceDim(2, 4);  // 3x2.
  CHECK_EQ(mda1(0,3,1,4), mda3(0,1));

  // Slice dimension by range.
  Range<int> r(2,4);
  MultiDimArray<int,4> mda4 = mda1.SliceDim(1, r);  // 3x2x2x6.
  CHECK_EQ(mda4.GetDim(1), 2);
  CHECK_EQ(mda1(0,3,1,4), mda4(0,1,1,4));
  r = Range<int>(4, -1, -2);
  mda4 = mda1.SliceDim(3, r);  // 3x5x3x6.
  CHECK_EQ(mda4.GetDim(3), 3);
  CHECK_EQ(mda1(0,3,1,4), mda4(0,3,1,0));
}

void OperatorTest() {
  int dims[4] = {3, 5, 2, 6};
  MultiDimArray<int,4> mda1(dims);
  mda1(0,3,1,4) = 100;
  // AssignData.
  MultiDimArray<int,4> mda2(dims);
  mda2.AssignData(mda1);

  // +=, -=, *=, /= another array.
  CHECK_EQ(mda2(0,3,1,4), 100);
  mda2 += mda1;
  CHECK_EQ(mda2(0,3,1,4), 200);
  mda2 -= mda1;
  CHECK_EQ(mda2(0,3,1,4), 100);
  mda2 *= mda1;
  CHECK_EQ(mda2(0,3,1,4), 10000);
  mda1 += 1;
  mda2 /= mda1;
  CHECK_EQ(mda2(0,3,1,4), 99);

  // +=, -=, *=, /= scalar.
  mda2 += 100;
  CHECK_EQ(mda2(0,3,1,4), 199);
  mda2 -= 99;
  CHECK_EQ(mda2(0,3,1,4), 100);
  mda2 *= 3;
  CHECK_EQ(mda2(0,3,1,4), 300);
  mda2 /= 100;
  CHECK_EQ(mda2(0,3,1,4), 3);
  mda2.AssignData(10000);
  CHECK_EQ(mda2(0,3,1,4), 10000);

  // Deep copy.
  MultiDimArray<int,4> mda3 = mda2.DeepCopy();
  CHECK_EQ(mda3(0,3,1,4), 10000);
  mda3(0,3,1,4) = 100;
  CHECK_EQ(mda3(0,3,1,4), 100);
  CHECK_EQ(mda2(0,3,1,4), 10000);
}

int main()  {
  Timer timer;
  LOG(INFO) << "Test on ...";

  IndexTest();
  SliceTest();
  OperatorTest();

  LOG(INFO) << "Passed. [" << timer.elapsed() << " seconds]";
  return 0;
}
