/**
 * Test on cuda::MultiDimArray class.
 *
 * Author: Ying Xiong.
 * Created: Mar 20, 2015.
 */

#include "cuda/MultiDimArray.cuh"

#include "LogAndCheck.h"
#include "MultiDimArray.h"
#include "Timer.h"

using namespace xyUtils;

using MDA_32f3 = MultiDimArray<float,3>;
using CMDA_32f3 = cuda::MultiDimArray<float,3>;

__global__
void AddMultiDimArrays(const CMDA_32f3 a, const CMDA_32f3 b,
                       CMDA_32f3 r) {
  for (int x = threadIdx.x; x < a.GetDim(0); x += blockDim.x) {
    for (int y = threadIdx.y; y < a.GetDim(1); y += blockDim.y) {
      for (int z = threadIdx.z; z < a.GetDim(2); z += blockDim.z) {
        r(x,y,z) = a(x,y,z) + b(x,y,z);
      }
    }
  }
}

int main() {
  Timer timer;
  LOG(INFO) << "Test on cuda::MultiDimArray class...";

  int dims[3] = {30, 40, 20};
  MDA_32f3 a(dims);
  MDA_32f3 b(dims);
  a(10, 20, 4) = -100;
  b(20, 30, 5) = 200;

  CMDA_32f3 d_a = cuda::CopyMultiDimArrayToDevice(a);
  CMDA_32f3 d_b = cuda::CopyMultiDimArrayToDevice(b);
  CMDA_32f3 d_c = cuda::CreateMultiDimArrayOnDevice<float,3>(dims);

  dim3 blockSize(8,8,2);
  AddMultiDimArrays<<<1,blockSize>>>(d_a, d_b, d_c);

  MDA_32f3 c(dims);
  d_c.CopyToHost(&c);
  for (int i = 0; i < dims[0]; ++i) {
    for (int j = 0; j < dims[1]; ++j) {
      for (int k = 0; k < dims[2]; ++k) {
        CHECK_NEAR(c(i,j,k), a(i,j,k)+b(i,j,k), 0.001);
      }
    }
  }

  d_a.FreeOnDevice();
  d_b.FreeOnDevice();
  d_c.FreeOnDevice();

  LOG(INFO) << "Passed. [" << timer.elapsed() << " seconds]";
  return 0;
}
