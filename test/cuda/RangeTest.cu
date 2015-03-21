/**
 * Test on cuda::Range class.
 *
 * Author: Ying Xiong.
 * Created: Mar 20, 2015.
 */

#include "cuda/Range.cuh"

#include "LogAndCheck.h"
#include "Range.h"
#include "Timer.h"

using namespace xyUtils;

__global__
void FillArrayByRange(const cuda::Range<float> r, float* dst) {
  for (int i = threadIdx.x; i < r.size(); i += blockDim.x) {
    dst[i] = r[i];
  }
}

int main() {
  Timer timer;
  LOG(INFO) << "Test on cuda::Range class...";

  Range<float> r(0.8f, -300.0001f, -0.1f);
  float* d_array;
  cudaMalloc(&d_array, r.size() * sizeof(float));
  int chunk = 64;

  FillArrayByRange<<<1,chunk>>>(r, d_array);

  float h_array[r.size()];
  cudaMemcpy(h_array, d_array, r.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < r.size(); ++i) {
    CHECK_NEAR(h_array[i], r[i], 0.0001f);
  }

  cudaFree(d_array);

  LOG(INFO) << "Passed. [" << timer.elapsed() << " seconds]";
  return 0;
}
