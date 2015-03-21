/**
 * Multi-dimensional array for CUDA.
 *
 * Author: Ying Xiong.
 * Created: Mar 20, 2015.
 */

#ifndef __XYUTILS_CUDA_MULTI_DIM_ARRAY_H__
#define __XYUTILS_CUDA_MULTI_DIM_ARRAY_H__

#include "../MultiDimArray.h"

namespace xyUtils {
namespace cuda {

// This class has similar but more restricted functionalities as
// xyUtils::MultiDimArray, with member functions can be called on device.
//
// This class does not share ownership of the data, and the user needs to
// explicitly call `FreeOnDevice` to free the device memory.
template <typename T, int N>
class MultiDimArray {
 public:
  // Create a multi-dimensional array with given data.
  __host__ __device__
  MultiDimArray(const int* dims, const int* strides, T* data);
  // Get metadata.
  __host__ __device__ int GetDim(int n) const { return dims_[n]; }
  __host__ __device__ const int* GetDims() const { return dims_; }
  __host__ __device__ int GetStride(int n) const { return strides_[n]; }
  __host__ __device__ const int* GetStrides() const { return strides_; }
  __host__ __device__ int GetNumElements() const { return numElements_; }
  // Set metadata. Note that these methods will not re-allocate the data array.
  __host__ __device__ void SetDims(const int* dims);
  __host__ __device__ void SetStrides(const int* strides);
  // Access data by multiple index. If the number of arguments does is not the
  // same as `N`, a compile-time error will be generated.
  __host__ __device__ T& operator()(int);
  __host__ __device__ T& operator()(int,int);
  __host__ __device__ T& operator()(int,int,int);
  __host__ __device__ T& operator()(int,int,int,int);
  __host__ __device__ T& operator()(int,int,int,int,int);
  __host__ __device__ T& operator()(int,int,int,int,int,int);
  __host__ __device__ const T& operator()(int) const;
  __host__ __device__ const T& operator()(int,int) const;
  __host__ __device__ const T& operator()(int,int,int) const;
  __host__ __device__ const T& operator()(int,int,int,int) const;
  __host__ __device__ const T& operator()(int,int,int,int,int) const;
  __host__ __device__ const T& operator()(int,int,int,int,int,int) const;
  // Get data pointer.
  __host__ __device__ T* data() { return data_; }
  __host__ __device__ const T* data() const { return data_; }
  // Copy data to host. This function assumes `h_mda` has already allocated
  // enough memory to be filled.
  __host__ void CopyToHost(xyUtils::MultiDimArray<T,N>* h_mda) const;
  // Free the memory on device.
  __host__ void FreeOnDevice() { cudaFree(data_); }

 private:
  int dims_[N];
  int strides_[N];
  int numElements_;
  T* data_;
};

// Copy a xyUtils::MultiDimArray to xyUtils::cuda::MultiDimArray with data on
// device. This function assumes there is no gap when indexing `h_mda`.
template <typename T, int N> __host__
MultiDimArray<T,N> CopyMultiDimArrayToDevice(
    const xyUtils::MultiDimArray<T,N> h_mda);

// Create a xyUtils::cuda::MultiDimArray with data allocated on device. If
// `strides` is `nullptr`, the default 'C' style strides (where the last
// dimension runs fastest) is assumed.
template <typename T, int N> __host__
MultiDimArray<T,N> CreateMultiDimArrayOnDevice(
    const int* dims, const int* strides=nullptr);

}   // namespace cuda
}   // namespace xyUtils

#include "MultiDimArray.tcc"

#endif   // __XYUTILS_CUDA_MULTI_DIM_ARRAY_H__
