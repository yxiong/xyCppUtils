/**
 * Implementation for templated functions of cuda::MultiDimArray' class. This
 * file is currently included at the end of "MultiDimArray.cuh" file.
 *
 * Author: Ying Xiong.
 * Created: Mar 20, 2015.
 */

#ifndef __XYUTILS_CUDA_MULTI_DIM_ARRAY_TCC__
#define __XYUTILS_CUDA_MULTI_DIM_ARRAY_TCC__

namespace xyUtils {
namespace cuda {

// ================================================================
// Constructors.
// ================================================================

template <typename T, int N>
MultiDimArray<T,N>::MultiDimArray(const int* dims, const int* strides,
                                  T* data) {
  numElements_ = 1;
  for (int i = 0; i < N; ++i) {
    dims_[i] = dims[i];
    strides_[i] = strides[i];
    numElements_ *= dims[i];
  }
  data_ = data;
}

template <typename T, int N>
void MultiDimArray<T,N>::SetDims(const int* dims) {
  numElements_ = 1;
  for (int i = 0; i < N; ++i) {
    dims_[i] = dims[i];
    numElements_ *= dims[i];
  }
}

template <typename T, int N>
void MultiDimArray<T,N>::SetStrides(const int* strides) {
  for (int i = 0; i < N; ++i) {
    strides_[i] = strides[i];
  }
}

// ================================================================
// Indexing.
// ================================================================

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0) {
  static_assert(N==1, "Invalid dimensionality.");
  return data_[i0*strides_[0]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1) {
  static_assert(N==2, "Invalid dimensionality.");
  return data_[i0*strides_[0] + i1*strides_[1]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2) {
  static_assert(N==3, "Invalid dimensionality.");
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                  int i3) {
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                  int i3, int i4) {
  static_assert(N==5, "Invalid dimensionality.");
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3] + i4*strides_[4]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                  int i3, int i4, int i5) {
  static_assert(N==6, "Invalid dimensionality.");
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3] + i4*strides_[4] + i5*strides_[5]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0) const {
  static_assert(N==1, "Invalid dimensionality.");
  return data_[i0*strides_[0]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1) const {
  static_assert(N==2, "Invalid dimensionality.");
  return data_[i0*strides_[0] + i1*strides_[1]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2) const {
  static_assert(N==3, "Invalid dimensionality.");
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                        int i3) const {
  static_assert(N==4, "Invalid dimensionality.");
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                        int i3, int i4) const {
  static_assert(N==5, "Invalid dimensionality.");
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3] + i4*strides_[4]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                        int i3, int i4, int i5) const {
  static_assert(N==6, "Invalid dimensionality.");
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3] + i4*strides_[4] + i5*strides_[5]];
}

// ================================================================
// Host and device memory management.
// ================================================================

template <typename T, int N>
void MultiDimArray<T,N>::CopyToHost(xyUtils::MultiDimArray<T,N>* h_mda) const {
  h_mda->SetDims(dims_);
  h_mda->SetStrides(strides_);
  cudaMemcpy(h_mda->data(), data_, numElements_ * sizeof(T),
             cudaMemcpyDeviceToHost);
}

template <typename T, int N>
MultiDimArray<T,N> CopyMultiDimArrayToDevice(
    const xyUtils::MultiDimArray<T,N> h_mda) {
  T* d_data;
  cudaMalloc(&d_data, h_mda.GetNumElements() * sizeof(T));
  cudaMemcpy(d_data, h_mda.data(), h_mda.GetNumElements() * sizeof(T),
             cudaMemcpyHostToDevice);
  return MultiDimArray<T,N>(h_mda.GetDims(), h_mda.GetStrides(), d_data);
}

template <typename T, int N>
MultiDimArray<T,N> CreateMultiDimArrayOnDevice(
    const int* dims, const int* strides) {
  int strides2[N];
  T* d_data;
  int numElements = 1;
  for (int i = N-1; i >= 0; --i) {
    strides2[i] = strides ? strides[i] : numElements;
    numElements *= dims[i];
  }
  cudaMalloc(&d_data, numElements * sizeof(T));
  return MultiDimArray<T,N>(dims, strides2, d_data);
}

}   // namespace cuda
}   // namespace xyUtils
#endif   // __XYUTILS_CUDA_MULTI_DIM_ARRAY_TCC__
