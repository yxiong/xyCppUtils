/**
 * Implementation for templated functions of 'MultiDimArray' class. This file is
 * currently included at the end of "MultiDimArray.h" file.
 *
 * Author: Ying Xiong.
 * Created: Jan 14, 2015.
 */

#ifndef __XYUTILS_MULTI_DIM_ARRAY_TCC__
#define __XYUTILS_MULTI_DIM_ARRAY_TCC__

namespace xyUtils {

template <typename T, int N>
MultiDimArray<T,N>::MultiDimArray(const int* dims, const int* strides) {
  numElements_ = 1;
  for (int i = N-1; i >= 0; --i) {
    dims_[i] = dims[i];
    strides_[i] = strides ? strides[i] : numElements_;
    numElements_ *= dims[i];
  }
  data_ = new T[numElements_];
  sharedData_ = std::shared_ptr<T>(data_, [](T* p) { delete[] p; });
  sharesOwnership_ = true;
}

template <typename T, int N>
MultiDimArray<T,N>::MultiDimArray(const int* dims, const int* strides,
                                  T* data) {
  numElements_ = 1;
  for (int i = N-1; i >= 0; --i) {
    dims_[i] = dims[i];
    strides_[i] = strides[i];
    numElements_ *= dims[i];
  }
  data_ = data;
  sharesOwnership_ = false;
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0) {
  return data_[i0*strides_[0]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1) {
  return data_[i0*strides_[0] + i1*strides_[1]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2) {
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
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3] + i4*strides_[4]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                  int i3, int i4, int i5) {
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3] + i4*strides_[4] + i5*strides_[5]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0) const {
  return data_[i0*strides_[0]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1) const {
  return data_[i0*strides_[0] + i1*strides_[1]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2) const {
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                        int i3) const {
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                        int i3, int i4) const {
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3] + i4*strides_[4]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                        int i3, int i4, int i5) const {
  return data_[i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3] + i4*strides_[4] + i5*strides_[5]];
}

}   // namespace xyUtils
#endif   // __XYUTILS_MULTI_DIM_ARRAY_TCC__
