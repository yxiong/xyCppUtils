/**
 * Implementation for templated functions of 'MultiDimArray' class. This file is
 * currently included at the end of "MultiDimArray.h" file.
 *
 * Author: Ying Xiong.
 * Created: Jan 14, 2015.
 */

#ifndef __XYUTILS_MULTI_DIM_ARRAY_TCC__
#define __XYUTILS_MULTI_DIM_ARRAY_TCC__

#include <stack>

namespace xyUtils {

// ================================================================
// Constructors.
// ================================================================

template <typename T, int N>
MultiDimArray<T,N>::MultiDimArray(const int* dims, const int* strides) {
  numElements_ = 1;
  for (int i = N-1; i >= 0; --i) {
    dims_[i] = dims[i];
    strides_[i] = strides ? strides[i] : numElements_;
    numElements_ *= dims[i];
  }
  firstIndex_ = 0;
  data_ = new T[numElements_];
  sharedData_ = std::shared_ptr<T>(data_, [](T* p) { delete[] p; });
  sharesOwnership_ = true;
}

template <typename T, int N>
MultiDimArray<T,N>::MultiDimArray(const int* dims, const int* strides,
                                  T* data, int firstIndex) {
  numElements_ = 1;
  for (int i = N-1; i >= 0; --i) {
    dims_[i] = dims[i];
    strides_[i] = strides[i];
    numElements_ *= dims[i];
  }
  firstIndex_ = firstIndex;
  data_ = data;
  sharesOwnership_ = false;
}

template <typename T, int N>
MultiDimArray<T,N>::MultiDimArray(const int* dims, const int* strides,
                                  const std::shared_ptr<T>& sharedData,
                                  int firstIndex) {
  numElements_ = 1;
  for (int i = N-1; i >= 0; --i) {
    dims_[i] = dims[i];
    strides_[i] = strides[i];
    numElements_ *= dims[i];
  }
  firstIndex_ = firstIndex;
  sharedData_ = sharedData;
  data_ = sharedData_.get();
  sharesOwnership_ = true;
}

// ================================================================
// Indexing.
// ================================================================

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0) {
  static_assert(N==1, "Invalid dimensionality.");
  return data_[firstIndex_ + i0*strides_[0]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1) {
  static_assert(N==2, "Invalid dimensionality.");
  return data_[firstIndex_ + i0*strides_[0] + i1*strides_[1]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2) {
  static_assert(N==3, "Invalid dimensionality.");
  return data_[firstIndex_ + i0*strides_[0] + i1*strides_[1] + i2*strides_[2]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                  int i3) {
  return data_[firstIndex_ + i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                  int i3, int i4) {
  static_assert(N==5, "Invalid dimensionality.");
  return data_[firstIndex_ + i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3] + i4*strides_[4]];
}

template <typename T, int N>
T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                  int i3, int i4, int i5) {
  static_assert(N==6, "Invalid dimensionality.");
  return data_[firstIndex_ + i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3] + i4*strides_[4] + i5*strides_[5]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0) const {
  static_assert(N==1, "Invalid dimensionality.");
  return data_[firstIndex_ + i0*strides_[0]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1) const {
  static_assert(N==2, "Invalid dimensionality.");
  return data_[firstIndex_ + i0*strides_[0] + i1*strides_[1]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2) const {
  static_assert(N==3, "Invalid dimensionality.");
  return data_[firstIndex_ + i0*strides_[0] + i1*strides_[1] + i2*strides_[2]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                        int i3) const {
  static_assert(N==4, "Invalid dimensionality.");
  return data_[firstIndex_ + i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                        int i3, int i4) const {
  static_assert(N==5, "Invalid dimensionality.");
  return data_[firstIndex_ + i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3] + i4*strides_[4]];
}

template <typename T, int N>
const T& MultiDimArray<T,N>::operator()(int i0, int i1, int i2,
                                        int i3, int i4, int i5) const {
  static_assert(N==6, "Invalid dimensionality.");
  return data_[firstIndex_ + i0*strides_[0] + i1*strides_[1] + i2*strides_[2] +
               i3*strides_[3] + i4*strides_[4] + i5*strides_[5]];
}

// ================================================================
// Slicing by dimension.
// ================================================================

template <typename T, int N>
MultiDimArray<T,N-1> MultiDimArray<T,N>::SliceDim(int dim, int index) const {
  int dims[N-1], strides[N-1];
  int i = 0;
  for (int j = 0; j < N; ++j) {
    if (j == dim)   continue;
    dims[i] = dims_[j];
    strides[i] = strides_[j];
    ++i;
  }
  int firstIndex = firstIndex_ + strides_[dim] * index;
  if (sharesOwnership_) {
    return MultiDimArray<T,N-1>(dims, strides, sharedData_, firstIndex);
  } else {
    return MultiDimArray<T,N-1>(dims, strides, data_, firstIndex);
  }
}

template <typename T, int N>
MultiDimArray<T,N> MultiDimArray<T,N>::SliceDim(int dim,
                                                const Range& range) const {
  int dims[N], strides[N];
  for (int i = 0; i < N; ++i) {
    if (i == dim) {
      dims[i] = range.size();
      strides[i] = strides_[i] * range.step();
    } else {
      dims[i] = dims_[i];
      strides[i] = strides_[i];
    }
  }
  int firstIndex = firstIndex_ + strides_[dim] * range.start();
  if (sharesOwnership_) {
    return MultiDimArray<T,N>(dims, strides, sharedData_, firstIndex);
  } else {
    return MultiDimArray<T,N>(dims, strides, data_, firstIndex);
  }
}

// ================================================================
// AssignData and operators.
// ================================================================

namespace multi_dim_array_details {

// We use template functor class here because C++ does not allow template
// partial instantiation for functions.
template <typename T, int N>
struct ApplyToAll {
  template <typename OP>
  void operator()(const int* dims, T* lData, int lIndex, const int* lStrides,
                  const T* rData, int rIndex, const int* rStrides, OP op) {
    ApplyToAll<T,N-1> nextHelper;
    for (int i = 0; i < dims[0]; ++i) {
      nextHelper(dims+1, lData, lIndex + lStrides[0]*i, lStrides+1,
                 rData, rIndex + rStrides[0]*i, rStrides+1, op);
    }
  }
};

template <typename T>
struct ApplyToAll<T,0> {
  template <typename OP>
  void operator()(const int* dims, T* lData, int lIndex, const int* lStrides,
                  const T* rData, int rIndex, const int* rStrides, OP op) {
    op(lData+lIndex, rData+rIndex);
  }
};

template <typename T, int N>
struct ApplyScalarToAll {
  template <typename OP>
  void operator()(const int* dims, T* lData, int lIndex, const int* lStrides,
                  T scalar, OP op) {
    ApplyScalarToAll<T,N-1> nextHelper;
    for (int i = 0; i < dims[0]; ++i) {
      nextHelper(dims+1, lData, lIndex+lStrides[0]*i, lStrides+1, scalar, op);
    }
  }
};

template <typename T>
struct ApplyScalarToAll<T,0> {
  template <typename OP>
  void operator()(const int* dims, T* lData, int lIndex, const int* lStrides,
                  T scalar, OP op) {
    op(lData+lIndex, scalar);
  }
};

}   // multi_dim_array_details

template <typename T, int N>
MultiDimArray<T,N> MultiDimArray<T,N>::DeepCopy() {
  MultiDimArray<T,N> result(dims_);
  result.AssignData(*this);
  return result;
}


template <typename T, int N>
void MultiDimArray<T,N>::AssignData(const MultiDimArray<T,N>& that) {
  multi_dim_array_details::ApplyToAll<T,N>()(
      dims_, data_, firstIndex_, strides_,
      that.data_, that.firstIndex_, that.strides_,
      [](T* l, const T* r) { *l =  *r; });
}

template <typename T, int N>
void MultiDimArray<T,N>::operator+=(const MultiDimArray<T,N>& that) {
  multi_dim_array_details::ApplyToAll<T,N>()(
      dims_, data_, firstIndex_, strides_,
      that.data_, that.firstIndex_, that.strides_,
      [](T* l, const T* r) { *l += *r; });
}

template <typename T, int N>
void MultiDimArray<T,N>::operator-=(const MultiDimArray<T,N>& that) {
  multi_dim_array_details::ApplyToAll<T,N>()(
      dims_, data_, firstIndex_, strides_,
      that.data_, that.firstIndex_, that.strides_,
      [](T* l, const T* r) { *l -= *r; });
}

template <typename T, int N>
void MultiDimArray<T,N>::operator*=(const MultiDimArray<T,N>& that) {
  multi_dim_array_details::ApplyToAll<T,N>()(
      dims_, data_, firstIndex_, strides_,
      that.data_, that.firstIndex_, that.strides_,
      [](T* l, const T* r) { *l *= *r; });
}

template <typename T, int N>
void MultiDimArray<T,N>::operator/=(const MultiDimArray<T,N>& that) {
  multi_dim_array_details::ApplyToAll<T,N>()(
      dims_, data_, firstIndex_, strides_,
      that.data_, that.firstIndex_, that.strides_,
      [](T* l, const T* r) { *l /= *r; });
}

template <typename T, int N>
void MultiDimArray<T,N>::AssignData(T scalar) {
  multi_dim_array_details::ApplyScalarToAll<T,N>()(
      dims_, data_, firstIndex_, strides_, scalar, [](T* l, T r) { *l = r; });
}

template <typename T, int N>
void MultiDimArray<T,N>::operator+=(T scalar) {
  multi_dim_array_details::ApplyScalarToAll<T,N>()(
      dims_, data_, firstIndex_, strides_, scalar, [](T* l, T r) { *l += r; });
}

template <typename T, int N>
void MultiDimArray<T,N>::operator-=(T scalar) {
  multi_dim_array_details::ApplyScalarToAll<T,N>()(
      dims_, data_, firstIndex_, strides_, scalar, [](T* l, T r) { *l -= r; });
}

template <typename T, int N>
void MultiDimArray<T,N>::operator*=(T scalar) {
  multi_dim_array_details::ApplyScalarToAll<T,N>()(
      dims_, data_, firstIndex_, strides_, scalar, [](T* l, T r) { *l *= r; });
}

template <typename T, int N>
void MultiDimArray<T,N>::operator/=(T scalar) {
  multi_dim_array_details::ApplyScalarToAll<T,N>()(
      dims_, data_, firstIndex_, strides_, scalar, [](T* l, T r) { *l /= r; });
}

}   // namespace xyUtils
#endif   // __XYUTILS_MULTI_DIM_ARRAY_TCC__
