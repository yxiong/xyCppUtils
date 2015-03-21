/**
 * Implementation for templated functions of 'MultiDimArray' class. This file is
 * currently included at the end of "MultiDimArray.h" file.
 *
 * Author: Ying Xiong.
 * Created: Jan 14, 2015.
 */

#ifndef __XYUTILS_MULTI_DIM_ARRAY_TCC__
#define __XYUTILS_MULTI_DIM_ARRAY_TCC__

#include <cmath>

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
MultiDimArray<T,N> MultiDimArray<T,N>::SliceDim(
    int dim, const Range<int>& range) const {
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

// ================================================================
// Find minimum/maximum elemnt.
// ================================================================
namespace multi_dim_array_details {

template <typename T, int N, template<typename,int,bool> class V, bool IgnoreNan>
struct MinMaxHelper {
  void operator()(const MultiDimArray<T,N>& mda, V<T,N,IgnoreNan>* v) {
    MinMaxHelper<T,N-1,V,IgnoreNan> nextHelper;
    for (int i = 0; i < mda.GetDim(0); ++i) {
      V<T,N-1,IgnoreNan> v2(v->value);
      nextHelper(mda.SliceDim(0, i), &v2);
      v->update(i, v2);
    }
  }
};

template <typename T, template<typename,int,bool> class V, bool IgnoreNan>
struct MinMaxHelper<T, 1, V, IgnoreNan> {
  void operator()(const MultiDimArray<T,1>& mda, V<T,1,IgnoreNan>* v) {
    for (int i = 0; i < mda.GetDim(0); ++i) {
      v->update(i, V<T,0,IgnoreNan>(mda(i)));
    }
  }
};

template <typename T, int N, bool IgnoreNan> struct MinV;
template <typename T, int N, bool IgnoreNan> struct MaxV;

// Ignore nan version.
template <typename T, int N>
struct MinV<T,N,true> {
  MinV(T v) : value(v) { }
  void update (int i, const MinV<T,N-1,true>& v2) {
    if ((std::isnan(value) && !std::isnan(v2.value)) || v2.value < value) {
      value = v2.value;
    }
  }
  T value;
};
template <typename T, int N>
struct MaxV<T,N,true> {
  MaxV(T v) : value(v) { }
  void update (int i, const MaxV<T,N-1,true>& v2) {
    if ((std::isnan(value) && !std::isnan(v2.value)) || v2.value > value) {
      value = v2.value;
    }
  }
  T value;
};

// Considering nan version.
template <typename T, int N>
struct MinV<T,N,false> {
  MinV(T v) : value(v) { }
  void update (int i, const MinV<T,N-1,false>& v2) {
    if (std::isnan(v2.value) || v2.value < value) {
      value = v2.value;
    }
  }
  T value;
};

template <typename T, int N>
struct MaxV<T,N,false> {
  MaxV(T v) : value(v) { }
  void update (int i, const MaxV<T,N-1,false>& v2) {
    if (std::isnan(v2.value) || v2.value > value) {
      value = v2.value;
    }
  }
  T value;
};

template <typename T, int N, bool IgnoreNan> struct ArgMinV;
template <typename T, int N, bool IgnoreNan> struct ArgMaxV;

// Ignore nan version.
template <typename T, int N>
struct ArgMinV<T,N,true> {
  ArgMinV(T v) : value(v) { }
  void update(int i, const ArgMinV<T,N-1,true>& v2) {
    if ((std::isnan(value) && !std::isnan(v2.value)) || v2.value < value) {
      value = v2.value;
      index[0] = i;
      for (int j = 1; j < N; ++j) { index[j] = v2.index[j-1]; }
    }
  }
  T value;
  std::array<int,N> index;
};
template <typename T, int N>
struct ArgMaxV<T,N,true> {
  ArgMaxV(T v) : value(v) { }
  void update(int i, const ArgMaxV<T,N-1,true>& v2) {
    if ((std::isnan(value) && !std::isnan(v2.value)) || v2.value > value) {
      value = v2.value;
      index[0] = i;
      for (int j = 1; j < N; ++j) { index[j] = v2.index[j-1]; }
    }
  }
  T value;
  std::array<int,N> index;
};

// Considering nan version.
template <typename T, int N>
struct ArgMinV<T,N,false> {
  ArgMinV(T v) : value(v) { }
  void update(int i, const ArgMinV<T,N-1,false>& v2) {
    if (std::isnan(value)) return;
    if (std::isnan(v2.value) || v2.value < value) {
      value = v2.value;
      index[0] = i;
      for (int j = 1; j < N; ++j) { index[j] = v2.index[j-1]; }
    }
  }
  T value;
  std::array<int,N> index;
};
template <typename T, int N>
struct ArgMaxV<T,N,false> {
  ArgMaxV(T v) : value(v) { }
  void update(int i, const ArgMaxV<T,N-1,false>& v2) {
    if (std::isnan(value)) return;
    if (std::isnan(v2.value) || v2.value > value) {
      value = v2.value;
      index[0] = i;
      for (int j = 1; j < N; ++j) { index[j] = v2.index[j-1]; }
    }
  }
  T value;
  std::array<int,N> index;
};

}   // namespace multi_dim_array_details

template <typename T, int N> template <bool IgnoreNan>
T MultiDimArray<T,N>::Min() const {
  using namespace multi_dim_array_details;
  MinV<T,N,IgnoreNan> v(data_[firstIndex_]);
  MinMaxHelper<T,N,MinV,IgnoreNan>()(*this, &v);
  return v.value;
}

template <typename T, int N> template <bool IgnoreNan>
T MultiDimArray<T,N>::Max() const {
  using namespace multi_dim_array_details;
  MaxV<T,N,IgnoreNan> v(data_[firstIndex_]);
  MinMaxHelper<T,N,MaxV,IgnoreNan>()(*this, &v);
  return v.value;
}

template <typename T, int N> template <bool IgnoreNan>
std::array<int,N> MultiDimArray<T,N>::ArgMin() const {
  using namespace multi_dim_array_details;
  ArgMinV<T,N,IgnoreNan> v(data_[firstIndex_]);
  for (int i = 0; i < N; ++i) {
    v.index[i] = 0;
  }
  MinMaxHelper<T,N,ArgMinV,IgnoreNan>()(*this, &v);
  return v.index;
}

template <typename T, int N> template <bool IgnoreNan>
std::array<int,N> MultiDimArray<T,N>::ArgMax() const {
  using namespace multi_dim_array_details;
  ArgMaxV<T,N,IgnoreNan> v(data_[firstIndex_]);
  for (int i = 0; i < N; ++i) {
    v.index[i] = 0;
  }
  MinMaxHelper<T,N,ArgMaxV,IgnoreNan>()(*this, &v);
  return v.index;
}

}   // namespace xyUtils
#endif   // __XYUTILS_MULTI_DIM_ARRAY_TCC__
