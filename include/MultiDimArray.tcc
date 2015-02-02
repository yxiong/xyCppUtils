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
template <typename T, int N, typename OP>
class ApplyToAll {
 public:
  void operator()(const int* dims, OP op,
                  T* lData, int lIndex, const int* lStrides,
                  const T* rData, int rIndex, const int* rStrides) {
    ApplyToAll<T,N-1,OP> nextHelper;
    for (int i = 0; i < dims[0]; ++i) {
      nextHelper(dims+1, op, lData, lIndex + lStrides[0]*i, lStrides+1,
                 rData, rIndex + rStrides[0]*i, rStrides+1);
    }
  }
};

template <typename T, typename OP>
class ApplyToAll<T,0,OP> {
 public:
  void operator()(const int* dims, OP(op),
                  T* lData, int lIndex, const int* lStrides,
                  const T* rData, int rIndex, const int* rStrides) {
    op(lData+lIndex, rData+rIndex);
  }
};

// We use functor class instead of inline function because the former is more
// likely to be inlined.
template <typename T>
class AssignmentOperator {
 public:
  void operator()(T* lData, const T* rData) {
    *lData = *rData;
  }
};

template <typename T>
class AddEqualOperator {
 public:
  void operator()(T* lData, const T* rData) {
    *lData += *rData;
  }
};

template <typename T>
class SubtractEqualOperator {
 public:
  void operator()(T* lData, const T* rData) {
    *lData -= *rData;
  }
};

template <typename T>
class MultiplyEqualOperator {
 public:
  void operator()(T* lData, const T* rData) {
    *lData *= *rData;
  }
};

template <typename T>
class DivideEqualOperator {
 public:
  void operator()(T* lData, const T* rData) {
    *lData /= *rData;
  }
};

template <typename T, int N, typename OP>
class ApplyScalarToAll {
 public:
  void operator()(const int* dims, OP op,
                  T* lData, int lIndex, const int* lStrides, T scalar) {
    ApplyScalarToAll<T,N-1,OP> nextHelper;
    for (int i = 0; i < dims[0]; ++i) {
      nextHelper(dims+1, op, lData, lIndex+lStrides[0]*i, lStrides+1, scalar);
    }
  }
};

template <typename T, typename OP>
class ApplyScalarToAll<T,0,OP> {
 public:
  void operator()(const int* dims, OP op,
                  T* lData, int lIndex, const int* lStrides, T scalar) {
    op(lData+lIndex, scalar);
  }
};

template <typename T>
class ScalarAssignOperator {
 public:
  void operator()(T* lData, T scalar) {
    *lData = scalar;
  }
};

template <typename T>
class ScalarAddEqualOperator {
 public:
  void operator()(T* lData, T scalar) {
    *lData += scalar;
  }
};

template <typename T>
class ScalarSubtractEqualOperator {
 public:
  void operator()(T* lData, T scalar) {
    *lData -= scalar;
  }
};

template <typename T>
class ScalarMultiplyEqualOperator {
 public:
  void operator()(T* lData, T scalar) {
    *lData *= scalar;
  }
};

template <typename T>
class ScalarDivideEqualOperator {
 public:
  void operator()(T* lData, T scalar) {
    *lData /= scalar;
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
  typedef multi_dim_array_details::AssignmentOperator<T> OP;
  multi_dim_array_details::ApplyToAll<T,N,OP>()(
      dims_, OP(), data_, firstIndex_, strides_,
      that.data_, that.firstIndex_, that.strides_);
}

template <typename T, int N>
void MultiDimArray<T,N>::operator+=(const MultiDimArray<T,N>& that) {
  typedef multi_dim_array_details::AddEqualOperator<T> OP;
  multi_dim_array_details::ApplyToAll<T,N,OP>()(
      dims_, OP(), data_, firstIndex_, strides_,
      that.data_, that.firstIndex_, that.strides_);
}

template <typename T, int N>
void MultiDimArray<T,N>::operator-=(const MultiDimArray<T,N>& that) {
  typedef multi_dim_array_details::SubtractEqualOperator<T> OP;
  multi_dim_array_details::ApplyToAll<T,N,OP>()(
      dims_, OP(), data_, firstIndex_, strides_,
      that.data_, that.firstIndex_, that.strides_);
}

template <typename T, int N>
void MultiDimArray<T,N>::operator*=(const MultiDimArray<T,N>& that) {
  typedef multi_dim_array_details::MultiplyEqualOperator<T> OP;
  multi_dim_array_details::ApplyToAll<T,N,OP>()(
      dims_, OP(), data_, firstIndex_, strides_,
      that.data_, that.firstIndex_, that.strides_);
}

template <typename T, int N>
void MultiDimArray<T,N>::operator/=(const MultiDimArray<T,N>& that) {
  typedef multi_dim_array_details::DivideEqualOperator<T> OP;
  multi_dim_array_details::ApplyToAll<T,N,OP>()(
      dims_, OP(), data_, firstIndex_, strides_,
      that.data_, that.firstIndex_, that.strides_);
}

template <typename T, int N>
void MultiDimArray<T,N>::AssignData(T scalar) {
  typedef multi_dim_array_details::ScalarAssignOperator<T> OP;
  multi_dim_array_details::ApplyScalarToAll<T,N,OP>()(
      dims_, OP(), data_, firstIndex_, strides_, scalar);
}

template <typename T, int N>
void MultiDimArray<T,N>::operator+=(T scalar) {
  typedef multi_dim_array_details::ScalarAddEqualOperator<T> OP;
  multi_dim_array_details::ApplyScalarToAll<T,N,OP>()(
      dims_, OP(), data_, firstIndex_, strides_, scalar);
}

template <typename T, int N>
void MultiDimArray<T,N>::operator-=(T scalar) {
  typedef multi_dim_array_details::ScalarSubtractEqualOperator<T> OP;
  multi_dim_array_details::ApplyScalarToAll<T,N,OP>()(
      dims_, OP(), data_, firstIndex_, strides_, scalar);
}

template <typename T, int N>
void MultiDimArray<T,N>::operator*=(T scalar) {
  typedef multi_dim_array_details::ScalarMultiplyEqualOperator<T> OP;
  multi_dim_array_details::ApplyScalarToAll<T,N,OP>()(
      dims_, OP(), data_, firstIndex_, strides_, scalar);
}

template <typename T, int N>
void MultiDimArray<T,N>::operator/=(T scalar) {
  typedef multi_dim_array_details::ScalarDivideEqualOperator<T> OP;
  multi_dim_array_details::ApplyScalarToAll<T,N,OP>()(
      dims_, OP(), data_, firstIndex_, strides_, scalar);
}

}   // namespace xyUtils
#endif   // __XYUTILS_MULTI_DIM_ARRAY_TCC__
