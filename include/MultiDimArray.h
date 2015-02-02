/**
  * Multi-dimensional array.
  *
  * Author: Ying Xiong.
  * Created: Jan 14, 2015.
  */

#ifndef __XYUTILS_MULTI_DIM_ARRAY_H__
#define __XYUTILS_MULTI_DIM_ARRAY_H__

#include <memory>

#include "Range.h"

namespace xyUtils  {
// A multi-dimensional array class.
//
// In terms of memory management, there are two kinds of 'MultiDimArray': one
// shares ownership of the data and one does not. For the former kind, the data
// is managed with a 'shared_ptr', and will be released if the last managing
// instance goes out of scope.
//
// Note that we optimize for performance over security, and perform minimal
// sanity checks. This means errors like access invalid memory could occur if
// the input arguments are incorrect.
template <typename T, int N>
class MultiDimArray {
 public:
  // Create a multi-dimensional array of given dimensions and strides.
  // * If 'strides' is set to 'nullptr', the default 'C' style strides (where
  //   the last dimension runs fastest) is provided.
  // * An uninitialized chunk of data will be allocated, and the created object
  //   takes shared ownership of it.
  MultiDimArray(const int* dims, const int* strides=nullptr);
  // Create a multi-dimensional array with given data. The created object does
  // not take ownership of the data in memory.
  MultiDimArray(const int* dims, const int* strides, T* data, int firstIndex=0);
  // Create a multi-dimensional array that shares ownership of the data.
  MultiDimArray(const int* dims, const int* strides,
                const std::shared_ptr<T>& sharedData, int firstIndex=0);
  // Get metadata.
  int GetDim(int n) const { return dims_[n]; }
  int GetStride(int n) const { return strides_[n]; }
  int GetNumElements() const { return numElements_; }
  // Make a deep copy of the current object. The result object will takes
  // ownership of the newly allocated data, which will be stored compactly in
  // memory.
  MultiDimArray<T,N> DeepCopy();
  // Access data by multiple index. If the number of arguments does is not the
  // same as `N`, a compile-time error will be generated.
  T& operator()(int);
  T& operator()(int,int);
  T& operator()(int,int,int);
  T& operator()(int,int,int,int);
  T& operator()(int,int,int,int,int);
  T& operator()(int,int,int,int,int,int);
  const T& operator()(int) const;
  const T& operator()(int,int) const;
  const T& operator()(int,int,int) const;
  const T& operator()(int,int,int,int) const;
  const T& operator()(int,int,int,int,int) const;
  const T& operator()(int,int,int,int,int,int) const;
  // Slice a particular dimension of the array. The result array will be a
  // "view" of the original one, which means they share the same chunck of data
  // memory.
  // Note that the methods are "bitwise constness" rather than "logical
  // constness", in the sense that returned object will be able to modify the
  // data content of current object.
  MultiDimArray<T,N-1> SliceDim(int dim, int index) const;
  MultiDimArray<T,N> SliceDim(int dim, const Range& range) const;
  // Assign the data of current array from a given array.
  void AssignData(const MultiDimArray<T,N>& that);
  // Assign all the data to the same scalar.
  void AssignData(T scalar);
  // Element-wise operators.
  void operator+=(const MultiDimArray<T,N>& that);
  void operator+=(T scalar);
  void operator-=(const MultiDimArray<T,N>& that);
  void operator-=(T scalar);
  void operator*=(const MultiDimArray<T,N>& that);
  void operator*=(T scalar);
  void operator/=(const MultiDimArray<T,N>& that);
  void operator/=(T scalar);

 private:
  int dims_[N];
  int strides_[N];
  int numElements_;
  int firstIndex_;
  // Note that 'data_' field will always be valid, but 'sharedData_' could be
  // empty if the object does not take ownership of the data in memory.
  T* data_;
  std::shared_ptr<T> sharedData_;
  bool sharesOwnership_;
};
}   // namespace xyUtils

#include "MultiDimArray.tcc"

#endif   // __XYUTILS_MULTI_DIM_ARRAY_H__
