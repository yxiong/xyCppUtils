/**
  * Multi-dimensional array.
  *
  * Author: Ying Xiong.
  * Created: Jan 14, 2015.
  */

#ifndef __XYUTILS_MULTI_DIM_ARRAY_H__
#define __XYUTILS_MULTI_DIM_ARRAY_H__

#include <memory>

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
  MultiDimArray(const int* dims, const int* strides, T* data);
  // Get metadata.
  int GetDim(int n) const { return dims_[n]; }
  int GetStride(int n) const { return strides_[n]; }
  int GetNumElements() const { return numElements_; }
  // Note that although we defined accessing function for up to 6 indices, only
  // the one corresponds to size 'N' makes sense. Accessing a 2-dimensional
  // array with 3 indices will not generate compiler time error, but likely to
  // have segment fault.
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

 private:
  int dims_[N];
  int strides_[N];
  int numElements_;
  // Note that 'data_' field will always be valid, but 'sharedData_' could be
  // empty if the object does not take ownership of the data in memory.
  T* data_;
  std::shared_ptr<T> sharedData_;
  bool sharesOwnership_;
};
}   // namespace xyUtils

#include "MultiDimArray.tcc"

#endif   // __XYUTILS_MULTI_DIM_ARRAY_H__
