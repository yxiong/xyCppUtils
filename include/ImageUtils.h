/**
  * Utility functions for Image.
  *
  * Author: Ying Xiong.
  * Created: Oct 24, 2014.
  */

#ifndef __XYUTILS_IMAGE_UTILS_H__
#define __XYUTILS_IMAGE_UTILS_H__

#include <vector>

#include "Eigen/Dense"

#include "Image.h"

namespace xyUtils  {

// Fill the holes in an image. The input 'image' will be modified in place, and
// all pixels (x,y) such that mask(x,y) is false will be replaced iteratively
// with some combination of other pixels from the image.
//
// The combination is determined by 'fillDirections' and 'fillScales'. More
// specifically, the masked-out pixel (x,y) will be replaced by the weighted
// average of unmasked pixels (x+fillDirections[i](1), y+fillDirections[i](2)),
// and the weights are based fillScales[i].
//
// Filling example 1: fill with 4-neightbor.
//   fillDirections = { [1,0], [-1,0], [0,1], [0,-1] }
//   fillScales = { 1, 1, 1, 1}.
//
// Filling example 2: fill with 8-neighbor with higher weights for the
// 4-neighbor.
//   fillDirections = { [1,0], [-1,0], [0,1], [0,-1],
//                      [1,1], [1,-1], [-1,1], [-1,-1]}
//   fillScales = { 2, 2, 2, 2, 1, 1, 1, 1}.
//
// NOTE: so far we do not explicilty handle overflow during weighted average,
// and therefore the type 'T' is suggested to be 'float' or 'double'.
template <typename T, typename B>
void FillImageHoles(
    Image<T>& image, const Image<B>& mask,
    const std::vector<Eigen::Vector2i>& fillDirections,
    const std::vector<T>& fillScales);
// Default behavior: fill with 4-neighbor.
template <typename T, typename B>
void FillImageHoles(Image<T>& image, const Image<B>& mask);

}   // namespace xyUtils

#include "ImageUtils.tcc"

#endif   // __XYUTILS_IMAGE_UTILS_H__
