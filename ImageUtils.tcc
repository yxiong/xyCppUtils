/**
 * Implementation for templated functions of ImageUtils. This file is currently
 * included at the end of "ImageUtils.h" file.
 *
 * Author: Ying Xiong.
 * Created: Oct 24, 2014.
 */

#ifndef __XYUTILS_IMAGE_UTILS_TCC__
#define __XYUTILS_IMAGE_UTILS_TCC__

#include <set>
#include <vector>

#include "Eigen/Dense"

#include "Image.h"

namespace xyUtils  {

namespace ImageUtilsDetails {

// Comparator for 'Eigen::Vector2i'.
struct Vector2iComp {
  bool operator()(const Eigen::Vector2i& v1, const Eigen::Vector2i& v2) {
    return v1(0) < v2(0) || (v1(0) == v2(0) && v1(1) < v2(1));
  }
};

// A 'seed' is a to-be-filled pixel whose filling value can be determined in the
// next round, which means it has at least one neighbor pixel whose value is
// already known.
template<typename T>
struct Seed {
  Eigen::Vector2i xy;
  std::vector<T> val;
  // Constructor with only location.
  Seed (int x, int y, int numChannels) : xy(x,y), val(numChannels) { }
};

// Given a potential seed 's' with its location 'xy' set, check whether it can
// be filled by already-known pixels or not. If it can be filled, the 'val'
// field of 's' will be set properly, otherwise it will not be changed.
template<typename T, typename B>
bool IsSeed(const Image<T>& image,
            const Image<B>& mask,
            const std::vector<Eigen::Vector2i>& fillDirections,
            const std::vector<T>& fillScales,
            Seed<T>* s) {
  int n = 0;
  for (size_t i = 0; i < fillDirections.size(); ++i) {
    Eigen::Vector2i pixel = s->xy + fillDirections[i];
    if (pixel(0) < 0 || pixel(0) >= image.GetWidth() ||
        pixel(1) < 0 || pixel(1) >= image.GetHeight()) {
      continue;
    }
    if (mask(pixel(0), pixel(1))) {
      for (int c = 0; c < image.GetNumChannels(); ++c) {
        s->val[c] += image(pixel(0), pixel(1), c) * fillScales[i];
      }
      n += fillScales[i];
    }
  }
  if (n > 0) {
    for (int c = 0; c < image.GetNumChannels(); ++c) { s->val[c] /= n; }
    return true;
  } else {
    return false;
  }
}

}   // namespace ImageUtilsDetails

template <typename T, typename B>
void FillImageHoles(
    Image<T>& image, const Image<B>& _mask,
    const std::vector<Eigen::Vector2i>& fillDirections,
    const std::vector<T>& fillScales) {
  // Make a copy of '_mask' so that we can modify it.
  Image<B> mask = _mask;

  // Initialization: find all 'seeds' that has at least one non-masked pixel
  // around it, and add all other masked-out pixels to 'toBeFilled'.
  std::vector<ImageUtilsDetails::Seed<T> > seeds;
  std::set<Eigen::Vector2i, ImageUtilsDetails::Vector2iComp> toBeFilled;
  for (int x = 0; x < mask.GetWidth(); ++x) {
    for (int y = 0; y < mask.GetHeight(); ++y) {
      if (!mask(x,y)) {
        ImageUtilsDetails::Seed<T> s(x,y,image.GetNumChannels());
        if (ImageUtilsDetails::IsSeed(
                image, mask, fillDirections, fillScales, &s)) {
          seeds.push_back(s);
        } else {
          toBeFilled.insert(s.xy);
        }
      }
    }
  }

  // At each iteration, fill all the current seeds, and find new pixels to be
  // filled, which must be the neighbor of one of seeds in this iteration.
  while (!seeds.empty()) {
    // The seeds to be filled are called 'curSeeds', and 'seed' itself will hold
    // pixels to be filled at next iteration.
    std::vector<ImageUtilsDetails::Seed<T> > curSeeds;
    curSeeds.swap(seeds);
    for (const auto& seed : curSeeds) {
      // Put current seed into the image and update the mask.
      for (int c = 0; c < image.GetNumChannels(); ++c) {
        image(seed.xy(0), seed.xy(1), c) = seed.val[c];
      }
      mask(seed.xy(0), seed.xy(1)) = true;
      // Add current seed's neighbors to the next iteration of seeds.
      for (const auto& direction : fillDirections) {
        Eigen::Vector2i pixel = seed.xy - direction;
        int toFill = toBeFilled.erase(pixel);
        if (toFill > 0) {
          seeds.push_back(ImageUtilsDetails::Seed<T>(
              pixel(0), pixel(1), image.GetNumChannels()));
        }
      }
    }
    // Compute the value of seeds for next iteration.
    for (auto& seed : seeds) {
      ImageUtilsDetails::IsSeed(
          image, mask, fillDirections, fillScales, &seed);
    }
  }
}

template <typename T, typename B>
void FillImageHoles(Image<T>& image, const Image<B>& mask) {
  // Set default 'fillDirections' and 'fillScales'.
  std::vector<Eigen::Vector2i> fillDirections;
  std::vector<T> fillScales;

  fillDirections.push_back(Eigen::Vector2i(+1,0));    fillScales.push_back(1.0);
  fillDirections.push_back(Eigen::Vector2i(-1,0));    fillScales.push_back(1.0);
  fillDirections.push_back(Eigen::Vector2i(0,+1));    fillScales.push_back(1.0);
  fillDirections.push_back(Eigen::Vector2i(0,-1));    fillScales.push_back(1.0);

  FillImageHoles(image, mask, fillDirections, fillScales);
}
}   // namespace xyUtils

#endif   // __XYUTILS_IMAGE_UTILS_TCC__
