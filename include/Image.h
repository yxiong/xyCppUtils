/**
  * Image class.
  *
  * Author: Ying Xiong.
  * Created: May 13, 2013.
  * Updated: Nov 03, 2014.
  */

#ifndef __XYUTILS_IMAGE_H__
#define __XYUTILS_IMAGE_H__

#include "xyUtilsConfig.h"

#include <string>
#include <vector>

#ifdef __USE_TR1__
#include <tr1/cstdint>
#else
#include <cstdint>
#endif

#if __USE_LIB_PNG__ > 0
#include "png.h"
#endif

#if __USE_LIB_BOOST__ > 0
#include <boost/serialization/vector.hpp>
#endif

namespace xyUtils  {

// 'Image' is a class templated on the image data type 'T'. Conceptually, it is
// a 3D array of pixels 'T'. Internally, all data are stored in a vector<T>
// (which means the image owns the data), and the actual memory layout is
// determined by the 'PixelIndexType'.
template <typename T = unsigned char>
class Image {
 public:
  // ================================================================
  // Type definitions.
  // ================================================================
  // Image types.
  enum ImageType {UnknownType, JpegType, PngType, PpmType};
  // Pixel index type, which determines the memory layout of the data. The
  // naming convention 'PixelIndexTypePQR' indicates the index runs fastest on R
  // dimension, and slowest in P dimension.
  enum PixelIndexType {
    PixelIndexTypeYXC, // Default type, stores a matrix of pixel values
                       // (C-tuples) in row-major fashion.
    PixelIndexTypeXYC, // Stores a matrix of pixel values (C-tuples) in
                       // column-major fashion.
    PixelIndexTypeCYX, // Stores a set of row-major matrices, each color channel
                       // in one slice.
    PixelIndexTypeCXY  // Stores a set of column-major matrices, each color
                       // channel in one slice.
  };
  // ================================================================
  // Public interface.
  // ================================================================
  // Construct an invalid object.
  Image() : width_(-1), height_(-1), numChannels_(-1),
            data_(), pixelIndexType_(PixelIndexTypeYXC) {
    SetPixelIndexFcn();
  }
  // Construct an image with specified size.
  Image(int width, int height, int numChannels = 1,
        const PixelIndexType& pixelIndexType = PixelIndexTypeYXC)
      : width_(width), height_(height), numChannels_(numChannels),
        data_(width*height*numChannels), pixelIndexType_(pixelIndexType) {
    SetPixelIndexFcn();
  }
  // Get image width.
  int GetWidth() const {  return width_;  }
  // Get image height.
  int GetHeight() const { return height_;  }
  // Get number of color channels, usually either 1 or 3.
  int GetNumChannels() const {  return numChannels_; }
  // Specify the memory layout of the data. The c-th channel of (x,y)-th pixel
  // has index PixelIndex(x, y, c) of 'data()' vector.
  int PixelIndex(int x, int y, int c = 0) const {
    return (this->*pixelIndexFcn_)(x,y,c);
  }
  // Set image set. The image data after this call are unspecified.
  void SetSize(int width, int height, int numChannels = 1);
  // Get pixel (x, y) at channel c.
  const T& Pixel(int x, int y, int c = 0) const {
    return data_[PixelIndex(x,y,c)];
  }
  T& Pixel(int x, int y, int c = 0) { return data_[PixelIndex(x,y,c)]; }
  const T& operator()(int x, int y, int c = 0) const { return Pixel(x,y,c); }
  T& operator()(int x, int y, int c = 0) { return Pixel(x,y,c); }
  // Set 'PixelIndexType'. Note that the data will not be re-arranged after this
  // call.
  void SetPixelIndexType(const PixelIndexType& pixelIndexType) {
    pixelIndexType_ = pixelIndexType;
    SetPixelIndexFcn();
  }
  // Return the data pointer. Note that the actual memory layout of the data is
  // determined by the 'PixelIndex' function, which is in turn specified by
  // 'pixelIndexFcn_' field.
  const T* data() const { return data_.data(); }
  T* data() { return data_.data(); }
  // Clear the data field and reset the object to an invalid state.
  void Clear() {
    width_ = height_ = numChannels_ = -1;
    data_.clear();
  }
  // Clear the data field and reclaim most of the memory. The metadata will
  // still be valid.
  void ClearData() { data_.clear(); }
  // Bilinear interpolation. Note: current implementation requires 'x' and 'y'
  // lying inside the image, and 'T' being float or double type.
  template<typename F>
  T BilinearInterp(F x, F y, int c) const;
  // Load the meta information or the whole image from an input file.
  // If the 'type' is not specified, it will be determined by the file suffix.
  void LoadMetaFromFile(const char* filename, ImageType type = UnknownType);
  void LoadMetaFromFile(
      const std::string& filename, ImageType type = UnknownType) {
    LoadMetaFromFile(filename.c_str(), type);
  }
  void LoadFromFile(const char* filename, ImageType type = UnknownType);
  void LoadFromFile(const std::string& filename, ImageType type = UnknownType) {
    LoadFromFile(filename.c_str(), type);
  }
#if __USE_LIB_JPEG__ > 0
  // Load the meta information or the whole image from a Jpeg file.
  // Note: currently can only read 8-bit jpeg images. (TODO)
  void LoadMetaFromJpegFile(const char* filename);
  void LoadMetaFromJpegFile(const std::string& filename) {
    LoadMetaFromJpegFile(filename.c_str());
  }
  void LoadFromJpegFile(const char* filename);
  void LoadFromJpegFile(const std::string& filename) {
    LoadFromJpegFile(filename.c_str());
  }
  // Write the image to a jpg file. The 'quality' parameter should be a number
  // between 0 and 100, with 0 being lowest quality and 100 highest. We suggest
  // use 80 for high quality output, 50 for medium and 30 for low.
  // Note: currently can only write 8-bit jpeg images. (TODO)
  void WriteToJpegFile(const char* filename, int quality) const;
  void WriteToJpegFile(const std::string& filename, int quality) const {
    WriteToJpegFile(filename.c_str(), quality);
  }
#endif

#if __USE_LIB_PNG__ > 0
  // Load the meta information or the whole image from a png file.
  void LoadMetaFromPngFile(const char* filename);
  void LoadMetaFromPngFile(const std::string& filename) {
    LoadMetaFromPngFile(filename.c_str());
  }
  void LoadFromPngFile(const char* filename);
  void LoadFromPngFile(const std::string& filename) {
    LoadFromPngFile(filename.c_str());
  }
#endif
  // Load the meta information or the whole image from a ppm file.
  void LoadMetaFromPpmFile(const char* filename);
  void LoadMetaFromPpmFile(const std::string& filename) {
    LoadMetaFromPpmFile(filename.c_str());
  }
  void LoadFromPpmFile(const char* filename);
  void LoadFromPpmFile(const std::string& filename) {
    LoadFromPpmFile(filename.c_str());
  }
  // Determine the image type from filename.
  static ImageType TypeFromFilename(const char* filename);
  static ImageType TypeFromFilename(const std::string& filename) {
    return TypeFromFilename(filename.c_str());
  }

#if __USE_LIB_BOOST__ > 0
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    (void) version;
    ar & width_;
    ar & height_;
    ar & numChannels_;
    ar & data_;
    ar & pixelIndexType_;
  }
#endif

 private:
  // ================================================================
  // Helper functions.
  // ================================================================
#if __USE_LIB_PNG__ > 0
  void ReadGrayPngDataSubByte(const png_bytep* png_data, int bit_depth,
                              uint8_t mask0, int scale);
  void ReadGrayPngData(const png_bytep* png_data, int bit_depth);
  void ReadRGBPngData(const png_bytep* png_data, int bit_depth);
#endif
  void LoadMetaFromPpmFileHelper(FILE* fp);
  // ================================================================
  // Pixel index functions.
  // ================================================================
  typedef int (Image<T>::*PixelIndexFcn)(int,int,int) const;
  int PixelIndexFcnYXC(int x, int y, int c) const {
    return c + numChannels_ * (x + width_ * y);
  }
  int PixelIndexFcnXYC(int x, int y, int c) const {
    return c + numChannels_ * (y + height_ * x);
  }
  int PixelIndexFcnCYX(int x, int y, int c) const {
    return x + width_ * (y + height_ * c);
  }
  int PixelIndexFcnCXY(int x, int y, int c) const {
    return y + height_ * (x + width_ * c);
  }
  // Set the 'pixelIndexFcn_' field according to 'pixelIndexType_'.
  void SetPixelIndexFcn();
  // ================================================================
  // Data fields.
  // ================================================================
  int width_, height_, numChannels_;
  std::vector<T> data_;
  PixelIndexType pixelIndexType_;
  PixelIndexFcn pixelIndexFcn_;
};

typedef Image<uint8_t> Image_8u;
typedef Image<uint16_t> Image_16u;
typedef Image<float> Image_32f;
typedef Image<double> Image_64f;

// Convert pixel value from data type to another: the value range for different
// types are:
//   uint8_t:    255
//   uint16_t: 65535
//   float:      1.0
//   double:     1.0
template<typename DST_TYPE, typename SRC_TYPE>
inline DST_TYPE PixelValueConvert(SRC_TYPE val);

}   // namespace xyUtils

#include "Image.tcc"

#endif   // __XYUTILS_IMAGE_H__
