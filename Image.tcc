/**
 * Implementation for templated functions of 'Image' class. This file is
 * currently included at the end of "Image.h" file.
 *
 * Author: Ying Xiong.
 * Created: Jun 25, 2013.
 */

#ifndef __XYUTILS_IMAGE_TCC__
#define __XYUTILS_IMAGE_TCC__

#include "xyUtilsConfig.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

#if __USE_LIB_JPEG__ > 0
#include "jpeglib.h"
#endif

#if __USE_LIB_PNG__ > 0
#include "png.h"
#endif

#include "LogAndCheck.h"

namespace xyUtils {

template<typename T>
void Image<T>::SetSize(int width, int height, int numChannels) {
  width_ = width;
  height_ = height;
  numChannels_ = numChannels;
  data_.resize(width * height * numChannels);
}

template<typename T> template<typename F>
T Image<T>::BilinearInterp(F x, F y, int c) const {
  int x0=int(x), x1=x0+1, y0=int(y), y1=y0+1;
  return (x-x0)*(y-y0)*Pixel(x1,y1,c) + (x-x0)*(y1-y)*Pixel(x1,y0,c) +
      (x1-x)*(y-y0)*Pixel(x0,y1,c) + (x1-x)*(y1-y)*Pixel(x0,y0,c);
}

template<typename T>
typename Image<T>::ImageType Image<T>::TypeFromFilename(
    const char* filename) {
  int n = strlen(filename);
#if __USE_LIB_JPEG__ > 0
  if (n > 4 && strcmp(filename+n-4, ".jpg") == 0) {
    return JpegType;
  } else
#endif
#if __USE_LIB_PNG__ > 0
  if (n > 4 && strcmp(filename+n-4, ".png") == 0) {
    return PngType;
  } else
#endif
  if (n > 4 && strcmp(filename+n-4, ".ppm") == 0) {
    return PpmType;
  } else {
    LOG(FATAL) << "Unknown or unsupported image type for file \""
               << filename << "\".";
    return UnknownType;
  }
}

template<typename T>
void Image<T>::LoadMetaFromFile(const char* filename, ImageType type) {
  if (type == UnknownType) {
    type = TypeFromFilename(filename);
  }
  switch (type) {
#if __USE_LIB_JPEG__
    case JpegType:
      LoadMetaFromJpegFile(filename); break;
#endif
#if __USE_LIB_PNG__
    case PngType:
      LoadMetaFromPngFile(filename); break;
#endif
    case PpmType:
      LoadMetaFromPpmFile(filename); break;
    default:
      LOG(FATAL) << "Internal error!";
  }
}

template<typename T>
void Image<T>::LoadFromFile(const char* filename, ImageType type) {
  if (type == UnknownType) {
    type = TypeFromFilename(filename);
  }
  switch (type) {
#if __USE_LIB_JPEG__
    case JpegType:
      LoadFromJpegFile(filename); break;
#endif
#if __USE_LIB_PNG__
    case PngType:
      LoadFromPngFile(filename); break;
#endif
    case PpmType:
      LoadFromPpmFile(filename); break;
    default:
      LOG(FATAL) << "Internal error!";
  }
}

#if __USE_LIB_JPEG__ > 0
// ================================================================
// Jpeg image interface.
// ================================================================
template<typename T>
void Image<T>::LoadMetaFromJpegFile(const char* filename) {
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  FILE* fp = fopen(filename, "rb");
  CHECK(fp);

  jpeg_stdio_src(&cinfo, fp);
  jpeg_read_header(&cinfo, TRUE);

  width_ = cinfo.image_width;
  height_ = cinfo.image_height;
  numChannels_ = cinfo.num_components;

  jpeg_destroy_decompress(&cinfo);
  fclose(fp);
}

template<typename T>
void Image<T>::LoadFromJpegFile(const char* filename) {
  // Set parameters.
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  FILE* fp = fopen(filename, "rb");
  CHECK(fp);

  jpeg_stdio_src(&cinfo, fp);
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  width_ = cinfo.output_width;
  height_ = cinfo.output_height;
  numChannels_ = cinfo.output_components;

  // Read the image.
  data_.resize(width_ * height_ * numChannels_);
  JSAMPROW row = new JSAMPLE[numChannels_ * width_];

  for (int y = 0; y < height_; ++y) {
    jpeg_read_scanlines(&cinfo, &row, 1);
      
    for (int x = 0; x < width_; ++x) {
      for (int c = 0; c < numChannels_; ++c) {
        Pixel(x,y,c) = PixelValueConvert<T, JSAMPLE>(row[numChannels_*x + c]);
      }
    }
  }

  // Clean up.
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  delete[] row;
  fclose(fp);
}

template<typename T>
void Image<T>::WriteToJpegFile(const char* filename, int quality) const {
  // Create an image buffer. This is not necessary if 'T' is the same as
  // 'JSAMPLE' (TODO).
  JSAMPLE* image_buffer = new JSAMPLE[numChannels_ * width_ * height_];
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      for (int c = 0; c < numChannels_; ++c) {
        image_buffer[c + numChannels_ * (x + width_ * y)] =
            PixelValueConvert<JSAMPLE, T>(Pixel(x,y,c));
      }
    }
  }

  // Setup parameters.
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  
  FILE* fp = fopen(filename, "wb");
  CHECK(fp);

  jpeg_stdio_dest(&cinfo, fp);

  cinfo.image_width = width_;
  cinfo.image_height = height_;
  cinfo.input_components = numChannels_;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);

  // Do the compression.
  jpeg_start_compress(&cinfo, TRUE);
  int row_stride = width_ * numChannels_;
  JSAMPROW row_pointer[1];
  while (int(cinfo.next_scanline) < height_) {
    row_pointer[0] = &image_buffer[cinfo.next_scanline * row_stride];
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  // Clean up.
  jpeg_finish_compress(&cinfo);
  fclose(fp);
  jpeg_destroy_compress(&cinfo);
  delete[] image_buffer;
}
#endif

#if __USE_LIB_PNG__ > 0
// ================================================================
// Png image interface.
// ================================================================
// Setup the 'png_struct' and 'png_info' structs and do 'png_init_io'.
inline bool SetupPngStructs(
    FILE* fp, png_structp* png_ptr, png_infop* info_ptr) {
  *png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,0,0,0);
  if (*png_ptr == NULL) {
    LOG(FATAL) << "Error: libpng version mismatch.";
    return false;
  }
  if (setjmp(png_jmpbuf(*png_ptr))) {
    LOG(FATAL) << "Error: fail to read png file.";
    return false;
  }

  png_init_io(*png_ptr, fp);
  *info_ptr = png_create_info_struct(*png_ptr);
  CHECK(*info_ptr);
  return true;
}

template<typename T>
void Image<T>::LoadMetaFromPngFile(const char* filename) {
  FILE* fp = fopen(filename, "rb");
  CHECK(fp);
  png_structp png_ptr = NULL;
  png_infop info_ptr = NULL;
  int success = SetupPngStructs(fp, &png_ptr, &info_ptr);
  png_read_info(png_ptr, info_ptr);
  CHECK(success);

  width_ = png_get_image_width(png_ptr, info_ptr);
  height_ = png_get_image_height(png_ptr, info_ptr);
  numChannels_ = png_get_channels(png_ptr, info_ptr);

  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
  fclose(fp);
}

template<typename T>
void Image<T>::ReadGrayPngDataSubByte(
    const png_bytep* png_data, int bit_depth,
    uint8_t mask0, int scale) {
  for (int y = 0; y < height_; ++y) {
    uint8_t mask = mask0;
    int idx = 0;
    int bit = 0;
    for (int x = 0; x < width_; ++x) {
      Pixel(x,y,0) = PixelValueConvert<T, uint8_t>(
          ((png_data[y][idx] & mask) >> bit) * scale);
      mask <<= bit_depth;
      bit += bit_depth;
      if (mask == 0) {
        mask = mask0;
        idx++;
        bit = 0;
      }
    }
  }
}

template<typename T>
void Image<T>::ReadGrayPngData(const png_bytep* png_data, int bit_depth) {
  if (bit_depth == 1) {
    ReadGrayPngDataSubByte(png_data, bit_depth, 1, 255);
  } else if (bit_depth == 2) {
    ReadGrayPngDataSubByte(png_data, bit_depth, 3, 85);
  } else if (bit_depth == 4) {
    ReadGrayPngDataSubByte(png_data, bit_depth, 15, 17);
  } else if (bit_depth == 8) {
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        Pixel(x,y,0) = PixelValueConvert<T, uint8_t>(png_data[y][x]);
      }
    }
  } else if (bit_depth == 16) {
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        Pixel(x,y,0) = PixelValueConvert<T, uint16_t>(
            uint16_t(png_data[y][2*x])*256 + png_data[y][2*x+1]);
      }
    }
  } else {
    LOG(FATAL) << "ReadGrayPngData Error: unknown bit_depth=" << bit_depth;
  }
}

template<typename T>
void Image<T>::ReadRGBPngData(const png_bytep* png_data, int bit_depth) {
  if (bit_depth == 8) {
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        for (int c = 0; c < numChannels_; ++c) {
          Pixel(x,y,c) = PixelValueConvert<T, uint8_t>(
              png_data[y][c+x*numChannels_]);
        }
      }
    }
  } else if (bit_depth == 16) {
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        for (int c = 0; c < numChannels_; ++c) {
          int idx = c + x*numChannels_;
          Pixel(x,y,c) = PixelValueConvert<T, uint16_t>(
              uint16_t(png_data[y][2*idx])*256 + png_data[y][2*idx+1]);
        }
      }
    }
  } else {
    LOG(FATAL) << "ReadRGBPngData Error: unknown bit_depth=" << bit_depth;
  }
}

template<typename T>
void Image<T>::LoadFromPngFile(const char* filename) {
  FILE* fp = fopen(filename, "rb");
  CHECK(fp);
  png_structp png_ptr = NULL;
  png_infop info_ptr = NULL;
  int success = SetupPngStructs(fp, &png_ptr, &info_ptr);
  CHECK(success);

  png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
  png_bytep* png_data = png_get_rows(png_ptr, info_ptr);

  width_ = png_get_image_width(png_ptr, info_ptr);
  height_ = png_get_image_height(png_ptr, info_ptr);
  numChannels_ = png_get_channels(png_ptr, info_ptr);
  int bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  int color_type = png_get_color_type(png_ptr, info_ptr);

  data_.resize(width_ * height_ * numChannels_);
  if (color_type == PNG_COLOR_TYPE_GRAY) {
    ReadGrayPngData(png_data, bit_depth);
  } else if (color_type == PNG_COLOR_TYPE_RGB) {
    ReadRGBPngData(png_data, bit_depth);
  } else {
    LOG(FATAL) << "Unhandled color type " << color_type;
  }

  png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
  fclose(fp);
}
#endif

// ================================================================
// Ppm image interface.
// ================================================================
// Forward the pointer in ppm file 'fp' to skip all the comments (between a '#'
// and a newline).
inline void SkipPpmComments(FILE* fp) {
  char c = getc(fp);
  while (c == '#') {
    while(getc(fp) != '\n') ;
    c = getc(fp);
  }
  ungetc(c, fp);
}

template<typename T>
void Image<T>::LoadMetaFromPpmFileHelper(FILE* fp) {
  // Read and check image format.
  char buf[16];
  if (!fgets(buf, sizeof(buf), fp)) {
    LOG(FATAL) << "Unable to read magic number.";
  }
  CHECK(buf[0] == 'P' && buf[1] == '6');
  // Read image size.
  SkipPpmComments(fp);
  if (fscanf(fp, "%d %d", &width_, &height_) != 2) {
    LOG(FATAL) << "Unable to read image size.";
  }
  // Check rgb component depth.
  SkipPpmComments(fp);
  int maxVal;
  if (fscanf(fp, "%d", &maxVal) != 1) {
    LOG(FATAL) << "Unable to read max value.";
  }
  CHECK_EQ(maxVal, 255);
  numChannels_ = 3;
}

template<typename T>
void Image<T>::LoadMetaFromPpmFile(const char* filename) {
  FILE* fp = fopen(filename, "rb");
  CHECK(fp);
  LoadMetaFromPpmFileHelper(fp);
  fclose(fp);
}

template<typename T>
void Image<T>::LoadFromPpmFile(const char* filename) {
  FILE* fp = fopen(filename, "rb");
  CHECK(fp);
  LoadMetaFromPpmFileHelper(fp);
  fgetc(fp); // A single white space character.
  data_.resize(width_ * height_ * numChannels_);
  unsigned char* tmp = new unsigned char[width_ * height_ * numChannels_];
  if (fread(tmp, 3*width_, height_, fp) != size_t(height_)) {
    LOG(FATAL) << "Unable to read the image";
  }
  // TODO: For T='uint8_t' and row-major layout, there exists a faster
  // implementation.
  for (int x = 0; x < width_; ++x) {
    for (int y = 0; y < height_; ++y) {
      for (int c = 0; c < numChannels_; ++c) {
        Pixel(x,y,c) = PixelValueConvert<T, uint8_t>(
            tmp[c + numChannels_ * (x + width_ * y)]);
      }
    }
  }
  delete[] tmp;
  fclose(fp);
}

template<typename T>
void Image<T>::SetPixelIndexFcn() {
  switch (pixelIndexType_) {
    case PixelIndexTypeYXC:
      pixelIndexFcn_ = &Image<T>::PixelIndexFcnYXC;
      break;
    case PixelIndexTypeXYC:
      pixelIndexFcn_ = &Image<T>::PixelIndexFcnXYC;
      break;
    case PixelIndexTypeCYX:
      pixelIndexFcn_ = &Image<T>::PixelIndexFcnCYX;
      break;
    case PixelIndexTypeCXY:
      pixelIndexFcn_ = &Image<T>::PixelIndexFcnCXY;
      break;
    default:
      LOG(FATAL) << "Internal error: unknown pixelIndexType_ "
                 << pixelIndexType_;
  }
}

template<>
inline uint8_t PixelValueConvert(uint8_t val)   { return val; }
template<>
inline uint16_t PixelValueConvert(uint8_t val)   { return uint16_t(val)*257; }
template<>
inline float PixelValueConvert(uint8_t val)   { return float(val)/255.0f; }
template<>
inline double PixelValueConvert(uint8_t val)   { return double(val)/255.0; }

template<>
inline uint8_t PixelValueConvert(uint16_t val)   { return uint8_t(val/257); }
template<>
inline uint16_t PixelValueConvert(uint16_t val)   { return val; }
template<>
inline float PixelValueConvert(uint16_t val)   { return float(val)/65535.0f; }
template<>
inline double PixelValueConvert(uint16_t val)   { return double(val)/65535.0; }

template<>
inline uint8_t PixelValueConvert(float val)   { return uint8_t(val*255); }
template<>
inline uint16_t PixelValueConvert(float val)   { return uint16_t(val*65535); }
template<>
inline float PixelValueConvert(float val)   { return val; }
template<>
inline double PixelValueConvert(float val)   { return double(val); }

template<>
inline uint8_t PixelValueConvert(double val)   { return uint8_t(val*255); }
template<>
inline uint16_t PixelValueConvert(double val)   { return uint16_t(val*65535); }
template<>
inline float PixelValueConvert(double val)   { return float(val); }
template<>
inline double PixelValueConvert(double val)   { return val; }

}   // namespace xyUtils

#endif   // __XYUTILS_IMAGE_TCC__
