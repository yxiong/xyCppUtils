/**
  * Test for Image class.
  *
  * Author: Ying Xiong.
  * Created: May 13, 2013.
  * Updated: Nov 03, 2014.
  */

#include "Image.h"

#include <fstream>
#include <string>

#ifdef __USE_TR1__
#include <tr1/cstdint>
#include <tr1/type_traits>
#else
#include <cstdint>
#include <type_traits>
#endif

#if __USE_LIB_BOOST__ > 0
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#endif

#include "CommandLineFlags.h"
#include "LogAndCheck.h"
#include "Timer.h"

using namespace std;
using namespace xyUtils;

#ifdef __USE_TR1__
using namespace std::tr1;
#endif

DEFINE_FLAG_string(test_data_dir, "./TestData",
                   "The base directory for test data.")

#if __USE_LIB_JPEG__ > 0
template <typename T>
void JpegImageTestHelper(Image<T>* image) {
  string imgName = FLAGS_test_data_dir + "/Images/libjpeg-testorig.jpg";
  image->LoadMetaFromFile(imgName);
  CHECK_EQ(image->GetWidth(), 227);
  CHECK_EQ(image->GetHeight(), 149);
  CHECK_EQ(image->GetNumChannels(), 3);

  image->LoadFromFile(imgName);
  // Check dimensionality.
  CHECK_EQ(image->GetWidth(), 227);
  CHECK_EQ(image->GetHeight(), 149);
  CHECK_EQ(image->GetNumChannels(), 3);
  // Check Pixel() function.
  CHECK_EQ(image->Pixel(12, 34, 2), (PixelValueConvert<T, uint8_t>(54)));
  image->Pixel(12, 34, 1) = 25;
  CHECK_EQ(image->Pixel(12, 34, 1), 25);
  // Check operator ().
  CHECK_EQ((*image)(12, 34, 2), (PixelValueConvert<T, uint8_t>(54)));
  (*image)(12, 34, 1) = 20;
  CHECK_EQ(int((*image)(12, 34, 1)), 20);
}
#endif

#if __USE_LIB_PNG__ > 0
template <typename T>
void PngImageTestHelper(Image<T>* image) {
  vector<string> imgNames;
  imgNames.push_back(FLAGS_test_data_dir + "/Images/pngsuite/basn0g01.png");
  imgNames.push_back(FLAGS_test_data_dir + "/Images/pngsuite/basn0g02.png");
  imgNames.push_back(FLAGS_test_data_dir + "/Images/pngsuite/basn0g04.png");
  imgNames.push_back(FLAGS_test_data_dir + "/Images/pngsuite/basn0g08.png");
  imgNames.push_back(FLAGS_test_data_dir + "/Images/pngsuite/basn0g16.png");
  imgNames.push_back(FLAGS_test_data_dir + "/Images/pngsuite/basn2c08.png");
  imgNames.push_back(FLAGS_test_data_dir + "/Images/pngsuite/basn2c16.png");
  // Image size and number of channels.
  int imgWidth = 32, imgHeight = 32;
  int numChannels[] = {1, 1, 1, 1, 1, 3, 3};
  // Pixel value at particular positions.
  int x1 = 11, y1 = 14, c1 = 0;
  uint16_t pixelVals1[] = {65535, 85*257, 85*257, 51*257, 32512, 255*257, 42281};
  int x2 = 23, y2 = 21, c2 = 2;   // c2 will be 0 for single channel image.
  uint16_t pixelVals2[] = {0, 170*257, 170*257, 185*257, 63744, 255*257, 27482};
  // Do the check.
  for (size_t i = 0; i < imgNames.size(); ++i) {
    // Check image size by metadata.
    image->LoadMetaFromFile(imgNames[i]);
    CHECK_EQ(image->GetWidth(), imgWidth);
    CHECK_EQ(image->GetHeight(), imgHeight);
    CHECK_EQ(image->GetNumChannels(), numChannels[i]);
    // Check reading the image contents.
    image->LoadFromFile(imgNames[i]);
    CHECK_EQ(image->Pixel(x1, y1, c1),
             (PixelValueConvert<T, uint16_t>(pixelVals1[i])));
    CHECK_EQ(int(image->Pixel(x2, y2, std::min(c2, image->GetNumChannels()-1))),
             int(PixelValueConvert<T, uint16_t>(pixelVals2[i])));
  }
}
#endif

template <typename T>
void PpmImageTestHelper(Image<T>* image) {
  string imgName = FLAGS_test_data_dir + "/Images/libjpeg-testimg.ppm";
  image->Clear();
  CHECK_EQ(image->GetWidth(), -1);
  image->LoadMetaFromFile(imgName);
  CHECK_EQ(image->GetWidth(), 227);
  CHECK_EQ(image->GetHeight(), 149);
  CHECK_EQ(image->GetNumChannels(), 3);

  image->LoadFromFile(imgName);
  CHECK_EQ(image->GetWidth(), 227);
  CHECK_EQ(image->GetHeight(), 149);
  CHECK_EQ(image->GetNumChannels(), 3);
  CHECK_EQ(image->Pixel(12, 34, 2), (PixelValueConvert<T, uint8_t>(54)));
  CHECK_EQ(image->Pixel(200, 100, 1), (PixelValueConvert<T, uint8_t>(121)));
}

template <typename T>
void PixelIndexTestHelper() {
  int width = 10, height = 20, numChannels = 3;
  int x = 5, y = 12, c = 1;
  // Default PixelIndexTypeYXC.
  Image<T> image(width, height, numChannels);
  CHECK_EQ(image.PixelIndex(x,y,c), c + numChannels*(x + width * y));
  // Set to PixelIndexTypeXYC.
  image.SetPixelIndexType(Image<T>::PixelIndexTypeXYC);
  CHECK_EQ(image.PixelIndex(x,y,c), c + numChannels*(y + height * x));
  // Construct with PixelIndex_CXY.
  Image<T> image2(width, height, numChannels, Image<T>::PixelIndexTypeCXY);
  CHECK_EQ(image2.PixelIndex(x,y,c), y + height * (x + width * c));
  // Set to PixelIndexTypeCYX.
  image2.SetPixelIndexType(Image<T>::PixelIndexTypeCYX);
  CHECK_EQ(image2.PixelIndex(x,y,c), x + width * (y + height * c));
  // Resize the image.
  width = 15, height = 30, numChannels = 4;
  image2.SetSize(width, height, numChannels);
  CHECK_EQ(image2.PixelIndex(x,y,c), x + width * (y + height * c));
}

template <typename T>
void CheckImagesSame(const Image<T>& image1, const Image<T>& image2) {
  CHECK_EQ(image1.GetWidth(), image2.GetWidth());
  CHECK_EQ(image1.GetHeight(), image2.GetHeight());
  CHECK_EQ(image1.GetNumChannels(), image2.GetNumChannels());
  for (int x = 0; x < image1.GetWidth(); ++x) {
    for (int y = 0; y < image1.GetHeight(); ++y) {
      for (int c = 0; c < image1.GetNumChannels(); ++c) {
        CHECK_EQ(image1(x,y,c), image2(x,y,c));
      }
    }
  }
}

#if __USE_LIB_BOOST__ > 0
template <typename T>
void BoostSerializationTestHelper() {
  // Create a test image.
  int width = 10, height = 20, numChannels = 3;
  Image<T> image1(width, height, numChannels);
  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      for (int c = 0; c < numChannels; ++c) {
        image1(x,y,c) = x + y + c;
      }
    }
  }

  // Write the image to a text file.
  {
    ofstream ofs("tmp.ImageTest.txt");
    boost::archive::text_oarchive oa(ofs);
    oa << image1;
    // Archive and stream closed when destructors are called.
  }

  // Read the image from the text file.
  Image<T> image2;
  {
    ifstream ifs("tmp.ImageTest.txt");
    boost::archive::text_iarchive ia(ifs);
    ia >> image2;
    // Archive and stream closed when destructors are called.
  }

  // Check the two images are the same.
  CheckImagesSame(image1, image2);
  remove("tmp.ImageTest.txt");
}
#endif

template <typename T>
void BilinearInterpTestHelper(Image<T>* image) {
  int width = 10, height = 25;
  image->SetSize(width, height, 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        image->Pixel(x,y,c) = c*100 + x + y;
      }
    }
  }
  double x = 5.3, y = 6.4;
  CHECK_NEAR(image->BilinearInterp(x, y, 1), 111.7, 1e-4);
}

// A general helper to test on various type of 'Image<T>' class.
template <typename T>
void ImageTestHelperGeneral(Image<T>* image) {
#if __USE_LIB_JPEG__ > 0
  JpegImageTestHelper(image);
#endif
#if __USE_LIB_PNG__ > 0
  PngImageTestHelper(image);
#endif
  PpmImageTestHelper(image);
  PixelIndexTestHelper<T>();
#if __USE_LIB_BOOST__ > 0
  BoostSerializationTestHelper<T>();
#endif
  // Test for float type only.
  if ((is_same<T, float>::value || is_same<T, double>::value)) {
    BilinearInterpTestHelper(image);
  }
}

int main(int argc, char** argv)  {
  Timer timer;
  LOG(INFO) << "Test on Image class...";

  CommandLineFlagsInit(&argc, &argv, true);

  Image_8u image_8u;
  ImageTestHelperGeneral(&image_8u);
  Image_16u image_16u(12, 34);
  ImageTestHelperGeneral(&image_16u);
  Image_32f image_32f(12, 34, 3);
  ImageTestHelperGeneral(&image_32f);
  Image_64f image_64f(5, 6, 1, Image_64f::PixelIndexTypeCXY);
  ImageTestHelperGeneral(&image_64f);

  LOG(INFO) << "Passed. [" << timer.elapsed() << " seconds]";
  return 0;
}
