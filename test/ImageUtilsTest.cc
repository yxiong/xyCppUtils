/**
 * Test for ImageUtils functions.
  *
  * Author: Ying Xiong.
  * Created: Oct 24, 2014.
  */

#include "ImageUtils.h"

#include <vector>
#include <Eigen/Dense>

#include "CommandLineFlags.h"
#include "Image.h"
#include "LogAndCheck.h"
#include "Timer.h"

using namespace std;
using namespace Eigen;
using namespace xyUtils;

DEFINE_FLAG_string(test_data_dir, "./TestData",
                   "The base directory for test data.")

int main(int argc, char** argv)  {
  Timer timer;
  LOG(INFO) << "Test on ...";

  CommandLineFlagsInit(&argc, &argv, true);

  // ================================================================
  // Test on a hole of 10x10.
  // ================================================================
  Image_32f image;
  image.LoadFromFile(FLAGS_test_data_dir + "/Images/libjpeg-testimg.ppm");
  // Create a hole.
  int x0 = 100, y0 = 50;
  for (int dx = 0; dx < 10; ++dx) {
    for (int dy = 0; dy < 10; ++dy) {
      for (int c = 0; c < 3; ++c) {
        image(x0+dx, y0+dy, c) = 0.0;
      }
    }
  }
  // Create a mask.
  Image_8u mask(image.GetWidth(), image.GetHeight());
  for (int x = 0; x < image.GetWidth(); ++x) {
    for (int y = 0; y < image.GetHeight(); ++y) {
      mask(x, y) = 1;
    }
  }
  for (int dx = 0; dx < 10; ++dx) {
    for (int dy = 0; dy < 10; ++dy) {
      mask(x0+dx, y0+dy) = 0;
    }
  }

  // Fill the holes.
  FillImageHoles(image, mask);

  CHECK_NEAR(image.Pixel(105, 55, 0), 0.952941, 0.0001);
  CHECK_NEAR(image.Pixel(105, 55, 1), 0.149020, 0.0001);
  CHECK_NEAR(image.Pixel(105, 55, 2), 0.219608, 0.0001);

  // ================================================================
  // Test on a single pixel with different fill directions.
  // ================================================================
  x0 = 200, y0 = 100;
  // Create the mask.
  for (int x = 0; x < image.GetWidth(); ++x) {
    for (int y = 0; y < image.GetHeight(); ++y) {
      mask(x, y) = 1;
    }
  }
  mask(x0, y0) = 0;

  FillImageHoles(image, mask);
  for (int c = 0; c < 3; ++c) {
    CHECK_NEAR(image.Pixel(x0, y0, c),
               (image.Pixel(x0-1, y0, c) + image.Pixel(x0+1, y0, c) +
                image.Pixel(x0, y0-1, c) + image.Pixel(x0, y0+1, c)) / 4.0,
               0.0001);
  }

  vector<Vector2i> fillDirections;
  vector<float> fillScales;
  fillDirections.push_back(Vector2i(+1,  0));  fillScales.push_back(2.0);
  fillDirections.push_back(Vector2i(-1,  0));  fillScales.push_back(2.0);
  fillDirections.push_back(Vector2i( 0, +1));  fillScales.push_back(2.0);
  fillDirections.push_back(Vector2i( 0, -1));  fillScales.push_back(2.0);
  fillDirections.push_back(Vector2i(-1, -1));  fillScales.push_back(1.0);
  fillDirections.push_back(Vector2i(-1, +1));  fillScales.push_back(1.0);
  fillDirections.push_back(Vector2i(+1, -1));  fillScales.push_back(1.0);
  fillDirections.push_back(Vector2i(+1, +1));  fillScales.push_back(1.0);

  FillImageHoles(image, mask, fillDirections, fillScales);
  for (int c = 0; c < 3; ++c) {
    CHECK_NEAR(image.Pixel(x0, y0, c),
               (2*image.Pixel(x0-1, y0, c) + 2*image.Pixel(x0+1, y0, c) +
                2*image.Pixel(x0, y0-1, c) + 2*image.Pixel(x0, y0+1, c) +
                image.Pixel(x0-1, y0-1, c) + image.Pixel(x0-1, y0+1, c) +
                image.Pixel(x0+1, y0-1, c) + image.Pixel(x0+1, y0+1, c)) / 12.0,
               0.0001);
  }

  LOG(INFO) << "Passed. [" << timer.elapsed() << " seconds]";
  return 0;
}
