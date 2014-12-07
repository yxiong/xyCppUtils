/**
 * Test for Eigen library utilities.
 *
 * Author: Ying Xiong.
 * Created: Apr 21, 2014.
 */

#include "EigenUtils.h"

#include <string>

#include "CommandLineFlags.h"
#include "LogAndCheck.h"
#include "NumericalCheck.h"
#include "Timer.h"

using namespace std;
using namespace Eigen;
using namespace xyUtils;
using namespace xyUtils::EigenUtils;

DEFINE_FLAG_string(test_data_dir, "./TestData",
                   "The base directory for test data.")

int main(int argc, char** argv) {
  Timer timer;
  LOG(INFO) << "Test on EigenUtils ...";

  CommandLineFlagsInit(&argc, &argv, true);

  VectorXd v1(4);
  v1 << 1.0, 2.0, 3.0, 4.0;
  LOG(INFO) << FLAGS_test_data_dir + "/Texts/EigenUtilsTest_RowVector.txt";
  VectorXd v2 = VectorXdFromTextFile(FLAGS_test_data_dir +
                                     "/Texts/EigenUtilsTest_RowVector.txt");
  CheckNear(v1, v2, 1e-6);
  v2 = VectorXdFromTextFile(FLAGS_test_data_dir +
                            "/Texts/EigenUtilsTest_ColVector.txt");
  CheckNear(v1, v2, 1e-6);

  MatrixXd m1(2, 3);
  m1 << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
  MatrixXd m2 = MatrixXdFromTextFile(FLAGS_test_data_dir +
                                     "/Texts/EigenUtilsTest_Matrix.txt");
  CheckNear(m1, m2, 1e-6);

  LOG(INFO) << "Passed.";
  return 0;
}
