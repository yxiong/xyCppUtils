/**
  * Test for file I/O utilities.
  *
  * Author: Ying Xiong.
  * Created: Apr 30, 2013.
  */

#include "FileIO.h"

#include <cstdio>
#include <string>

#include "CommandLineFlags.h"
#include "LogAndCheck.h"
#include "Timer.h"

using namespace std;
using namespace xyUtils;

DEFINE_FLAG_string(test_data_dir, "./TestData",
                   "The base directory for test data.")

int main(int argc, char** argv)  {
  Timer timer;
  LOG(INFO) << "Test on FileIO ...";

  CommandLineFlagsInit(&argc, &argv, true);

  string filename = FLAGS_test_data_dir + "/Texts/FileIOTestFile.txt";
  // Read the whole file.
  std::string texts = FileIO::ReadWholeFileToString(filename.c_str());
  // Read the file line by line and check if get the same result.
  FILE* fp = fopen(filename.c_str(), "r");
  std::string line = FileIO::ReadLineToString(fp);
  size_t index = 0;
  while (!line.empty()) {
    size_t len = line.size();
    CHECK_EQ(line, texts.substr(index, len));
    index += len;
    line = FileIO::ReadLineToString(fp);    
  }
  CHECK_EQ(index, texts.size());
  fclose(fp);

  LOG(INFO) << "Passed. [" << timer.elapsed() << " seconds]";
  return 0;
}
