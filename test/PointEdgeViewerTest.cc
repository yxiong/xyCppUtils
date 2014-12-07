/**
  * Test for PointEdgeViewer class.
  *
  * Author: Ying Xiong.
  * Created: Jun 13, 2013.
  */

#include "PointEdgeViewer.h"

#include <vector>

#include "CommandLineFlags.h"
#include "LogAndCheck.h"
#include "Timer.h"

using namespace std;
using namespace xyUtils;

DEFINE_FLAG_string(test_data_dir, "./TestData",
                   "The base directory for test data.")

int main(int argc, char** argv)  {
  Timer timer;
  LOG(INFO) << "Test on PointEdgeViewer ...";

  CommandLineFlagsInit(&argc, &argv, true);

  // Read points and colors.
  vector<double> points;
  vector<double> colors;
  FILE* fp = fopen((FLAGS_test_data_dir+"/Models/skull_nodes.txt").c_str(), "r");
  CHECK(fp);
  double p[3], c[3];
  while (fscanf(fp, "%lf %lf %lf %lf %lf %lf", p, p+1, p+2, c, c+1, c+2) > 0) {
    points.push_back(p[0]);
    points.push_back(p[1]);
    points.push_back(p[2]);
    colors.push_back(c[0]);
    colors.push_back(c[1]);
    colors.push_back(c[2]);
  }
  fclose(fp);

  // Read edges.
  vector<int> edges;
  fp = fopen((FLAGS_test_data_dir + "/Models/skull_edges.txt").c_str(), "r");
  CHECK(fp);
  int e[2];
  while(fscanf(fp, "%d %d", e, e+1) > 0) {
    edges.push_back(e[0]);
    edges.push_back(e[1]);
  }

  // View points and edges.
  PointEdgeViewer viewer;
  viewer.Launch();
  viewer.AddPoints(points.size()/3, points.data(), colors.data());
  viewer.AddEdges(edges.size()/2, edges.data());
  viewer.CenterView();
      
  LOG(INFO) << "Passed. [" << timer.elapsed() << " seconds]";
  return 0;
}
