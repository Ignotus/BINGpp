/************************************************************************/
/* This source code is  free for both academic and industry use.         */
/* Some important information for better using the source code could be */
/* found in the project page: http://mmcheng.net/bing                   */
/************************************************************************/

#include "stdafx.h"
#include "Objectness.h"
#include "ValStructVec.h"
#include "CmShow.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>

#include "stdafx.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    if(argc < 2){
        std::cerr << "Please pass the data path to as first argument" << std::endl;
        return 1;
    }
    CStr dataPath = argv[1];

    DataSetVOC voc2007(dataPath);

    double base = 2;
    int W = 8;
    int NSS = 2;
    int numPerSz = 130;

    Objectness objNess(voc2007, base, W, NSS);
    objNess.loadTrainedModel("ObjNessB2W8MAXBGR");

    const int MAX_THREAD_NUM = omp_get_max_threads();
    initGPU(MAX_THREAD_NUM);

    Mat image;
    image = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    std::cout << "image readed" << std::endl;

    ValStructVec<float, Vec4i> boxes;
    objNess.getObjBndBoxes(image, boxes, numPerSz);
    boxes.sort();

    const std::vector<Vec4i>& bbs = boxes.getSortedStructVal();
    for (int i = 0; i < bbs.size(); ++i) {
        std::cout << bbs[i][0] << "," << bbs[i][1] << "," << bbs[i][2] << "," << bbs[i][3] << std::endl;
    }

    releaseGPU(MAX_THREAD_NUM);

    return 0;
}