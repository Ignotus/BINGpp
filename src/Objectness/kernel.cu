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

#include <boost/python.hpp>

#include "stdafx.h"

using namespace cv;
using namespace std;

namespace py = boost::python;

py::list std_vector_to_py_list(const std::vector<Vec4i>& v)
{
    py::object get_iter = py::iterator<std::vector<Vec4i> >();
    py::object iter = get_iter(v);
    py::list l(iter);
    return l;
}

class BINGpp
{
    DataSetVOC voc2007;
    Objectness objNess;
public:
    BINGpp(const std::string& dataPath);
    ~BINGpp();

    py::list getObjBndBoxes(const Mat& image);
};

BINGpp::BINGpp(const std::string& dataPath)
    : voc2007(dataPath)
    , objNess(voc2007, 2, 8, 2)
{
    objNess.loadTrainedModel("ObjNessB2W8MAXBGR");
    const int MAX_THREAD_NUM = omp_get_max_threads();
    initGPU(MAX_THREAD_NUM);
}

BING::~BINGpp()
{
    const int MAX_THREAD_NUM = omp_get_max_threads();
    releaseGPU(MAX_THREAD_NUM);
}

py::list BINGpp::getObjBndBoxes(const Mat& image)
{
    ValStructVec<float, Vec4i> boxes;
    objNess.getObjBndBoxes(image, boxes, 130);
    boxes.sort();
    return boxes;
}

BOOST_PYTHON_MODULE(bingpp)
{
    using namespace boost::python;
    class_<BINGpp>("BINGpp", init<std::string>())
        .def("getObjBndBoxes", &BINGpp::getObjBndBoxes)
}
