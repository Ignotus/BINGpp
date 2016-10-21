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
#include "conversion.h"

using namespace cv;
using namespace std;

namespace py = boost::python;

PyObject* std_vector_to_py_list(const std::vector<Vec4i>& v)
{
    npy_intp shape[2] = {v.size(), 4};

    std::vector<int> flat(v.size() * 4);
    for (int i = 0 ; i < v.size(); ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            flat[i * 4 + j] = v[i][j];
        }
    }

    PyObject* obj = PyArray_SimpleNewFromData(2, shape, NPY_INT, flat.data());
    return obj;
}

class BINGpp
{
    DataSetVOC voc2007;
    Objectness objNess;
public:
    BINGpp(const std::string& dataPath);
    ~BINGpp();

    PyObject* getObjBndBoxes(PyObject* image);
};

BINGpp::BINGpp(const std::string& modelPath)
    : voc2007("")
    , objNess(voc2007, 2, 8, 2)
{
    objNess.loadTrainedModel(modelPath + "/ObjNessB2W8MAXBGR");
    const int MAX_THREAD_NUM = omp_get_max_threads();
    initGPU(MAX_THREAD_NUM);
}

BINGpp::~BINGpp()
{
    const int MAX_THREAD_NUM = omp_get_max_threads();
    releaseGPU(MAX_THREAD_NUM);
}

PyObject* BINGpp::getObjBndBoxes(PyObject* image)
{
    cv::Mat mat = NDArrayConverter().toMat(image);
    ValStructVec<float, Vec4i> boxes;
    objNess.getObjBndBoxes(mat, boxes, 130);
    boxes.sort(true);
 
    return std_vector_to_py_list(boxes.getSortedStructVal());
}

BOOST_PYTHON_MODULE(bingpp)
{
    using namespace boost::python;

    // defined in numpy
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    import_array();

    class_<BINGpp>("BINGpp", init<std::string>())
        .def("getObjBndBoxes", &BINGpp::getObjBndBoxes);
}
