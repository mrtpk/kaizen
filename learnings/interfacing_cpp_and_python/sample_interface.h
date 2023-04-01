#ifndef SAMPLE_INTERFACE
#define SAMPLE_INTERFACE
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;
namespace bpy = boost::python;

class SampleInterface{
    public:
    SampleInterface();
    np::ndarray foo_bar(np::ndarray &input_array);
};
#endif // end of SAMPLE_INTERFACE