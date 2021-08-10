#include "sample_interface.h" 
#include <iostream> 
using namespace std;

SampleInterface::SampleInterface(){
    np::initialize();
}

np::ndarray SampleInterface::foo_bar(np::ndarray &input_array){
    return input_array;
}

BOOST_PYTHON_MODULE(cpp_interface) { // cpp_interface should be name of .so file
    using namespace boost::python;
    class_<SampleInterface>("SampleInterface")
        .def("foo", &SampleInterface::foo_bar)
        ;
    }