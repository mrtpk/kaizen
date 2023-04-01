---
layout: post
title:  "Interfacing CPP and Python using Python Boost"
date:   2021-08-10 21:19:00 +0530
categories: cpp python numpy
mathjax: true
---

Sometimes there will be situations where we need to pass data between CPP and Python codes. We can use Boost. Python to interface between CPP and Python. In this note, let's see how to call CPP functions from Python and transfer Numpy arrays between them.

### Test System Configuration
+ Ubuntu 20.04 LTS on WSL 1, Windows 10
+ Python 3.8
+ Boost 1.71 [To see Boost version, open up `/usr/include/boost/version.hpp` and look for `#define BOOST_LIB_VERSION`]

### Install dependencies, if needed
+ `sudo apt-get install libboost-all-dev`
+ `sudo apt install python-dev` or `sudo apt-get install python3-dev` depending upon the Python
+ `sudo apt install libboost-numpy-dev`

### Find out library names and their paths
We need the library names and their paths for linking.

+ Python - Find out where the `pyconfig` is [use `find /usr/include/ -name pyconfig*`]. In my system it was at `/usr/include/python3.8/`.
+ Boost Python - use `find /usr/lib -name libboost*`. In my system, boost_python library was at `/usr/lib/x86_64-linux-gnu/libboost_python38`. So the name of the library is `boost_python38`.
+ Boost Numpy - Just like Boost Python, I found out libboost_numpy was at `/usr/lib/x86_64-linux-gnu/libboost_numpy38`, and hence the library name was `boost_numpy38`

### Code
We need to create a shared library (`.so`) using Boost by compiling `cpp` code and linking existing libraries (`boost_python`, `python`, `boost_numpy`).

#### CPP header file - `sample_interface.h`

{% highlight cpp %}
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
{% endhighlight %}

#### CPP definition file - `sample_interface.cpp`

{% highlight cpp %}
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
{% endhighlight %}

#### Generating shared library (.so) file

`g++ sample_interface.cpp -I/usr/include/python3.8/ -lboost_python38  -lpython3.8 -lboost_numpy38 -fPIC --shared -o cpp_interface.so`

#### Python file - `calling_cpp_from_python.py`

{% highlight python %}
import numpy as np
import cpp_interface

arr = np.random.randint(1,10, size=(2, 3))
print("shape of input array:", arr.shape)
print("Input array:")
print(arr)

interface = cpp_interface.SampleInterface()
out_arr = interface.foo(arr)
print("shape of array from cpp:", out_arr.shape)
print("Output array:")
print(out_arr)
{% endhighlight %}

All of these snippets are available [here](https://github.com/mrtpk/kaizen/tree/master/snippets/interfacing_cpp_and_python).

**Acknowledgement**
+ [Sambhu Surya Mohan](https://www.linkedin.com/in/sambhu-surya-mohan-0147a02a/)

**Reference:**
+ [Interfacing C++ and Python with Boost.Python][Interfacing-cpp-and-python]

[Interfacing-cpp-and-python]: https://flanusse.net/interfacing-c++-with-python.html
