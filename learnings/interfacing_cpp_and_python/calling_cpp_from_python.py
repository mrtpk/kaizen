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
