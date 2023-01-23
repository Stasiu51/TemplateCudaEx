import numpy as np
import TemplateCudaEx as m
print('imported')
m.helloworld()
print(m.fib(10))
a = np.array([1,2,3], dtype = float)
b = np.array([2,3,4], dtype = float)
print(m.vector_add(a,b))
print(m.vector_add_gpu(a,b))
print(m.vector_add_gpu(a,b))
