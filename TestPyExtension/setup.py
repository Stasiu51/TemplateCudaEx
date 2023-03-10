import os
from setuptools import setup, Extension
import numpy as np

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

if 'CUDA_PATH' in os.environ:
   CUDA_PATH = os.environ['CUDA_PATH']
else:
   print("Could not find CUDA_PATH in environment variables. Defaulting to /usr/local/cuda!")
   CUDA_PATH = "/usr/local/cuda"

if not os.path.isdir(CUDA_PATH):
   print("CUDA_PATH {} not found. Please update the CUDA_PATH variable and rerun".format(CUDA_PATH))
   exit(0)

setup(name = 'TemplateCudaEx', version = '1.0',  \
   ext_modules = [
      Extension('TemplateCudaEx', ['TemplateCudaEx.cpp'], 
      include_dirs=[np.get_include(), os.path.join(CUDA_PATH, "include")],
      libraries=[".\\CUDA\\build\\cuda_lib", "cudart"],
      library_dirs = [".",os.path.join(CUDA_PATH, "lib","x64")],
      extra_compile_args = ['-v']
    )],
)