import os
from setuptools import setup
from torch.utils import cpp_extension

setup(name="sparse_conv2d",
      ext_modules=[cpp_extension.CppExtension("sparse_conv2d",
                                              [os.path.join("cpp_extension", "forward_backward.cpp")],
                                              extra_compile_args=["-std=c++14", "-fopenmp"])],
      cmdclass={"build_ext": cpp_extension.BuildExtension})
