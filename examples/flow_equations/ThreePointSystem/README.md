# Three Point System

Implementation of the Three Point System providing the possibility to use all functionalities of the ODEVisualization library with the newly generated flow equation system.


# Prerequisites

Building Three Point System requires the following software installed:

* A C++17-compliant compiler
* CMake `>= 3.15`
* CUDA ('Tested with Version 10.1)
* Python `>= 3.6` for building Python bindings


# Building Three Point System

The following sequence of commands builds the C++/CUDA related part of Three Point System for a possible integration into a C++ library. The sequence assumes that your current working directory is the top-level directory
of the project:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

The build process can be customized with the following CMake variables,
which can be set by adding `-D<var>=...` to the `cmake` call:

* `CMAKE_INSTALL_PREFIX`: Local install path
* `BUILD_PYTHON_BINDINGS=ON`: Whether the python bindings are supposed to be built,

leading, for example, to the following command for a local installation:

```
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./../install ..
```

Similarly, the flow equations can be used in Python by executing:

```
pip install --use-feature=in-tree-build .
```

# Integration

For an integration into C++ or Python, see the examples of the ODEVisualization library.






