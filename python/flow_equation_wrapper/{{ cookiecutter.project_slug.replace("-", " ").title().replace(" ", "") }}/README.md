# {{ cookiecutter.project_name }}

Implementation of the {{ cookiecutter.project_name }} providing the possibility to use all functionalities of the ODEVisualization library with the newly generated flow equation system.

{# The white-space control of the below template is quite delicate - if you add one, do it exactly like this (mind the -'s) -#}
{%- set python_package = cookiecutter.project_slug.replace("-", "") -%}
{% if cookiecutter.license == "MIT" -%}
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
{% endif -%}
{% if cookiecutter.license == "BSD-2" -%}
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
{% endif -%}
{% if cookiecutter.license == "GPL-3.0" -%}
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
{% endif -%}
{% if cookiecutter.license == "LGPL-3.0" -%}
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
{% endif -%}
{{ "\n" -}}

# Prerequisites

Building {{ cookiecutter.project_name }} requires the following software installed:

* A C++17-compliant compiler
* CMake `>= 3.15`
* CUDA
* ODEVisualizationLib
{%- if cookiecutter.python_bindings == "Yes" -%}
* Python `>= 3.6` for building Python bindings
{% endif -%}

# Building {{ cookiecutter.project_name }}

The following sequence of commands builds the C++/CUDA related part of {{ cookiecutter.project_name }} for a possible integration into a C++ library. The sequence assumes that your current working directory is the top-level directory
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
* `CMAKE_PREFIX_PATH`: Installation path of the ODEVisualization library
{%- if cookiecutter.python_bindings == "Yes" -%}
* `BUILD_PYTHON_BINDINGS=ON`: Whether the python bindings are supposed to be built.
{% endif -%}

leading, for example, to the following command for a local installation (based on a local installation of the ODEVisualization library):

```
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./../install -DCMAKE_PREFIX_PATH=~/ODEVisualization/install -DBUILD_PYTHON_BINDINGS=ON ..
```

{%- if cookiecutter.python_bindings == "Yes" -%}
Similarly, the flow equations can be used in Python by executing:

```
pip install --use-feature=in-tree-build --install-option="--odevisualization-cmake-prefix-path='~/ODEVisualization/install/'" .
```

where the `odevisualization-cmake-prefix-path` is only required if the ODEVisualization library has been installed locally.
{% endif -%}

# Integration

For an integration into C++ or Python, see the examples of the ODEVisualization library.






