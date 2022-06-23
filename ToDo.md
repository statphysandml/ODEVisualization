- Correctly Integrate pybind11_json

- Speed check for new project_collection_package_on_expanded_cube_and_depth_per_cube_indices function and rename ending (also in ofther functions)
- Project cube index... <-> think about using an own function for this
- Write pybind11 wrapper for KMeans clustering
- Write pybind11 wrapper for Evolution <-> Integrate Boost and all its functionalities
- Reorder to have clear modes, etc. think about introducing core...
- Write code generator for PyTorch =)
- Integration with TorchDiffEq


Supported:

- Gitlab Actions CI:
- Gitlab CI:
    - ci
    - pypi?
    - sonarcloud?
- Read the docs
- Doxygen
- pybind11
- codecovio
- sonarcloud


Important commands:

Library::

Build the library:

mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./../install ..
make install -j9

Build the python binding package:

pip install --use-feature=in-tree-build .

Run tests:

ctest oder make test

Examples::

Build an example:

cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=~/ODEVisualization/install ..
cmake --build . -j9

Without python bindings:

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=~/ODEVisualization/install -DBUILD_PYTHON_BINDINGS=OFF ..

Options to generate a flow equation system:

- Use the cookiecutter to generate a flow equation system:
  
  cookiecutter python/flow_equation_wrapper -o ./examples/flow_equations/  

  in this case the a default project is generated for the flow equations of the Lorentz Attractor

- Using the gen_ode_system.py file from the command line
  
  python python/gen_ode_system.py -o "./examples/flow_equations/" -n "My Flow Equation System"

- Using the odesolver python package to call the same function from python
  
  from odesolver.gen_ode_system import generate_ode_system
  generate_ode_system(output_dir="./examples/flow_equations/", project_name="My Flow Equation System", flow_equation_path="None")
  
Build an ODESystem generated with the flow_equation wrapper

pip install --use-feature=in-tree-build --install-option="--odevisualization-cmake-prefix-path='~/ODEVisualization/install/'" .

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./../install -DCMAKE_PREFIX_PATH=~/ODEVisualization/install -DBUILD_PYTHON_BINDINGS=ON ..


python python/gen_ode_system.py -o "./examples/flow_equations/" -n "Four Point System" -fep "./examples/notebooks/four_point_system/"


cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="~/ODEVisualization/install;~/ODEVisualization/examples/flow_equations/LorentzAttractor/install" ..
