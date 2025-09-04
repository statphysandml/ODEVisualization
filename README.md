# ODEVisualization

A comprehensive Python library for visualizing and analyzing ordinary differential equations (ODEs) with C++/CUDA acceleration under the hood.

The library can also be used solely with C++/CUDA.

## Features

ODEVisualization is a high-performance C++/CUDA library with Python bindings that focuses on fast evaluation of high-dimensional sets of ordinary differential equations on GPUs.

The library can be used to:

- **Find and evaluate fixed points** (attractive and repulsive) in high-dimensional flow fields
- **Visualize two-dimensional views** of high-dimensional flow fields  
- **Integrate sets of ODEs** with GPU acceleration
- **Perform recursive grid computations** for detailed phase space analysis
- **Analyze flow equations and Jacobians** for dynamical systems

## Installation (Python)

The package consists of a main package called odevisualization. This package can be used for customized flow equation systems.

### Prerequisites

- Python 3.10 or higher
- CMake 3.18 or higher
- CUDA Toolkit (for GPU acceleration)
- C++17 compatible compiler

### Install from PyPI (Recommended)

Disclaimer: Uploading the package to PyPI is still work in progress, which is why for now the package has to be built from source.

```bash
pip install odevisualization
```

### Install from Source

```bash
git clone https://github.com/statphysandml/ODEVisualization.git
cd ODEVisualization/odevis
pip install .
```

## Quick Start

1. Auto-generating code: A good start is to rebuild the Lorentz Attractor flow equations. Alternatively, this step can be skipped and one can continue to directly installed the prepared LorentzAttractor example.

First take a look a examples/notebooks/lorentz_attractor/lorentz_attractor.nb. This script
generates the flow_equations.txt and the jacobian.txt files, which will be used by the
mathematica_parser to translate the set of equations into CUDA code. For this, run:

```bash
python odevis/python/gen_ode_system.py -o examples/custom_flow_equations/ -n "Lorentz Attractor" -fep examples/notebooks/lorentz_attractor/ -li 'MIT'
```

A LorentzAttractor directory has been generated and contains the resulting C++ / CUDA code
together with the required Python bindings to be able to make use of the flow equations solely in Python.

2. Installation: As a next step the flow equation specific Python library can be installed via:

```bash
cd examples/custom_flow_equations/LorentzAttractor
pip install -e .
```

or, alternatively,

```bash
cd examples/flow_equations/LorentzAttractor
pip install -e .
```

3. Usage: You should now be able to execute all Python examples in the examples/applications/python directory as long as the defined flow equation system refers to the one that has benn installed. The default one is LorentzAttractor.

For example, the following code is looking for fixed points based on a recursive search and a subsequent clustering of potential fixed points (required due to finite machine precision):

```python
from lorentz_attractor import LorentzAttractor

from odesolver.recursive_search import RecursiveSearch
from odesolver.kmeans_clustering import KMeansClustering
from odesolver.fixed_point_criterion import FixedPointCriterion


if __name__ == '__main__':
    lorentz_attractor = LorentzAttractor()

    fixed_point_criterion = FixedPointCriterion()

    recursive_fixed_point_search = RecursiveSearch(
        maximum_recursion_depth=18,
        n_branches_per_depth=[[10, 10, 10]] + [[2, 2, 2]] * 17,
        variable_ranges=[[-12.0, 12.0], [-12.0, 12.0], [-1.0, 31.0]],
        criterion=fixed_point_criterion,
        flow_equations=lorentz_attractor,
        number_of_cubes_per_gpu_call=20000,
        maximum_number_of_gpu_calls=1000
    )
    recursive_fixed_point_search.eval("dynamic")

    fixed_points = recursive_fixed_point_search.solutions()

    fixed_point_cube_index_path = recursive_fixed_point_search.solutions("cube_indices")

    print("Initial fixed points\n", fixed_points.transpose())

    kmeans_clustering = KMeansClustering(
        maximum_expected_number_of_clusters=10,
        upper_bound_for_min_distance=0.0005,
        maximum_number_of_iterations=1000
    )

    clustered_fixed_points = kmeans_clustering.eval(fixed_points)

    print("Clustered fixed points\n", clustered_fixed_points.transpose())
```

resulting in the following output:

```bash
Number of alive collections per depth:
 Depth 0: 1

        Remaining number of elements in depth 0: 1000 = 100%
Number of alive collections per depth:
 Depth 1: 10

Number of alive collections per depth:
 Depth 2: 10

Number of alive collections per depth:
 Depth 3: 11

Number of alive collections per depth:
 Depth 4: 9

Number of alive collections per depth:
 Depth 5: 13

Number of alive collections per depth:
 Depth 6: 14

Number of alive collections per depth:
 Depth 7: 12

Number of alive collections per depth:
 Depth 8: 13

Number of alive collections per depth:
 Depth 9: 14

Number of alive collections per depth:
 Depth 10: 14

Number of alive collections per depth:
 Depth 11: 13

Number of alive collections per depth:
 Depth 12: 12

Number of alive collections per depth:
 Depth 13: 13

Number of alive collections per depth:
 Depth 14: 14

Number of alive collections per depth:
 Depth 15: 14

Number of alive collections per depth:
 Depth 16: 13

Number of alive collections per depth:
 Depth 17: 12

Number of alive collections per depth:

Initial fixed points
 [[-9.15527344e-06 -9.15527344e-06 -1.22070313e-05]
 [-9.15527344e-06 -9.15527344e-06  1.22070313e-05]
 [ 9.15527344e-06  9.15527344e-06 -1.22070313e-05]
 [ 9.15527344e-06  2.74658203e-05 -1.22070313e-05]
 [ 9.15527344e-06  9.15527344e-06  1.22070313e-05]
 [ 9.15527344e-06  2.74658203e-05  1.22070313e-05]
 [-8.48528137e+00 -8.48528137e+00  2.69999878e+01]
 [-8.48529968e+00 -8.48528137e+00  2.70000122e+01]
 [-8.48528137e+00 -8.48528137e+00  2.70000122e+01]
 [ 8.48526306e+00  8.48528137e+00  2.69999878e+01]
 [ 8.48528137e+00  8.48528137e+00  2.69999878e+01]
 [ 8.48528137e+00  8.48529968e+00  2.69999878e+01]
 [ 8.48528137e+00  8.48528137e+00  2.70000122e+01]]
Averaged distance for k = 1: 15.9966
Averaged distance for k = 2: 10.9095
Averaged distance for k = 3: 1.73281e-05
Averaged distance for k = 4: 1.44165e-05
Averaged distance for k = 5: 1.25385e-05
Averaged distance for k = 6: 1.06604e-05
Averaged distance for k = 7: 8.82077e-06
Averaged distance for k = 8: 7.41227e-06
Averaged distance for k = 9: 7.04743e-06
Averaged distance for k = 10: 4.22551e-06
Adjacent differences: 15.9966 -5.08712 -10.9095 -2.91169e-06 -1.878e-06 -1.878e-06 -1.83968e-06 -1.4085e-06 -3.64832e-07 -2.82192e-06
Maximum diff k: 15.9966
Final averaged distance: 1.73281e-05 for k=3
Cluster center -8.48529 -8.48528 27
Cluster center 8.48528 8.48529 27
Cluster center 3.05176e-06 9.15527e-06 0
Clustered fixed points
 [[-8.48528748e+00 -8.48528137e+00  2.70000041e+01]
 [ 8.48527679e+00  8.48528595e+00  2.69999939e+01]
 [ 3.05175781e-06  9.15527344e-06  0.00000000e+00]]
```

## Example overview

The library includes several example systems:

- **Lorenz Attractor**: Classic chaotic system
- **Three-Point System**: Multi-body dynamics
- **Four-Point System**: Extended multi-body system

See the `examples/` directory for complete implementations.

## Documentation

Full documentation is available at: https://odevisualization.readthedocs.io. Note that this is subject to future work.

## Support and Development

For support, questions, or development discussions:
- Email: statphysandml@thphys.uni-heidelberg.de
- GitHub Issues: https://github.com/statphysandml/ODEVisualization/issues

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Citation

If you use ODEVisualization in your research, please cite:

```bibtex
@software{odevisualization,
  title={ODE Visualization: GPU-Accelerated Analysis of Ordinary Differential Equations for the Functional Renormalization Group},
  author={Kades, Lukas; Sadlo, Filip; Pawlowski, Jan M.}
  url={https://github.com/statphysandml/ODEVisualization},
  version={1.0.0},
  year={2025}
}
```

# TODOs
 
- Resolve warnings when pip install -e . --verbose for flow_equations
- Note that capital letters like X are not support as variable names within the ode system
- Fix all c++ examples...
- Clearly document how to use this with python and how to use it with c++ (in particular, how to build everything)