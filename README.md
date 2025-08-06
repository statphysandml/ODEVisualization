# ODEVisualization

A comprehensive Python library for visualizing and analyzing ordinary differential equations (ODEs) with CUDA acceleration.

## Features

ODEVisualization is a high-performance C++/CUDA library with Python bindings that focuses on fast evaluation of high-dimensional sets of ordinary differential equations on GPUs.

The library can be used to:

- **Find and evaluate fixed points** (attractive and repulsive) in high-dimensional flow fields
- **Visualize two-dimensional views** of high-dimensional flow fields  
- **Integrate sets of ODEs** with GPU acceleration
- **Perform recursive grid computations** for detailed phase space analysis
- **Analyze flow equations and Jacobians** for dynamical systems

## Installation

### Prerequisites

- Python 3.8 or higher
- CMake 3.18 or higher
- CUDA Toolkit (for GPU acceleration)
- C++17 compatible compiler

### Install from PyPI (Recommended)

```bash
pip install odevisualization
```

### Install from Source

```bash
git clone https://github.com/statphysandml/ODEVisualization.git
cd ODEVisualization/odevis
pip install .
```

#### CUDA Architecture Support

If you need to specify CUDA architectures:

```bash
pip install . --install-option="--cmake-cuda-architectures=70;75;80"
```

#### Build Options

**Use superbuild (automatic dependency management):**
```bash
pip install . --install-option="--use-superbuild"
```

**Use system dependencies:**
```bash
pip install . --install-option="--use-system-deps"
```

## Quick Start

```python
import odesolver as ode
import numpy as np

# Define your ODE system
# Example: Lorenz attractor
def lorenz_system(t, state, params):
    x, y, z = state
    sigma, rho, beta = params
    
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    
    return np.array([dxdt, dydt, dzdt])

# Set up initial conditions and parameters
initial_state = np.array([1.0, 1.0, 1.0])
params = np.array([10.0, 28.0, 8.0/3.0])

# Create and run simulation
# (Specific API details will depend on the actual implementation)
```

## Examples

The library includes several example systems:

- **Lorenz Attractor**: Classic chaotic system
- **Three-Point System**: Multi-body dynamics
- **Four-Point System**: Extended multi-body system

See the `examples/` directory for complete implementations.

## Development Installation

For development:

```bash
git clone https://github.com/statphysandml/ODEVisualization.git
cd ODEVisualization/odevis
pip install -e .[dev]
```

Run tests:
```bash
pytest
```

## Documentation

Full documentation is available at: https://odevisualization.readthedocs.io

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

