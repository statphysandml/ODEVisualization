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

## Examples

The library includes several example systems:

- **Lorenz Attractor**: Classic chaotic system
- **Three-Point System**: Will follow
- **Four-Point System**: Will follow

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
  title = {ODE Visualization: GPU-Accelerated Analysis of Ordinary Differential Equations for the Functional Renormalization Group},
  author = {Kades, Lukas and Sadlo, Filip and Pawlowski, Jan M.},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/statphysandml/ODEVisualization},
  note = {Accessed: 2025-08-06}
}
```

