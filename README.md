# JAX Moving MNIST Project

## Description
This project implements a Moving MNIST dataset generator and processing pipeline using JAX. The code creates sequences of moving digits for video prediction tasks, with configurable parameters for sequence length, digit count, and motion patterns.

## Installation
```bash
uv pip install -r requirements.txt
```

## Requirements
- JAX
- NumPy
- Matplotlib (for visualization)
- Python 3.8+

## Usage
```bash
python jax_moving_mnist.py
```

## Features
- Configurable sequence length and digit count
- Random trajectory generation
- Collision handling
- Frame rendering pipeline

## License
MIT License - see LICENSE file for details

## Project Structure
```
├── jax_moving_mnist.py    # Core implementation
├── requirements.txt       # Dependency list
└── README.md              # This documentation
