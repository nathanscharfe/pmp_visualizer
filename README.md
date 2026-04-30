# PMP Visualizer

A Python GUI application for visualizing and understanding **Pontryagin's Maximum Principle (PMP)** in optimal control theory.

## Overview

Pontryagin's Maximum Principle is a fundamental result in optimal control theory that provides necessary conditions for the optimality of a control trajectory. This project provides an interactive graphical interface to:

- Visualize optimal control problems and their solutions
- Display costate trajectories (adjoint variables)
- Show phase portraits and control histories
- Animate system trajectories under optimal control
- Analyze Hamiltonian dynamics

## Features

- **Interactive Visualization**: Real-time plotting of state trajectories, costates, and controls
- **Multiple Problem Types**: Support for various optimal control problems (finite-horizon, tracking, energy minimization, etc.)
- **Phase Portrait Analysis**: Visualize system dynamics in phase space
- **Hamiltonian Dynamics**: Display and analyze Hamiltonian flow
- **Customizable Parameters**: Adjust problem parameters and initial conditions dynamically
- **Export Results**: Save visualizations and solution data

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pmp_visualizer
```

2. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Launch the GUI application:
```bash
python main.py
```

### Basic Workflow

1. **Select or Define a Problem**: Choose from predefined optimal control problems or define a custom problem
2. **Configure Parameters**: Set initial conditions, time horizon, and cost function parameters
3. **Solve**: Run the solver to compute the optimal trajectory using Pontryagin's Maximum Principle
4. **Visualize**: Explore interactive plots of states, costates, controls, and system dynamics
5. **Analyze**: Examine phase portraits, Hamiltonian values, and boundary conditions

## Project Structure

```
pmp_visualizer/
├── README.md
├── requirements.txt
├── main.py                 # Entry point
├── ui/
│   ├── __init__.py
│   └── main_window.py      # Main GUI window
├── core/
│   ├── __init__.py
│   ├── problems.py         # Optimal control problem definitions
│   ├── solver.py           # PMP-based solver
│   └── utils.py            # Utility functions
├── visualization/
│   ├── __init__.py
│   ├── plotter.py          # Matplotlib visualization
│   └── animations.py       # Animation utilities
└── examples/
    └── example_problems.py # Example problem definitions
```

## Theory

**Pontryagin's Maximum Principle** states that for an optimal control trajectory $u^*(t)$, there exist costate variables $\lambda(t)$ such that the Hamiltonian:

$$H(x, \lambda, u, t) = L(x, u, t) + \lambda^T f(x, u, t)$$

is maximized with respect to $u$ at each time $t$. The adjoint (costate) equations are:

$$\dot{\lambda} = -\frac{\partial H}{\partial x}$$

with terminal condition:

$$\lambda(T) = \frac{\partial \Phi(x(T))}{\partial x}$$

where $\Phi$ is the terminal cost.

For more information, refer to optimal control textbooks such as:
- "Optimal Control: An Introduction" by Kirk
- "The Brachistochrone Problem" and classical calculus of variations references

## Dependencies

See `requirements.txt` for a complete list. Main dependencies include:
- matplotlib
- numpy
- scipy
- PyQt5 or PySimpleGUI (for GUI)

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is part of ASE 381P3 (Optimal Control) coursework.

## Author

Created for ASE 381P3 - Optimal Control (Instructor: Dr. Bakolas)

## References

- L.D. Berkovitz, "Optimal Control Theory", Springer-Verlag
- A.E. Bryson and Y.C. Ho, "Applied Optimal Control"
- D.E. Kirk, "Optimal Control Theory: An Introduction"
