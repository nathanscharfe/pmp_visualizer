# PMP Minimum-Time Visualizer

A Python GUI application for visualizing Pontryagin's Maximum Principle for a minimum-time double-integrator problem.

## About

The app solves the bounded-control double integrator

```text
x_dot = v
v_dot = u
|u| <= u_max
```

with fixed initial and final states and free final time. It displays the bang-bang optimal control, state trajectory, phase portrait, costates, Hamiltonian, switch time, and terminal error. The perturbation tools simulate a noisy or biased control over the optimal horizon `T*` so you can see how deviations from the PMP control miss the terminal constraint.

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```
