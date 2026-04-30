# PMP Double-Integrator Visualizer

A Python GUI application for exploring Pontryagin's Maximum Principle (PMP) on double-integrator optimal-control problems.

The app compares three representative PMP cases with the same dynamics:

```text
x_dot = v
v_dot = u
```

Each mode plots the state, control, phase portrait, costate, Hamiltonian, and either terminal error or cost.

## Problem Modes

### Minimum Time (Bang-Bang)

Free-final-time problem with bounded input:

```text
minimize    T
subject to  x_dot = v
            v_dot = u
            |u| <= u_max
```

PMP gives a bang-bang control. The app solves the one-switch double-integrator solution analytically and reports:

- minimum time `T*`
- switching time
- bang-bang arc sequence
- terminal error

Variation control:

```text
Modified Switch Time
```

Changing it live re-simulates the bang-bang trajectory over the optimal horizon `T*`, so you can see how moving the switch misses the terminal state.

### Minimum Energy (Smooth)

Fixed-final-time quadratic-control problem:

```text
minimize    integral 0.5 u(t)^2 dt
subject to  x_dot = v
            v_dot = u
            x(0), v(0), x(T), v(T) fixed
```

The unconstrained PMP control is smooth and affine in time:

```text
u*(t) = a t + b
```

Optional input constraint:

```text
|u| <= u_max
```

When enabled, the app solves the saturated PMP law:

```text
u*(t) = clip(-p2(t), -u_max, u_max)
```

Variation control:

```text
Perturb Control Multiplier
```

Changing it live scales the nominal smooth control law and shows how the terminal error and energy change.

### Classical LQ PMP (Unconstrained)

Fixed-time linear-quadratic boundary-value problem:

```text
minimize    integral 0.5 (x(t)^2 + u(t)^2) dt
subject to  x_dot = v
            v_dot = u
            x(0), v(0), x(T), v(T) fixed
```

The Hamiltonian is:

```text
H = 0.5 (x^2 + u^2) + p1 v + p2 u
```

Stationarity gives:

```text
u*(t) = -p2(t)
```

The app solves the full Hamiltonian state-costate boundary-value problem using a matrix exponential.

Variation control:

```text
Initial Costate Multiplier
```

Changing it live scales the shooting parameter `p(0)` and re-integrates the Hamiltonian system. This shows how a wrong initial costate still follows the PMP ODEs but generally misses the final boundary condition.

## Plot Guide

- State Trajectory: position and velocity versus time.
- Control: optimal and varied input trajectories.
- Phase Portrait: velocity versus position.
- Costate: `p1(t)` and `p2(t)`.
- Hamiltonian: Hamiltonian along the trajectory.
- Bottom-right panel:
  - minimum-time mode: terminal error comparison
  - minimum-energy mode: energy comparison
  - classical LQ mode: cost functional comparison

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

If you are using the included virtual environment on Windows:

```powershell
venv\Scripts\python.exe main.py
```

Otherwise:

```bash
python main.py
```

## Project Structure

```text
main.py              Application entry point
core/solver.py       PMP problem solvers
ui/main_window.py    PyQt5 GUI and plotting
requirements.txt     Python dependencies
```

## Notes

The minimum-time and minimum-energy examples illustrate the contrast between bounded free-time PMP, which produces bang-bang control, and fixed-time quadratic PMP, which produces smooth control unless an input bound is enabled.

The classical LQ mode is useful for seeing the Hamiltonian/costate equations directly, including the shooting sensitivity of the initial costate.
