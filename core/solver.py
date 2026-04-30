import numpy as np
from scipy.integrate import odeint
from scipy.linalg import expm
from scipy.optimize import least_squares


class DoubleIntegratorProblem:
    """
    Minimum-time double integrator problem.
    
    State: x = [position, velocity]
    Dynamics: dx/dt = [velocity, u]
    Control: u in [-u_max, u_max]
    Cost: minimize final time T
    """
    
    def __init__(self, x0, xf, T, control_bound=1.0):
        """
        Initialize the problem.
        
        Args:
            x0: Initial state [x_pos_0, x_vel_0]
            xf: Final state [x_pos_f, x_vel_f]
            T: Initial time horizon guess (ignored during solving)
            control_bound: Maximum control magnitude
        """
        self.x0 = np.array(x0, dtype=float)
        self.xf = np.array(xf, dtype=float)
        self.T = float(T)
        self.umax = float(control_bound)

        if self.x0.shape != (2,) or self.xf.shape != (2,):
            raise ValueError("x0 and xf must be length-2 state vectors: [position, velocity].")
        if self.umax <= 0:
            raise ValueError("control_bound must be positive.")
        
    def solve(self, p0_guess=None):
        """
        Solve the minimum-time optimal control problem.
        
        Returns:
            Dictionary containing the time vector, state, control, costate, Hamiltonian,
            and minimum final time `T`.
        """
        candidates = []
        dx = self.xf[0] - self.x0[0]
        v0 = self.x0[1]
        vf = self.xf[1]

        if np.allclose(self.x0, self.xf, atol=1e-12):
            best = self._build_zero_time_solution()
            self.T = best['T']
            return best

        # Try pure bang arcs with u = +/- umax and no switch
        for u_value in [self.umax, -self.umax]:
            if np.isclose(u_value, 0.0):
                continue
            T = (vf - v0) / u_value
            if T <= 0:
                continue
            x_end = self.x0[0] + v0 * T + 0.5 * u_value * T**2
            if np.isclose(x_end, self.xf[0], atol=1e-6):
                sol = self._build_constant_control_solution(u_value, T)
                candidates.append(sol)

        # Try bang-bang solutions with one switch
        for a in [self.umax, -self.umax]:
            candidates.extend(self._solve_bang_bang(a, dx, v0, vf))

        if not candidates:
            raise RuntimeError("No feasible minimum-time trajectory found for the given boundary conditions.")

        best = min(candidates, key=lambda s: s['T'])
        self.T = best['T']
        return best

    def _terminal_error(self, x):
        return float(np.linalg.norm(np.asarray(x, dtype=float) - self.xf))

    def _build_zero_time_solution(self):
        t = np.array([0.0])
        x = np.array([[self.x0[0], self.x0[1]]])
        u = np.array([0.0])
        p = np.zeros((1, 2))
        H = np.zeros(1)

        return {
            't': t,
            'x': x,
            'u': u,
            'p': p,
            'H': H,
            'T': 0.0,
            'switch_time': None,
            'arc_sequence': 'already at target',
            'terminal_error': 0.0,
        }

    def _solve_bang_bang(self, a, dx, v0, vf):
        """
        Solve for a single bang-bang sequence starting with control `a` and switching to `-a`.
        """
        results = []
        D = (v0 - vf) / a
        A = a
        B = 2.0 * v0
        C = v0 * D - 0.5 * a * D**2 - dx
        discriminant = B**2 - 4.0 * A * C

        if discriminant < 0:
            return results

        roots = np.roots([A, B, C])
        for root in roots:
            if not np.isreal(root):
                continue
            t1 = float(np.real(root))
            if t1 < 0:
                continue
            T = 2.0 * t1 + D
            if T <= t1 or T <= 0:
                continue

            sol = self._build_bang_bang_solution(a, t1, T)
            if sol is None:
                continue
            results.append(sol)

        return results

    def _build_constant_control_solution(self, u_value, T):
        t = np.linspace(0, T, 500)
        x = self.x0[0] + self.x0[1] * t + 0.5 * u_value * t**2
        v = self.x0[1] + u_value * t
        u = np.full_like(t, u_value)
        p1 = np.zeros_like(t)
        p2 = np.full_like(t, -1.0 / u_value)
        p = np.column_stack([p1, p2])
        H = np.zeros_like(t)

        return {
            't': t,
            'x': np.column_stack([x, v]),
            'u': u,
            'p': p,
            'H': H,
            'T': T,
            'switch_time': None,
            'arc_sequence': f'u = {u_value:.3g}',
            'terminal_error': self._terminal_error([x[-1], v[-1]]),
        }

    def _build_bang_bang_solution(self, a, t1, T):
        v1 = self.x0[1] + a * t1
        if np.isclose(v1, 0.0):
            return None

        t = np.linspace(0, T, 500)
        u = np.where(t <= t1, a, -a)

        x = np.zeros_like(t)
        v = np.zeros_like(t)
        mask = t <= t1
        x[mask] = self.x0[0] + self.x0[1] * t[mask] + 0.5 * a * t[mask]**2
        v[mask] = self.x0[1] + a * t[mask]

        if not np.all(mask):
            tau = t[~mask] - t1
            x1 = self.x0[0] + self.x0[1] * t1 + 0.5 * a * t1**2
            v1 = self.x0[1] + a * t1
            x[~mask] = x1 + v1 * tau - 0.5 * a * tau**2
            v[~mask] = v1 - a * tau

        x_end = np.array([x[-1], v[-1]])
        if not np.allclose(x_end, self.xf, atol=1e-5):
            return None

        p1 = -1.0 / v1
        p2 = p1 * (t1 - t)
        p = np.column_stack([np.full_like(t, p1), p2])
        H = 1.0 + p1 * v + p2 * u

        return {
            't': t,
            'x': np.column_stack([x, v]),
            'u': u,
            'p': p,
            'H': H,
            'T': T,
            'switch_time': t1,
            'arc_sequence': f'u = {a:.3g} then u = {-a:.3g}',
            'terminal_error': self._terminal_error(x_end),
        }

    def simulate_with_control(self, u_func):
        """
        Simulate the system with a given control input on the current horizon.
        
        Args:
            u_func: Function that returns control u given time t
        
        Returns:
            Dictionary with trajectory data
        """
        def dyn_manual(y, t):
            x_pos, x_vel = y
            u = u_func(t)
            u = np.clip(u, -self.umax, self.umax)
            return [x_vel, u]

        y0 = [self.x0[0], self.x0[1]]
        t = np.linspace(0, self.T, 500)
        solution = odeint(dyn_manual, y0, t)

        x_pos = solution[:, 0]
        x_vel = solution[:, 1]
        u_traj = np.array([np.clip(u_func(ti), -self.umax, self.umax) for ti in t])
        terminal_state = np.array([x_pos[-1], x_vel[-1]])

        return {
            't': t,
            'x': np.column_stack([x_pos, x_vel]),
            'u': u_traj,
            'cost': self.T,
            'T': self.T,
            'terminal_error': self._terminal_error(terminal_state),
        }


class MinimumEnergyDoubleIntegratorProblem:
    """
    Fixed-time minimum-energy double integrator problem.

    State: x = [position, velocity]
    Dynamics: dx/dt = [velocity, u]
    Cost: minimize integral 0.5 * u^2 dt over a fixed final time T
    """

    def __init__(self, x0, xf, T, control_bound=None):
        self.x0 = np.array(x0, dtype=float)
        self.xf = np.array(xf, dtype=float)
        self.T = float(T)
        self.control_bound = None if control_bound is None else float(control_bound)

        if self.x0.shape != (2,) or self.xf.shape != (2,):
            raise ValueError("x0 and xf must be length-2 state vectors: [position, velocity].")
        if self.T <= 0:
            raise ValueError("minimum-energy problem requires a positive fixed time horizon T.")
        if self.control_bound is not None and self.control_bound <= 0:
            raise ValueError("control_bound must be positive when enabled.")

    def solve(self):
        """
        Solve the fixed-time minimum-energy problem.

        PMP gives H = 0.5 u^2 + p1 v + p2 u and stationarity gives u = -p2.
        Since p1 is constant and p2 is linear, the optimal control is u(t) = a t + b.
        """
        a, b = self._solve_control_coefficients()
        if self.control_bound is None:
            return self.build_solution_from_coefficients(a, b)

        unconstrained = self.build_solution_from_coefficients(a, b)
        if np.max(np.abs(unconstrained['u_command'])) <= self.control_bound + 1e-9:
            unconstrained['control_bound'] = self.control_bound
            unconstrained['constraint_active'] = False
            return unconstrained

        a_bounded, b_bounded = self._solve_bounded_control_coefficients(a, b)
        solution = self.build_solution_from_coefficients(a_bounded, b_bounded)
        if solution['terminal_error'] > 1e-5:
            raise RuntimeError(
                "No feasible bounded-input minimum-energy trajectory found for this fixed time."
            )
        return solution

    def _solve_control_coefficients(self):
        dx = self.xf[0] - self.x0[0]
        dv = self.xf[1] - self.x0[1]
        position_residual = dx - self.x0[1] * self.T

        system = np.array(
            [
                [0.5 * self.T**2, self.T],
                [self.T**3 / 6.0, 0.5 * self.T**2],
            ]
        )
        rhs = np.array([dv, position_residual])
        return np.linalg.solve(system, rhs)

    def build_solution_from_coefficients(self, a, b, multiplier=1.0):
        a_scaled = float(multiplier) * float(a)
        b_scaled = float(multiplier) * float(b)

        t = np.linspace(0.0, self.T, 500)
        u_command = a_scaled * t + b_scaled
        u = self._apply_control_bound(u_command)
        if np.allclose(u, u_command, atol=1e-12):
            v = self.x0[1] + 0.5 * a_scaled * t**2 + b_scaled * t
            x = self.x0[0] + self.x0[1] * t + (a_scaled * t**3) / 6.0 + 0.5 * b_scaled * t**2
            cost = self.smooth_energy_cost(a_scaled, b_scaled)
        else:
            x, v = self._integrate_clipped_affine(t, a_scaled, b_scaled)
            cost = self.clipped_energy_cost(a_scaled, b_scaled)

        p1 = np.full_like(t, a_scaled)
        p2 = -u_command
        p = np.column_stack([p1, p2])
        H = 0.5 * u**2 + p1 * v + p2 * u
        terminal_state = np.array([x[-1], v[-1]])
        bound_text = ""
        if self.control_bound is not None:
            bound_text = f", -{self.control_bound:.3g}, {self.control_bound:.3g})"
            control_text = f'u(t) = clip({a_scaled:.3g} t + {b_scaled:.3g}{bound_text}'
        else:
            control_text = f'u(t) = {a_scaled:.3g} t + {b_scaled:.3g}'

        return {
            't': t,
            'x': np.column_stack([x, v]),
            'u': u,
            'u_command': u_command,
            'p': p,
            'H': H,
            'T': self.T,
            'cost': cost,
            'switch_time': None,
            'arc_sequence': control_text,
            'terminal_error': self._terminal_error(terminal_state),
            'control_coefficients': (float(a), float(b)),
            'control_multiplier': float(multiplier),
            'control_bound': self.control_bound,
            'constraint_active': self.control_bound is not None and np.any(
                np.abs(u_command) > self.control_bound + 1e-9
            ),
        }

    def build_multiplier_solution(self, optimal_solution, multiplier):
        a, b = optimal_solution['control_coefficients']
        return self.build_solution_from_coefficients(a, b, multiplier=multiplier)

    def _solve_bounded_control_coefficients(self, a_guess, b_guess):
        self._validate_bounded_feasibility()

        guesses = [
            np.array([a_guess, b_guess], dtype=float),
            np.array([0.0, self.control_bound], dtype=float),
            np.array([0.0, -self.control_bound], dtype=float),
            np.array([-2.0 * self.control_bound / self.T, self.control_bound], dtype=float),
            np.array([2.0 * self.control_bound / self.T, -self.control_bound], dtype=float),
            np.array([4.0 * self.control_bound / self.T, -2.0 * self.control_bound], dtype=float),
            np.array([-4.0 * self.control_bound / self.T, 2.0 * self.control_bound], dtype=float),
        ]

        best = None
        for guess in guesses:
            result = least_squares(
                self._bounded_terminal_residual,
                guess,
                xtol=1e-11,
                ftol=1e-11,
                gtol=1e-11,
                max_nfev=500,
            )
            error = np.linalg.norm(self._bounded_terminal_residual(result.x))
            if best is None or error < best[0]:
                best = (error, result.x)

        if best is None or best[0] > 1e-6:
            raise RuntimeError(
                "No feasible bounded-input minimum-energy trajectory found for this fixed time."
            )
        return best[1]

    def _bounded_terminal_residual(self, coefficients):
        a, b = coefficients
        terminal_state = self._terminal_state_for_coefficients(a, b)
        return terminal_state - self.xf

    def _validate_bounded_feasibility(self):
        try:
            min_time = DoubleIntegratorProblem(
                self.x0,
                self.xf,
                self.T,
                self.control_bound,
            ).solve()['T']
        except RuntimeError as exc:
            raise RuntimeError("The bounded-input fixed-time problem is infeasible.") from exc

        if min_time > self.T + 1e-7:
            raise RuntimeError(
                f"The fixed time T = {self.T:.6g} is shorter than the bounded-input minimum time "
                f"T* = {min_time:.6g}."
            )

    def _apply_control_bound(self, u):
        if self.control_bound is None:
            return np.asarray(u, dtype=float)
        return np.clip(u, -self.control_bound, self.control_bound)

    def _terminal_state_for_coefficients(self, a, b):
        x, v = self._propagate_interval(self.x0[0], self.x0[1], 0.0, self.T, a, b)
        return np.array([x, v])

    def _integrate_clipped_affine(self, t, a, b):
        x = np.zeros_like(t)
        v = np.zeros_like(t)
        x[0] = self.x0[0]
        v[0] = self.x0[1]

        for index in range(1, len(t)):
            x[index], v[index] = self._propagate_interval(
                x[index - 1],
                v[index - 1],
                t[index - 1],
                t[index],
                a,
                b,
            )

        return x, v

    def _propagate_interval(self, x0, v0, t0, t1, a, b):
        x = x0
        v = v0
        points = [t0, *self._clip_breakpoints(a, b, t0, t1), t1]

        for start, end in zip(points[:-1], points[1:]):
            if np.isclose(start, end):
                continue
            x, v = self._propagate_smooth_piece(x, v, start, end, a, b)

        return x, v

    def _propagate_smooth_piece(self, x0, v0, t0, t1, a, b):
        dt = t1 - t0
        control = self._control_value(0.5 * (t0 + t1), a, b)

        if self.control_bound is not None and np.isclose(abs(control), self.control_bound):
            x1 = x0 + v0 * dt + 0.5 * control * dt**2
            v1 = v0 + control * dt
            return x1, v1

        dv = 0.5 * a * (t1**2 - t0**2) + b * dt
        weighted_control_integral = (
            t1 * (0.5 * a * (t1**2 - t0**2) + b * dt)
            - ((a / 3.0) * (t1**3 - t0**3) + 0.5 * b * (t1**2 - t0**2))
        )
        x1 = x0 + v0 * dt + weighted_control_integral
        v1 = v0 + dv
        return x1, v1

    def _clip_breakpoints(self, a, b, t0, t1):
        if self.control_bound is None or np.isclose(a, 0.0):
            return []

        points = []
        for level in [-self.control_bound, self.control_bound]:
            time = (level - b) / a
            if t0 < time < t1:
                points.append(float(time))
        return sorted(points)

    def _control_value(self, t, a, b):
        value = a * t + b
        if self.control_bound is None:
            return value
        return float(np.clip(value, -self.control_bound, self.control_bound))

    def _integrate_control(self, t, u):
        dt = np.diff(t)
        increments = 0.5 * (u[:-1] + u[1:]) * dt
        velocity_integral = np.concatenate([[0.0], np.cumsum(increments)])
        v = self.x0[1] + velocity_integral

        position_increments = 0.5 * (v[:-1] + v[1:]) * dt
        position_integral = np.concatenate([[0.0], np.cumsum(position_increments)])
        x = self.x0[0] + position_integral
        return x, v

    def energy_cost(self, t, u):
        return float(0.5 * np.trapezoid(u**2, t))

    def smooth_energy_cost(self, a, b):
        return float(
            0.5
            * (
                (a**2 * self.T**3) / 3.0
                + a * b * self.T**2
                + b**2 * self.T
            )
        )

    def clipped_energy_cost(self, a, b):
        points = [0.0, *self._clip_breakpoints(a, b, 0.0, self.T), self.T]
        cost = 0.0

        for start, end in zip(points[:-1], points[1:]):
            if np.isclose(start, end):
                continue
            control = self._control_value(0.5 * (start + end), a, b)
            if self.control_bound is not None and np.isclose(abs(control), self.control_bound):
                cost += 0.5 * control**2 * (end - start)
            else:
                cost += 0.5 * (
                    (a**2 / 3.0) * (end**3 - start**3)
                    + a * b * (end**2 - start**2)
                    + b**2 * (end - start)
                )

        return float(cost)

    def _terminal_error(self, x):
        return float(np.linalg.norm(np.asarray(x, dtype=float) - self.xf))


class ClassicalLQDoubleIntegratorProblem:
    """
    Fixed-time unconstrained linear-quadratic double integrator problem.

    State: x = [position, velocity]
    Dynamics: x_dot = v, v_dot = u
    Cost: minimize integral 0.5 * (q_x * x^2 + q_v * v^2 + r * u^2) dt
    Boundary conditions: initial and final states fixed.
    """

    def __init__(
        self,
        x0,
        xf,
        T,
        position_weight=1.0,
        velocity_weight=0.0,
        control_weight=1.0,
    ):
        self.x0 = np.array(x0, dtype=float)
        self.xf = np.array(xf, dtype=float)
        self.T = float(T)
        self.qx = float(position_weight)
        self.qv = float(velocity_weight)
        self.r = float(control_weight)

        if self.x0.shape != (2,) or self.xf.shape != (2,):
            raise ValueError("x0 and xf must be length-2 state vectors: [position, velocity].")
        if self.T <= 0:
            raise ValueError("classical LQ problem requires a positive fixed time horizon T.")
        if self.qx < 0 or self.qv < 0:
            raise ValueError("state weights must be nonnegative.")
        if self.r <= 0:
            raise ValueError("control weight must be positive.")

    def solve(self):
        """
        Solve the Hamiltonian boundary-value problem exactly with a matrix exponential.

        PMP:
            H = 0.5(q_x x^2 + q_v v^2 + r u^2) + p1 v + p2 u
            u* = -p2 / r
            p1_dot = -q_x x
            p2_dot = -q_v v - p1
        """
        A = self.hamiltonian_matrix()
        transition = expm(A * self.T)
        free_response = transition[:2, :2] @ self.x0
        costate_to_terminal = transition[:2, 2:]

        try:
            p0 = np.linalg.solve(costate_to_terminal, self.xf - free_response)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("The classical LQ boundary-value problem is singular.") from exc

        return self.build_solution_from_initial_costate(p0)

    def build_solution_from_initial_costate(self, p0, multiplier=1.0):
        p0_scaled = float(multiplier) * np.asarray(p0, dtype=float)
        initial_augmented_state = np.array(
            [self.x0[0], self.x0[1], p0_scaled[0], p0_scaled[1]]
        )
        A = self.hamiltonian_matrix()
        t = np.linspace(0.0, self.T, 500)
        augmented = np.array([expm(A * time) @ initial_augmented_state for time in t])

        position = augmented[:, 0]
        velocity = augmented[:, 1]
        p1 = augmented[:, 2]
        p2 = augmented[:, 3]
        control = -p2 / self.r
        H = 0.5 * (
            self.qx * position**2 + self.qv * velocity**2 + self.r * control**2
        ) + p1 * velocity + p2 * control
        running_cost = 0.5 * (
            self.qx * position**2 + self.qv * velocity**2 + self.r * control**2
        )
        cost = float(np.trapezoid(running_cost, t))
        terminal_state = np.array([position[-1], velocity[-1]])

        return {
            't': t,
            'x': np.column_stack([position, velocity]),
            'u': control,
            'p': np.column_stack([p1, p2]),
            'H': H,
            'T': self.T,
            'cost': cost,
            'switch_time': None,
            'arc_sequence': f'u(t) = -p2(t) / {self.r:.3g}',
            'terminal_error': self._terminal_error(terminal_state),
            'p0': p0_scaled,
            'weights': (self.qx, self.qv, self.r),
            'costate_multiplier': float(multiplier),
        }

    def build_multiplier_solution(self, optimal_solution, multiplier):
        return self.build_solution_from_initial_costate(
            optimal_solution['p0'],
            multiplier=multiplier,
        )

    def hamiltonian_matrix(self):
        return np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0 / self.r],
                [-self.qx, 0.0, 0.0, 0.0],
                [0.0, -self.qv, -1.0, 0.0],
            ]
        )

    def _terminal_error(self, x):
        return float(np.linalg.norm(np.asarray(x, dtype=float) - self.xf))
