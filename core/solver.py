import numpy as np
from scipy.integrate import odeint


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
