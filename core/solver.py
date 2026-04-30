import numpy as np
from scipy.integrate import odeint, solve_bvp
from scipy.optimize import minimize_scalar


class DoubleIntegratorProblem:
    """
    Double integrator optimal control problem.
    
    State: x = [position, velocity]
    Dynamics: dx/dt = [velocity, u]
    Control: u in [-1, 1]
    Cost: integral of u^2 dt (energy minimization)
    """
    
    def __init__(self, x0, xf, T, control_bound=1.0):
        """
        Initialize the problem.
        
        Args:
            x0: Initial state [x_pos_0, x_vel_0]
            xf: Final state [x_pos_f, x_vel_f]
            T: Time horizon
            control_bound: Maximum control magnitude
        """
        self.x0 = np.array(x0)
        self.xf = np.array(xf)
        self.T = T
        self.umax = control_bound
        
    def dynamics(self, y, t):
        """
        Full augmented dynamics: [x, p]
        
        Args:
            y: [x_pos, x_vel, p_1, p_2]
            t: Time
        
        Returns:
            dy/dt
        """
        x_pos, x_vel, p1, p2 = y
        
        # Optimal control (clamped to [-umax, umax])
        u_opt = -p2 / 2.0
        u = np.clip(u_opt, -self.umax, self.umax)
        
        # State equations
        dx_pos = x_vel
        dx_vel = u
        
        # Costate (p) equations
        dp1 = 0
        dp2 = -p1
        
        return [dx_pos, dx_vel, dp1, dp2]
    
    def residuals(self, p0):
        """
        Residuals for boundary value problem.
        p0 = [p_1(0), p_2(0)]
        
        We need to find p0 such that x(T) = xf.
        """
        # Initial conditions
        y0 = [self.x0[0], self.x0[1], p0[0], p0[1]]
        
        # Integrate forward
        t = np.linspace(0, self.T, 500)
        try:
            solution = odeint(self.dynamics, y0, t)
            x_final = solution[-1, :2]
            
            # Residuals on boundary conditions
            return np.array([x_final[0] - self.xf[0], x_final[1] - self.xf[1]])
        except:
            return np.array([1e10, 1e10])
    
    def solve(self, p0_guess=None):
        """
        Solve the optimal control problem.
        
        Returns:
            t: Time vector
            x: State trajectory [x_pos, x_vel]
            u: Control trajectory
            p: Costate trajectory [p_1, p_2]
            H: Hamiltonian trajectory
        """
        if p0_guess is None:
            p0_guess = np.array([0.1, 0.1])
        
        # Solve for initial costates using root finding
        from scipy.optimize import fsolve
        p0_opt = fsolve(self.residuals, p0_guess)
        
        # Integrate with optimal initial costates
        y0 = [self.x0[0], self.x0[1], p0_opt[0], p0_opt[1]]
        t = np.linspace(0, self.T, 500)
        solution = odeint(self.dynamics, y0, t)
        
        # Extract trajectory
        x_pos = solution[:, 0]
        x_vel = solution[:, 1]
        p1 = solution[:, 2]
        p2 = solution[:, 3]
        
        # Compute control along trajectory
        u = -p2 / 2.0
        u = np.clip(u, -self.umax, self.umax)
        
        # Compute Hamiltonian H = u^2 + p1*x_vel + p2*u
        H = u**2 + p1 * x_vel + p2 * u
        
        return {
            't': t,
            'x': np.column_stack([x_pos, x_vel]),
            'u': u,
            'p': np.column_stack([p1, p2]),
            'H': H,
            'p0': p0_opt
        }
    
    def simulate_with_control(self, u_func):
        """
        Simulate the system with a given control input.
        
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
        
        # Integrate forward with given control
        y0 = [self.x0[0], self.x0[1]]
        t = np.linspace(0, self.T, 500)
        solution = odeint(dyn_manual, y0, t)
        
        x_pos = solution[:, 0]
        x_vel = solution[:, 1]
        u_traj = np.array([u_func(ti) for ti in t])
        
        # Compute cost (integral of u^2)
        cost = np.trapz(u_traj**2, t)
        
        return {
            't': t,
            'x': np.column_stack([x_pos, x_vel]),
            'u': u_traj,
            'cost': cost
        }
