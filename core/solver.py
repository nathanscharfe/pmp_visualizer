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
        
    def dynamics(self, t, y):
        """
        Full augmented dynamics: [x, lambda]
        
        Args:
            t: Time
            y: [x_pos, x_vel, lambda_1, lambda_2]
        
        Returns:
            dy/dt
        """
        x_pos, x_vel, lam1, lam2 = y
        
        # Optimal control (clamped to [-umax, umax])
        u_opt = -lam2 / 2.0
        u = np.clip(u_opt, -self.umax, self.umax)
        
        # State equations
        dx_pos = x_vel
        dx_vel = u
        
        # Costate equations
        dlam1 = 0
        dlam2 = -lam1
        
        return [dx_pos, dx_vel, dlam1, dlam2]
    
    def residuals(self, lam0):
        """
        Residuals for boundary value problem.
        lam0 = [lambda_1(0), lambda_2(0)]
        
        We need to find lam0 such that x(T) = xf.
        """
        # Initial conditions
        y0 = [self.x0[0], self.x0[1], lam0[0], lam0[1]]
        
        # Integrate forward
        t = np.linspace(0, self.T, 500)
        try:
            solution = odeint(self.dynamics, y0, t)
            x_final = solution[-1, :2]
            
            # Residuals on boundary conditions
            return np.array([x_final[0] - self.xf[0], x_final[1] - self.xf[1]])
        except:
            return np.array([1e10, 1e10])
    
    def solve(self, lam0_guess=None):
        """
        Solve the optimal control problem.
        
        Returns:
            t: Time vector
            x: State trajectory [x_pos, x_vel]
            u: Control trajectory
            lam: Costate trajectory [lambda_1, lambda_2]
        """
        if lam0_guess is None:
            lam0_guess = np.array([0.1, 0.1])
        
        # Solve for initial costates using root finding
        from scipy.optimize import fsolve
        lam0_opt = fsolve(self.residuals, lam0_guess)
        
        # Integrate with optimal initial costates
        y0 = [self.x0[0], self.x0[1], lam0_opt[0], lam0_opt[1]]
        t = np.linspace(0, self.T, 500)
        solution = odeint(self.dynamics, y0, t)
        
        # Extract trajectory
        x_pos = solution[:, 0]
        x_vel = solution[:, 1]
        lam1 = solution[:, 2]
        lam2 = solution[:, 3]
        
        # Compute control along trajectory
        u = -lam2 / 2.0
        u = np.clip(u, -self.umax, self.umax)
        
        # Compute Hamiltonian
        H = u**2 + lam1 * x_vel + lam2 * u
        
        return {
            't': t,
            'x': np.column_stack([x_pos, x_vel]),
            'u': u,
            'lambda': np.column_stack([lam1, lam2]),
            'H': H,
            'lam0': lam0_opt
        }
