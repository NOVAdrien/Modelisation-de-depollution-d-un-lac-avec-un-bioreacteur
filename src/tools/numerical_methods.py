import numpy as np
from scipy.optimize import root

class NumericalMethod:
    def __init__(self, system, jacobian=None):
        """
        Parameters:
        - system: function(t, state) -> np.array
        - jacobian: function(state) -> np.array (optional, needed for Newton)
        """
        self.system = system
        self.jacobian = jacobian

    def euler_explicit(self, y0, t):
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        for i in range(len(t) - 1):
            dt = t[i+1] - t[i]
            y[i+1] = y[i] + dt * self.system(t[i], y[i])
        return y

    def euler_implicit_fixed_point(self, y0, t, max_iter=10, tol=1e-6):
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        for i in range(len(t) - 1):
            dt = t[i+1] - t[i]
            yn = y[i]
            yn1 = yn.copy()
            for _ in range(max_iter):
                yn1_new = yn + dt * self.system(t[i+1], yn1)
                if np.linalg.norm(yn1_new - yn1) < tol:
                    break
                yn1 = yn1_new
            y[i+1] = yn1
        return y

    def euler_implicit_newton(self, y0, t, tol=1e-6):
        if self.jacobian is None:
            raise ValueError("Jacobian function is required for Newton method.")

        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        for i in range(len(t) - 1):
            dt = t[i+1] - t[i]
            yn = y[i]

            def g(y_next):
                return y_next - yn - dt * self.system(t[i+1], y_next)

            def J(y_next):
                return np.eye(len(y0)) - dt * self.jacobian(y_next)

            sol = root(g, yn, jac=J, method='hybr')
            y[i+1] = sol.x
        return y

    def rk4(self, y0, t):
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        for i in range(len(t) - 1):
            dt = t[i+1] - t[i]
            k1 = self.system(t[i], y[i])
            k2 = self.system(t[i] + dt / 2, y[i] + dt / 2 * k1)
            k3 = self.system(t[i] + dt / 2, y[i] + dt / 2 * k2)
            k4 = self.system(t[i] + dt, y[i] + dt * k3)
            y[i+1] = y[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y
