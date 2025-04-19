import numpy as np
from model import system, jacobian
from numerical_methods import NumericalMethod
from visualizer import Visualizer
from utils import get_critical_point

# === Paramètres ===
mu = 1.0
Q = 1.0
epsilon = 0.1
t = np.linspace(0, 50, 1000)
init_state = np.array([1.0, 1.5, 2.0])  # cas instable

# === Initialisation ===
method = NumericalMethod(system=lambda t, y: system(t, y, mu, Q, epsilon),
                         jacobian=lambda y: jacobian(y, mu, Q, epsilon))
viz = Visualizer(t)

# === Résolution ===
sol_rk4 = method.rk4(init_state, t)
sol_exp = method.euler_explicit(init_state, t)
sol_imp_fp = method.euler_implicit_fixed_point(init_state, t)
sol_imp_newton = method.euler_implicit_newton(init_state, t)

# === Visualisations ===
viz.plot_xyz(sol_rk4, label="RK4")
viz.plot_3d(sol_rk4, label="RK4", color='blue', critical_point=get_critical_point(mu, Q))

viz.plot_projections(
    [sol_exp, sol_imp_fp, sol_imp_newton],
    labels=["Euler explicit", "Euler implicit (FP)", "Euler implicit (Newton)"],
    colors=["green", "orange", "red"],
    critical_points=[get_critical_point(mu, Q)]*3
)
