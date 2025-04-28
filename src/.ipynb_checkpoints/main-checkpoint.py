import numpy as np
from tools.model import Model
from tools.numerical_methods import NumericalMethod
from tools.visualizer import Visualizer
from tools.utils import get_critical_point

# === Paramètres ===
x0 = 0.14 # Concentration biomasse
y0 = 0.14 # Concentration polluant bioréacteur
z0 = 0.14 # Concentration polluant lac
init_state = np.array([x0, y0, z0])  # cas instable
mu = 10 # Loi de croissance de la biomasse
Q = 1.0 # Débit entrée/sortie entre lac et bioréacteur
VR = 4.5e10 # Volume bioréacteur
VL = 4.5e12 # Volume lac
epsilon = VR/VL # Rapport VR/VL
t0 = 0 # Début observation (en jours)
tf = 360 # Fin observation (en jours)
h = 1 # Pas de temps (= 1 jour idéalement)
t = np.arange(t0, tf + h, h) # Plage d'observation

# === Initialisation ===
method = NumericalMethod(system = lambda t, y : Model.system(t, y, mu, Q, epsilon),
                         jacobian = lambda y : Model.jacobian(y, mu, Q, epsilon))
viz = Visualizer(t)

# === Résolution ===
sol_rk4 = method.rk4(init_state, t)
# Changer h = 1 à h = 0.1 pour avoir la stabilité d'Euler
h = 0.1 # Pas de temps (= 1 jour idéalement)
t = np.arange(t[0], t[len(t)-1], h)
sol_exp = method.euler_explicit(init_state, t)[::10]
sol_imp_fp = method.euler_implicit_fixed_point(init_state, t)[::10]
sol_imp_newton = method.euler_implicit_newton(init_state, t)[::10]

# === Visualisations ===
viz.plot_xyz(sol_rk4, label = "RK4")
viz.plot_3d(sol_rk4, label = "RK4", color = 'blue', critical_point = get_critical_point(mu, Q), arrow_index = len(sol_rk4)//4)

viz.plot_projections(
    [sol_exp, sol_imp_fp, sol_imp_newton],
    labels = ["Euler explicit", "Euler implicit (FP)", "Euler implicit (Newton)"],
    colors = ["green", "orange", "red"],
    critical_points = [get_critical_point(mu, Q)]*3,
    arrow_index = len(sol_exp)//4
)
