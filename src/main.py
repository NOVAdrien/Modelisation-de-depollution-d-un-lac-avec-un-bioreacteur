# importer les bibliothèques
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import root
from scipy.optimize import minimize_scalar

# importer les classes
from tools.model import System
from tools.numerical_methods import Methods
from tools.utils import Points, Function
from tools.visualize import Plot


# Cas $\epsilon = 0$
## Point critique 1 : $A = \begin{pmatrix} 0 \\ z0 \\ z0 \end{pmatrix} $
### Cas 1 : $\mu z_0 < Q$ -> test stabilité

# === Paramètres du modèle === #
mu = 0.1
Q = 1
epsilon = 0

# === Paramètres de simulation === #
t_min = 0
t_max = 30
pas = 1
t = np.arange(start = t_min, stop = t_max + 1, step = pas)

# === Conditions initiales === #
x0 = 2
y0 = 3
z0 = 5
initial_state = np.array([x0, y0, z0])

# === Simulation === #
solution = Methods.rk4(System.system, initial_state, t, mu, Q, epsilon)
print("point critique = ", solution[t_max])

# === Tracé des résultats === #
plt.figure(figsize=(10, 6))
# Courbes
plt.plot(t, solution[:, 0], "r", label = 'x(t) : biomasse')
plt.plot(t, solution[:, 1], "g", label = 'y(t) : polluant réacteur')
plt.plot(t, solution[:, 2], "b", label = 'z(t) : polluant lac')
# Point critique
plt.scatter(t_max, 0 , color='r', s = 100, marker='x', label = "x* = 0")
plt.scatter(t_max, z0, color='g', s = 100, marker='x', label = "y* = z0")
plt.scatter(t_max, z0, color='b', s = 100, marker='x', label = "z* = z0")
plt.xlabel('Temps')
plt.ylabel('Concentrations')
plt.title('Simulation du système pour µz0 < Q')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### Cas 2 : $\mu z_0 = Q$ -> test stabilité

# === Paramètres du modèle === #
mu = 0.2
Q = 1
epsilon = 0

# === Paramètres de simulation === #
t_min = 0
t_max = 300
pas = 1
t = np.arange(start = t_min, stop = t_max + 1, step = pas)

# === Conditions initiales === #
x0 = 2
y0 = 3
z0 = 5
initial_state = np.array([x0, y0, z0])

# === Simulation === #
solution = Methods.rk4(System.system, initial_state, t, mu, Q, epsilon)
print("point critique = ", solution[t_max])

# === Tracé des résultats === #
plt.figure(figsize=(10, 6))
# Courbes
plt.plot(t, solution[:, 0], "r", label = 'x(t) : biomasse')
plt.plot(t, solution[:, 1], "g", label = 'y(t) : polluant réacteur')
plt.plot(t, solution[:, 2], "b", label = 'z(t) : polluant lac')
# Point critique
plt.scatter(t_max, 0 , color='r', s = 100, marker='x', label = "x* = 0")
plt.scatter(t_max, z0, color='g', s = 100, marker='x', label = "y* = z0")
plt.scatter(t_max, z0, color='b', s = 100, marker='x', label = "z* = z0")
plt.xlabel('Temps')
plt.ylabel('Concentrations')
plt.title('Simulation du système pour µz0 = Q')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### Cas 3 : $\mu z_0 > Q$ -> test stabilité

# === Paramètres du modèle === #
mu = 0.5
Q = 1
epsilon = 0

# === Paramètres de simulation === #
t_min = 0
t_max = 4
pas = 1
t = np.arange(start = t_min, stop = t_max + 1, step = pas)

# === Conditions initiales === #
x0 = 2
y0 = 3
z0 = 5
initial_state = np.array([x0, y0, z0])

# === Simulation === #
solution = Methods.rk4(System.system, initial_state, t, mu, Q, epsilon)
print("point critique = ", solution[t_max])

# === Tracé des résultats === #
plt.figure(figsize=(10, 6))
# Courbes
plt.plot(t, solution[:, 0], "r", label = 'x(t) : biomasse')
plt.plot(t, solution[:, 1], "g", label = 'y(t) : polluant réacteur')
plt.plot(t, solution[:, 2], "b", label = 'z(t) : polluant lac')
# Point critique
plt.scatter(t_max, 0 , color='r', s = 100, marker='x', label = "x* = 0")
plt.scatter(t_max, z0, color='g', s = 100, marker='x', label = "y* = z0")
plt.scatter(t_max, z0, color='b', s = 100, marker='x', label = "z* = z0")
plt.xlabel('Temps')
plt.ylabel('Concentrations')
plt.title('Simulation du système pour µz0 > Q')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### Cas 3 : $\mu z_0 > Q$ -> test instabilité

# === Paramètres du modèle === #
mu = 0.5
Q = 1
epsilon = 0

# === Paramètres de simulation === #
t_min = 0
t_max = 4
pas = 1
t = np.arange(start = t_min, stop = t_max + 1, step = pas)

# === Conditions initiales === #
x0 = 0
y0 = 5
z0 = 5
initial_state = np.array([x0, y0, z0])

# === Simulation === #
solution = Methods.rk4(System.system, initial_state, t, mu, Q, epsilon)
print("point critique = ", solution[t_max])

# === Tracé des résultats === #
plt.figure(figsize=(10, 6))
# Courbes
plt.plot(t, solution[:, 0], "r", label = 'x(t) : biomasse')
plt.plot(t, solution[:, 1], "g", label = 'y(t) : polluant réacteur')
plt.plot(t, solution[:, 2], "b", label = 'z(t) : polluant lac')
# Point critique
plt.scatter(t_max, 0 , color='r', s = 100, marker='x', label = "x* = 0")
plt.scatter(t_max, z0, color='g', s = 100, marker='x', label = "y* = z0")
plt.scatter(t_max, z0, color='b', s = 100, marker='x', label = "z* = z0")
plt.xlabel('Temps')
plt.ylabel('Concentrations')
plt.title('Simulation du système pour µz0 > Q')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## Point critique 2 : $B = \begin{pmatrix} z0 - Q/\mu \\ Q/\mu \\ z0 \end{pmatrix} $
### Cas 1 : $\mu z_0 > Q$ -> test stabilité

# === Paramètres du modèle === #
mu = 0.5
Q = 1
epsilon = 0

# === Paramètres de simulation === #
t_min = 0
t_max = 4
pas = 1
t = np.arange(start = t_min, stop = t_max + 1, step = pas)

# === Conditions initiales === #
x0 = 2
y0 = 3
z0 = 5
initial_state = np.array([x0, y0, z0])

# === Simulation === #
solution = Methods.rk4(System.system, initial_state, t, mu, Q, epsilon)
print("point critique = ", solution[t_max])

# === Tracé des résultats === #
plt.figure(figsize=(10, 6))
# Courbes
plt.plot(t, solution[:, 0], "r", label = 'x(t) : biomasse')
plt.plot(t, solution[:, 1], "g", label = 'y(t) : polluant réacteur')
plt.plot(t, solution[:, 2], "b", label = 'z(t) : polluant lac')
# Point critique
plt.scatter(t_max, z0 - Q/mu , color='r', s = 100, marker='x', label = "x* = z0 - Q/mu")
plt.scatter(t_max, Q/mu, color='g', s = 100, marker='x', label = "y* = Q/mu")
plt.scatter(t_max, z0, color='b', s = 100, marker='x', label = "z* = z0")
plt.xlabel('Temps')
plt.ylabel('Concentrations')
plt.title('Simulation du système pour µz0 > Q')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### Cas 2 : $\mu z_0 = Q$ -> test stabilité

# === Paramètres du modèle === #
mu = 0.2
Q = 1
epsilon = 0

# === Paramètres de simulation === #
t_min = 0
t_max = 300
pas = 1
t = np.arange(start = t_min, stop = t_max + 1, step = pas)

# === Conditions initiales === #
x0 = 2
y0 = 3
z0 = 5
initial_state = np.array([x0, y0, z0])

# === Simulation === #
solution = Methods.rk4(System.system, initial_state, t, mu, Q, epsilon)
print("point critique = ", solution[t_max])

# === Tracé des résultats === #
plt.figure(figsize=(10, 6))
# Courbes
plt.plot(t, solution[:, 0], "r", label = 'x(t) : biomasse')
plt.plot(t, solution[:, 1], "g", label = 'y(t) : polluant réacteur')
plt.plot(t, solution[:, 2], "b", label = 'z(t) : polluant lac')
# Point critique
plt.scatter(t_max, z0 - Q/mu , color='r', s = 100, marker='x', label = "x* = z0 - Q/mu")
plt.scatter(t_max, Q/mu, color='g', s = 100, marker='x', label = "y* = Q/mu")
plt.scatter(t_max, z0, color='b', s = 100, marker='x', label = "z* = z0")
plt.xlabel('Temps')
plt.ylabel('Concentrations')
plt.title('Simulation du système pour µz0 = Q')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Cas $\epsilon > 0$
## Point critique : $C = \begin{pmatrix} 0 \\ y^* \\ y^* \end{pmatrix} $
### Cas 1 : $\mu y^* < Q$ -> test stabilité

# === Paramètres du modèle === #
mu = 0.1
Q = 1
epsilon = 0.1

# === Paramètres de simulation === #
t_min = 0
t_max = 30
pas = 1
t = np.arange(start = t_min, stop = t_max + 1, step = pas)

# === Conditions initiales === #
x0 = 2
y0 = 3
z0 = 5
initial_state = np.array([x0, y0, z0])

# === Simulation === #
solution = Methods.rk4(System.system, initial_state, t, mu, Q, epsilon)
print("point critique = ", solution[t_max])

# === Tracé des résultats === #
plt.figure(figsize=(10, 6))
# Courbes
plt.plot(t, solution[:, 0], "r", label = 'x(t) : biomasse')
plt.plot(t, solution[:, 1], "g", label = 'y(t) : polluant réacteur')
plt.plot(t, solution[:, 2], "b", label = 'z(t) : polluant lac')
# Point critique
plt.scatter(t_max, 0, color='r', s = 100, marker='x', label = "x* = 0")
plt.scatter(t_max, 5, color='g', s = 100, marker='x', label = "y* = 5")
plt.scatter(t_max, 5, color='b', s = 100, marker='x', label = "z* = 5")
plt.xlabel('Temps')
plt.ylabel('Concentrations')
plt.title('Simulation du système pour µy* < Q')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### Cas 2 : $\mu y^* = Q$ -> test stabilité

# === Paramètres du modèle === #
mu = 1/4.7
Q = 1
epsilon = 0.1

# === Paramètres de simulation === #
t_min = 0
t_max = 30
pas = 1
t = np.arange(start = t_min, stop = t_max + 1, step = pas)

# === Conditions initiales === #
x0 = 2
y0 = 3
z0 = 5
initial_state = np.array([x0, y0, z0])

# === Simulation === #
solution = Methods.rk4(System.system, initial_state, t, mu, Q, epsilon)
print("point critique = ", solution[t_max])

# === Tracé des résultats === #
plt.figure(figsize=(10, 6))
# Courbes
plt.plot(t, solution[:, 0], "r", label = 'x(t) : biomasse')
plt.plot(t, solution[:, 1], "g", label = 'y(t) : polluant réacteur')
plt.plot(t, solution[:, 2], "b", label = 'z(t) : polluant lac')
# Point critique
plt.scatter(t_max, 0, color='r', s = 100, marker='x', label = "x* = 0")
plt.scatter(t_max, 4.7, color='g', s = 100, marker='x', label = "y* = 4.7")
plt.scatter(t_max, 4.7, color='b', s = 100, marker='x', label = "z* = 4.7")
plt.xlabel('Temps')
plt.ylabel('Concentrations')
plt.title('Simulation du système pour µy* = Q')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### Cas 3 : $\mu y^* > Q$ -> test stabilité

# === Paramètres du modèle === #
mu = 1.25
Q = 1
epsilon = 0.1

# === Paramètres de simulation === #
t_min = 0
t_max = 5
pas = 1
t = np.arange(start = t_min, stop = t_max + 1, step = pas)

# === Conditions initiales === #
x0 = 2
y0 = 3
z0 = 5
initial_state = np.array([x0, y0, z0])

# === Simulation === #
solution = Methods.rk4(System.system, initial_state, t, mu, Q, epsilon)
print("evolution du point (x, y, z) : ", solution)

# === Tracé des résultats === #
plt.figure(figsize=(10, 6))
# Courbes
plt.plot(t, solution[:, 0], "r", label = 'x(t) : biomasse')
plt.plot(t, solution[:, 1], "g", label = 'y(t) : polluant réacteur')
plt.plot(t, solution[:, 2], "b", label = 'z(t) : polluant lac')
# Point critique
plt.scatter(t_max, 0, color='r', s = 100, marker='x', label = "x* = 0")
plt.scatter(t_max, 0.9, color='g', s = 100, marker='x', label = "y* = 0.9")
plt.scatter(t_max, 0.9, color='b', s = 100, marker='x', label = "z* = 0.9")
plt.xlabel('Temps')
plt.ylabel('Concentrations')
plt.title('Simulation du système pour µy* > Q')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## Point critique : $D = \begin{pmatrix} 0 \\ Q/ \mu \\ Q/ \mu \end{pmatrix} $ -> test stabilité

# === Paramètres du modèle ===
mu = 1.0
Q = 1.0
epsilon = 0.01

# === Paramètres de simulation ===
t_min = 0
t_max = 500
pas = 1
t = np.arange(start = t_min, stop = t_max + 1, step = pas)

# === Conditions initiales ===
x0 = 1.0   # biomasse initiale
y0 = 1.0   # concentration polluant dans le réacteur
z0 = 3.0   # concentration dans le lac
initial_state = np.array([x0, y0, z0])

# Résoudre
solution = Methods.rk4(System.system, initial_state, t, mu, Q, epsilon)
print("point critique = ", solution[t_max])

# === Tracé des résultats ===
plt.figure(figsize=(10, 6))
plt.plot(t, solution[:, 0], "r", label='x(t) - biomasse')
plt.plot(t, solution[:, 1], "g", label='y(t) - polluant réacteur')
plt.plot(t, solution[:, 2], "b",label='z(t) - polluant lac')
plt.xlabel('Temps')
plt.ylabel('Concentrations')
plt.title('Simulation du système')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Portrait de phase :
## On peut voir un portrait de phase pour le cas epsilon = 0, et le point singulier  (0,z0,z0)

# === Paramètres ===
mu = 1.0
Q = 1.0
epsilon = 0
t = np.linspace(0, 50, 1000)
dt = t[1] - t[0]

# === Conditions initiales ===
init_stable = np.array([1.0, 1.0, 0.7])     # µz0 < Q
init_limite = np.array([1.0, 1.0, 1.0])     # µz0 = Q
init_pas_stable = np.array([5.0, 3.0, 2.0])   # µz0 > Q

pt_stable = Points.point_critique_eps_egale_zero_B(init_stable[2])
pt_limite = Points.point_critique_eps_egale_zero_B(init_limite[2])
pt_pas_stable = Points.point_critique_eps_egale_zero_B(init_pas_stable[2])  

# === Simulation Euler explicite ===
sol_stable = Methods.rk4(System.system, init_stable, t, mu, Q, epsilon)
sol_limite = Methods.rk4(System.system, init_limite, t, mu, Q, epsilon)
sol_pas_stable = Methods.rk4(System.system, init_pas_stable, t, mu, Q, epsilon)

# === Flèche directionnelle ===
i_arrow = 300

# === Projections 2D ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

Plot.tracage_de_phase_stable_pas_stable_limite(sol_stable,sol_limite,sol_pas_stable,pt_stable, pt_limite, pt_pas_stable,i_arrow,axes,"RK4")
Plot.trajectoire_3D_stable_pas_stable_limite(sol_stable,sol_limite,sol_pas_stable,pt_stable, pt_limite, pt_pas_stable,i_arrow,axes,"RK4")

# Dépollution du lac de la manière la plus optimale
## On s'intéresse désormais à la dépollution la plus efficace du lac en jouant sur le débit Q, pour cela nous avons émis plusieurs hypothèses concernant notre système (cf rapport)
### Optimisation de Q, pour Q constant

mu = 1.0
epsilon = 0.01
z0 = 1.0
z1 = 0.2

# Trouver le minimum de T(Q)
result = minimize_scalar(Function.T, args=(z0, z1, mu, epsilon), bounds=(0.01, mu * z1 - 0.001), method='bounded')
Q_opt = result.x
T_opt = result.fun

print(f"Q optimal constant ≈ {Q_opt:.6f}")
print(f"T(Q_opt) ≈ {T_opt:.6f}")
print("On retrouve le Q constant optimal, renvoyant un temps minimum de dépollution. Donc au mieux, on pourra dépolluer le lac en 1889 jours")

# Tracé
t_vals = np.linspace(0, T_opt * 1.2, 300)
Q_values = [0.05, Q_opt, 0.19]  # Exemples de débits à comparer

plt.figure(figsize=(8,6))
for Q in Q_values:
    z_vals = Function.z_t(Q, z0, z1, mu, epsilon, t_vals)
    label = f"Q = {Q:.3f} (T ≈ {Function.T(Q, z0, z1, mu, epsilon):.2f})"
    plt.plot(t_vals, z_vals, label=label)

plt.axhline(z1, color='gray', linestyle='--', label='Seuil $z_1$')
plt.xlabel("Temps t")
plt.ylabel("Concentration $z(t)$")
plt.title("Évolution de $z(t)$ pour différents débits $Q$")
plt.legend()
plt.grid()
plt.show()

### Optimisation de Q, pour Q variant au cours du temps

# Paramètres du modèle
mu = 1.0
epsilon = 0.01
z0 = 1.0
z1 = 0.2
N = 10  # Nombre d'intervalles de contrôle

# Génération des seuils intermédiaires y_i
y_vals = np.linspace(z0, z1, N + 1)  # y0, y1, ..., yN = z1

# Initialisation des résultats
Q_list = []
T_list = []
z_list = [z0]
t_list = [0]

# Optimisation sur chaque intervalle [yi, yi+1]
for i in range(N):
    yi = y_vals[i]
    yi1 = y_vals[i + 1]

    # Trouver le Q_i qui minimise T_step
    result = minimize_scalar(lambda Q: Function.T_step(Q, mu, epsilon, yi, yi1, Function.y_inf), bounds=(1e-3, mu * yi1 - 1e-6), method='bounded')
    Q_opt = result.x
    T_opt = Function.T_step(Q_opt, mu, epsilon, yi, yi1, Function.y_inf)

    # Stockage
    Q_list.append(Q_opt)
    T_list.append(T_opt)
    t_list.append(t_list[-1] + T_opt)
    z_list.append(yi1)

# Conversion en arrays
t_array = np.array(t_list)
z_array = np.array(z_list)
Q_array = np.array(Q_list)

# Tracé de z(t)
plt.figure(figsize=(8,5))
plt.plot(t_array, z_array, marker='o', label='z(t)')
plt.axhline(z1, color='gray', linestyle='--', label='Seuil final $z_1$')
plt.xlabel("Temps t")
plt.ylabel("Concentration $z(t)$")
plt.title("Évolution de z(t) avec Q(t) variable optimal par sous-intervalles")
plt.grid()
plt.legend()
plt.show()

# Tracé de Q(t)
plt.figure(figsize=(8,5))
plt.step(t_array[:-1], Q_array, where='post', label='Q(t)', color='green')
plt.xlabel("Temps t")
plt.ylabel("Débit Q(t)")
plt.title("Débit optimal Q(t) par morceaux")
plt.grid()
plt.legend()
plt.show()

# Trouver le Q optimal global pour débit constant
res_cte = minimize_scalar(Function.T_global, args = (mu, epsilon, z0, z1), bounds=(1e-3, mu * z1 - 1e-6), method='bounded')
Q_cte = res_cte.x
T_cte = Function.T_global(Q_cte, mu, epsilon, z0, z1)


# Tracer z(t) avec Q constant optimal
t_fine = np.linspace(0, t_array[-1], 300)  # même durée que pour le Q variable
z_cte_vals = Function.z_t_constant(t_fine, Q_cte, mu, epsilon, z0)

plt.figure(figsize=(8,5))
plt.plot(t_array, z_array, marker='o', label='z(t) avec Q(t) variable')
plt.plot(t_fine, z_cte_vals, linestyle='--', label=f'z(t) avec Q constant (Q={Q_cte:.3f})')
plt.axhline(z1, color='gray', linestyle='--', label='Seuil final $z_1$')
plt.xlabel("Temps t")
plt.ylabel("Concentration z(t)")
plt.title("Comparaison : Q(t) variable vs Q constant optimal")
plt.grid()
plt.legend()
plt.show()

# Comparaison aux autres méthodes numériques

# === Paramètres ===
mu = 1.0
Q = 1.0
epsilon = 0
t = np.linspace(0, 100, 1000)

# === Conditions initiales ===
init_stable = np.array([10.0, 5.0, 3.0])     # µz > Q
init_limite = np.array([2, 4.0, 1.0])     # µz = Q

# === Système différentiel ===
sol_stable_explicite = Methods.euler_explicit(System.system, init_stable, t, mu, Q, epsilon)
sol_limite_explicite = Methods.euler_explicit(System.system, init_limite, t, mu, Q, epsilon)
sol_stable_rk4 = Methods.rk4(System.system, init_stable, t, mu, Q, epsilon)
sol_limite_rk4 = Methods.rk4(System.system, init_limite, t, mu, Q, epsilon)
sol_stable_implicite = Methods.euler_implicit(System.system, init_stable, t, mu, Q, epsilon)
sol_limite_implicite = Methods.euler_implicit(System.system, init_limite, t, mu, Q, epsilon)


#Point Critique
pt_stable = Points.point_critique_eps_egale_zero_A(mu, Q, init_stable[2])
pt_limite = Points.point_critique_eps_egale_zero_A(mu, Q, init_limite[2])

i_arrow = 30  # index pour position de la flèche

# === Projections 2D ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Configuration pour les 3 graphes
plot_configs = [
    {'x_idx': 0, 'y_idx': 1, 'xlabel': 'x (biomasse)', 'ylabel': 'y (réacteur)', 'title': 'y(x)'},
    {'x_idx': 0, 'y_idx': 2, 'xlabel': 'x (biomasse)', 'ylabel': 'z (lac)', 'title': 'z(x)'},
    {'x_idx': 1, 'y_idx': 2, 'xlabel': 'y (réacteur)', 'ylabel': 'z (lac)', 'title': 'z(y)'}
]
# Toutes les solutions et leurs styles associés
solutions = [
    (sol_stable_rk4, 'green', '-', 'mu.z0 > Q (RK4)'),
    (sol_limite_rk4, 'blue', '-', 'mu.z0 = Q (RK4)'),
    (sol_stable_implicite, 'green', '-.', 'mu.z0 > Q (Implicite)'),
    (sol_limite_implicite, 'blue', '-.', 'mu.z0 = Q (Implicite)'),
    (sol_stable_explicite, 'green', '--', 'mu.z0 > Q (Explicite)'),
    (sol_limite_explicite, 'blue', '--', 'mu.z0 = Q (Explicite)'),
   
]
# Points spéciaux à scatter
scatter_points = [
    (pt_stable, 'green'),
    (pt_limite, 'blue'),
]
# Boucle principale sur les axes
for idx, config in enumerate(plot_configs):
    ax = axes[idx]
    for sol, color, linestyle, label in solutions:
        ax.plot(sol[:, config['x_idx']], sol[:, config['y_idx']], linestyle, color=color, label=label)
        # Flèche pour RK4 uniquement
        if 'RK4' in label:
            ax.annotate('', 
                        xy=(sol[i_arrow+1, config['x_idx']], sol[i_arrow+1, config['y_idx']]),
                        xytext=(sol[i_arrow, config['x_idx']], sol[i_arrow, config['y_idx']]),
                        arrowprops=dict(arrowstyle="->", color=color))
    # Scatter points
    for pt, color in scatter_points:
        ax.scatter(pt[config['x_idx']], pt[config['y_idx']], color=color, marker='x', s=80)

    # Labels et titre
    ax.set_xlabel(config['xlabel'])
    ax.set_ylabel(config['ylabel'])
    ax.set_title(config['title'])
    ax.legend()
plt.tight_layout()
plt.show()

# === Trajectoire 3D ===
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol_stable_rk4[:, 0], sol_stable_rk4[:, 1], sol_stable_rk4[:, 2], label='mu.z0 > Q (RK4)', linestyle = '-', color='green')
ax.plot(sol_limite_rk4[:, 0], sol_limite_rk4[:, 1], sol_limite_rk4[:, 2], label='mu.z0 = Q (RK4)', linestyle = '-', color='blue')
ax.plot(sol_stable_implicite[:, 0], sol_stable_implicite[:, 1], sol_stable_implicite[:, 2], label='mu.z0 > Q (Implicite)', linestyle = '-.', color='green')
ax.plot(sol_limite_implicite[:, 0], sol_limite_implicite[:, 1], sol_limite_implicite[:, 2], label='mu.z0 = Q (Implicite)', linestyle = '-.', color='blue')
ax.plot(sol_stable_explicite[:, 0], sol_stable_explicite[:, 1], sol_stable_explicite[:, 2], label='mu.z0 > Q (Explicite)', linestyle = '--', color='green')
ax.plot(sol_limite_explicite[:, 0], sol_limite_explicite[:, 1], sol_limite_explicite[:, 2], label='mu.z0 = Q (Explicite)', linestyle = '--', color='blue')

# Points critiques
ax.scatter(*pt_stable, color='green', marker='x', s=80)
ax.scatter(*pt_limite, color='blue', marker='x', s=80)

# Flèche 3D sur cas limite
dx = sol_limite_rk4[i_arrow+1, 0] - sol_limite_rk4[i_arrow, 0]
dy = sol_limite_rk4[i_arrow+1, 1] - sol_limite_rk4[i_arrow, 1]
dz = sol_limite_rk4[i_arrow+1, 2] - sol_limite_rk4[i_arrow, 2]
ax.quiver(sol_limite_rk4[i_arrow, 0], sol_limite_rk4[i_arrow, 1], sol_limite_rk4[i_arrow, 2],
          dx, dy, dz, color='black', arrow_length_ratio=0.1)

ax.set_xlabel('x (biomasse)')
ax.set_ylabel('y (réacteur)')
ax.set_zlabel('z (lac)')
ax.set_title('Trajectoires 3D comparées avec flèches et points critiques')
ax.legend()
plt.tight_layout()
plt.show()

# Ordre de convergence des méthodes numériques

# Paramètres du modèle
mu = 1.0
epsilon = 0.01
z0_init = 1.0
y0_init = 0.0
x0_init = 0.1
tf = 5.0
Q = 0.1
u0 = np.array([x0_init, y0_init, z0_init])

# Listes des pas de temps
h_list = [0.4, 0.2, 0.1, 0.05, 0.025]
errors_euler = []
errors_rk4 = []

for h in h_list:
    errors_euler.append(Methods.compare_h_h2(System.system, Methods.euler_system, u0, h, tf, mu, Q, epsilon))
    errors_rk4.append(Methods.compare_h_h2(System.system, Methods.rk4_system, u0, h, tf, mu, Q, epsilon))

# Affichage du graphe log-log
plt.figure(figsize=(8,6))
plt.loglog(h_list, errors_euler, 'o-', label='Euler (ordre 1)')
plt.loglog(h_list, errors_rk4, 's-', label='RK4 (ordre 4)')
X = np.linspace(0.025,0.4,1000)
plt.loglog(X, X, label='X')
plt.loglog(X, X**4, label='X^4')

plt.xlabel("Pas de temps h")
plt.ylabel("Erreur ||u_h - u_{h/2}||")
plt.title("Estimation numérique de l'ordre de convergence")
plt.grid(which="both")
plt.legend()
plt.show()

# Paramètres du modèle
mu = 20
VR = 4.5e5 # 50 * 100 * 90
VL = 4.5e9
epsilon = VR/VL
z0 = 1.0
z1 = 0.2
N = 10  # Nombre d'intervalles de contrôle

# Génération des seuils intermédiaires y_i
y_vals = np.linspace(z0, z1, N + 1)  # y0, y1, ..., yN = z1

# Initialisation des résultats
Q_list = []
T_list = []
z_list = [z0]
t_list = [0]

# Optimisation sur chaque intervalle [yi, yi+1]
for i in range(N):
    yi = y_vals[i]
    yi1 = y_vals[i + 1]

    # Trouver le Q_i qui minimise T_step
    result = minimize_scalar(lambda Q: Function.T_step(Q, mu, epsilon, yi, yi1, Function.y_inf), bounds=(1e-3, mu * yi1 - 1e-6), method='bounded')
    Q_opt = result.x
    T_opt = Function.T_step(Q_opt, mu, epsilon, yi, yi1, Function.y_inf)

    # Stockage
    Q_list.append(Q_opt)
    T_list.append(T_opt)
    t_list.append(t_list[-1] + T_opt)
    z_list.append(yi1)

# Conversion en arrays
t_array = np.array(t_list)
z_array = np.array(z_list)
Q_array = np.array(Q_list)

# Tracé de z(t)
plt.figure(figsize=(8,5))
plt.plot(t_array, z_array, marker='o', label='z(t)')
plt.plot(t_array, t_array*0 + 0.5, '--', label = 'Lac dépollué de moitié')
plt.xlabel("Temps (jours)")
plt.ylabel("Concentration $z(t)$ (mg/L)")
plt.title("Dépollution du lac Thaï (Chine) avec débit variable")
plt.grid()
plt.legend()
plt.show()