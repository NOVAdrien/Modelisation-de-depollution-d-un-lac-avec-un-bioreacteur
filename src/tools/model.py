import numpy as np

# === Système différentiel ===
def system(t, state, mu=1.0, Q=1.0, epsilon=0.1):
    x, y, z = state
    dx = mu * x * y - Q * x
    dy = -mu * x * y + Q * (z - y)
    dz = epsilon * Q * (y - z)
    return np.array([dx, dy, dz])

# === Jacobienne du système ===
def jacobian(state, mu=1.0, Q=1.0, epsilon=0.1):
    x, y, z = state
    return np.array([
        [mu * y - Q, mu * x, 0],
        [-mu * y, -mu * x - Q, Q],
        [0, epsilon * Q, -epsilon * Q]
    ])
