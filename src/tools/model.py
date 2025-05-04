import numpy as np

class System :
    # === Système différentiel ===
    def system(t, state, mu, Q, epsilon):
        x, y, z = state
        dx = mu * x * y - Q * x
        dy = -mu * x * y + Q * (z - y)
        dz = epsilon * Q * (y - z)
        return np.array([dx, dy, dz])