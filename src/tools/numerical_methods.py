import numpy as np

class Methods :
    # === Implémentation des différentes méthodes ===
    # RK4
    def rk4(f, y0, t, mu, Q, epsilon):
        n = len(t)
        y = np.zeros((n, len(y0)))
        y[0] = y0
        for i in range(n - 1):
            h = t[i+1] - t[i]
            k1 = f(t[i], y[i], mu, Q, epsilon)
            k2 = f(t[i] + h/2, y[i] + h/2 * k1, mu, Q, epsilon)
            k3 = f(t[i] + h/2, y[i] + h/2 * k2, mu, Q, epsilon)
            k4 = f(t[i] + h, y[i] + h * k3, mu, Q, epsilon)
            y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        return y
    
    # RK4 explicite
    def rk4_system(f, u0, h, tf, mu, Q, epsilon):
        N = int(tf / h)
        t = np.linspace(0, N * h, N + 1)
        u = np.zeros((N + 1, len(u0)))
        u[0] = u0
        for n in range(N):
            k1 = f(t[n], u[n], mu, Q, epsilon)
            k2 = f(t[n] + h/2, u[n] + h/2 * k1, mu, Q, epsilon)
            k3 = f(t[n] + h/2, u[n] + h/2 * k2, mu, Q, epsilon)
            k4 = f(t[n] + h, u[n] + h * k3, mu, Q, epsilon)
            u[n+1] = u[n] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        return t, u

    # Euler Explicite
    def euler_explicit(f, y0, t, mu, Q, epsilon):
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        for i in range(len(t) - 1):
            dt = t[i+1] - t[i]
            y[i+1] = y[i] + dt * f(t[i], y[i], mu, Q, epsilon)
        return y
    
    # Euler explicite (avec maillage correct)
    def euler_system(f, u0, h, tf, mu, Q, epsilon):
        N = int(tf / h)
        t = np.linspace(0, N * h, N + 1)
        u = np.zeros((N + 1, len(u0)))
        u[0] = u0
        for n in range(N):
            u[n + 1] = u[n] + h * f(t[n], u[n], mu, Q, epsilon)
        return t, u

    # Euler Implicite
    def euler_implicit(f, y0, t, mu, Q, epsilon, max_iter=10, tol=1e-6):
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        dt = t[1] - t[0]
        for i in range(len(t) - 1):
            yn = y[i]
            yn1 = yn.copy()
            for _ in range(max_iter):
                yn1_new = yn + dt * f(t[i+1], yn1, mu, Q, epsilon)
                if np.linalg.norm(yn1_new - yn1) < tol:
                    break
                yn1 = yn1_new
            y[i+1] = yn1
        return y
    
    # Comparaison entre pas h et h/2
    def compare_h_h2(f, method_solver, u0, h, tf, mu, Q, epsilon):
        t_h, u_h = method_solver(f, u0, h, tf, mu, Q, epsilon)
        t_h2, u_h2 = method_solver(f, u0, h/2, tf, mu, Q, epsilon)
        u_h2_interp = u_h2[::2]  # Sous-échantillonnage
        error = np.linalg.norm(u_h - u_h2_interp, ord=np.inf)
        return error