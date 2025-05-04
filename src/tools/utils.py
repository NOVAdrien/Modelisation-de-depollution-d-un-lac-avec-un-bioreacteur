import numpy as np

class Points :
    #Renvoie le point critique pour le cas (0,z0,z0) 
    def point_critique_eps_egale_zero_B(z0):
        return (0,z0, z0)
    #Renvoie le point critique pour le cas (z0-Q/mu,Q/mu,z0) 
    def point_critique_eps_egale_zero_A(mu, Q, z0):
        return (z0 - Q/mu, Q / mu, z0)

class Function :
    #Fonction Q(t)
    def Q_func_log(t):
        Q0 = 2.0      # débit initial
        a = 1.0       # paramètre de décroissance
        return Q0 / (1 + a * np.log(1 + t))
    
    # Fonction T(Q)
    def T(Q, z0, z1, mu, epsilon):
        if Q <= 0 or Q >= mu * z1:
            return np.inf
        try:
            return -(1 / (epsilon * Q)) * np.log((z1 - Q / mu) / (z0 - Q / mu))
        except:
            return np.inf
    
    # Fonction z(t) pour un Q donné
    def z_t(Q, z0, z1, mu, epsilon, t):
        return (Q / mu) + (z0 - Q / mu) * np.exp(-epsilon * Q * t)
    
    # Fonction y_inf(Q) = Q / mu
    def y_inf(Q, mu):
        return Q / mu
    
    # Temps nécessaire pour passer de yi à yi+1 à débit Q
    def T_step(Q, mu, epsilon, yi, yi1, y_inf):
        yinf = y_inf(Q, mu)
        num = yi - yinf
        den = yi1 - yinf
        if Q <= 0 or den <= 0 or num <= 0:
            return np.inf
        try:
            return -(1 / (epsilon * Q)) * np.log(den / num)
        except:
            return np.inf
    
    # Fonction z(t) pour Q constant optimal
    def z_t_constant(t, Q, mu, epsilon, z_init):
        yinf = Q / mu
        return yinf + (z_init - yinf) * np.exp(-epsilon * Q * t)
    
    # Fonction T_global(Q) = temps total pour passer de z0 à z1 avec Q constant
    def T_global(Q, mu, epsilon, z0, z1):
        yinf = Q / mu
        if Q <= 0 or yinf >= z0 or yinf >= z1:
            return np.inf
        try:
            return -(1 / (epsilon * Q)) * np.log((z1 - yinf) / (z0 - yinf))
        except:
            return np.inf