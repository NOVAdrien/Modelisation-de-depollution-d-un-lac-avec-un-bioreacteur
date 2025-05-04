import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Plot :
    #Trace les portraits de phase de x , y ,z selon les cas stables, pas_stables et limites 
    def tracage_de_phase_stable_pas_stable_limite(sol_stable,sol_limite,sol_pas_stable,pt_stable, pt_limite, pt_pas_stable,i_arrow,axes,title=""):
        axes[0].plot(sol_stable[:, 0], sol_stable[:, 1], color='green', label='mu.z0 < Q')
        axes[0].plot(sol_limite[:, 0], sol_limite[:, 1], color='orange', label='mu.z0 = Q')
        axes[0].plot(sol_pas_stable[:, 0], sol_pas_stable[:, 1], color='red', label='mu.z0 > Q')
        axes[0].scatter(0, pt_stable[1], color='green', marker='x', s=80)
        axes[0].scatter(0, pt_limite[1], color='orange', marker='x', s=80)
        axes[0].scatter(0, pt_pas_stable[1], color='red', marker='x', s=80)
        axes[0].annotate('', xy=(sol_stable[i_arrow+1, 0], sol_stable[i_arrow+1, 1]),
                        xytext=(sol_stable[i_arrow, 0], sol_stable[i_arrow, 1]),
                        arrowprops=dict(arrowstyle="->", color='black'))
        axes[0].annotate('', xy=(sol_pas_stable[i_arrow+1, 0], sol_pas_stable[i_arrow+1, 1]),
                        xytext=(sol_pas_stable[i_arrow, 0], sol_pas_stable[i_arrow, 1]),
                        arrowprops=dict(arrowstyle="->", color='black'))
        axes[0].annotate('', xy=(sol_limite[i_arrow+1, 0], sol_limite[i_arrow+1, 1]),
                        xytext=(sol_limite[i_arrow, 0], sol_limite[i_arrow, 1]),
                        arrowprops=dict(arrowstyle="->", color='black'))
        axes[0].set_xlabel('x (biomasse)')
        axes[0].set_ylabel('y (réacteur)')
        axes[0].set_title('y(x) - ' +title)
        axes[0].legend()
        # z vs x
        axes[1].plot(sol_stable[:, 0], sol_stable[:, 2], color='green', label='mu.z0 < Q')
        axes[1].plot(sol_limite[:, 0], sol_limite[:, 2], color='orange', label='mu.z0 = Q')
        axes[1].plot(sol_pas_stable[:, 0], sol_pas_stable[:, 2], color='red', label='mu.z0 > Q')
        axes[1].scatter(0, pt_stable[2], color='green', marker='x', s=80)
        axes[1].scatter(0, pt_limite[2], color='orange', marker='x', s=80)
        axes[1].scatter(0, pt_pas_stable[2], color='red', marker='x', s=80)
        axes[1].annotate('', xy=(sol_stable[i_arrow+1, 0], sol_stable[i_arrow+1, 2]),
                        xytext=(sol_stable[i_arrow, 0], sol_stable[i_arrow, 2]),
                        arrowprops=dict(arrowstyle="->", color='black'))
        axes[1].annotate('', xy=(sol_limite[i_arrow+1, 0], sol_limite[i_arrow+1, 2]),
                        xytext=(sol_limite[i_arrow, 0], sol_limite[i_arrow, 2]),
                        arrowprops=dict(arrowstyle="->", color='black'))
        axes[1].annotate('', xy=(sol_pas_stable[i_arrow+1, 0], sol_pas_stable[i_arrow+1, 2]),
                        xytext=(sol_pas_stable[i_arrow, 0], sol_pas_stable[i_arrow, 2]),
                        arrowprops=dict(arrowstyle="->", color='black'))
        axes[1].set_xlabel('x (biomasse)')
        axes[1].set_ylabel('z (lac)')
        axes[1].set_title('z(x) - '+title)
        axes[1].legend()
        # z vs y
        axes[2].plot(sol_stable[:, 1], sol_stable[:, 2], color='green', label='mu.z0 < Q')
        axes[2].plot(sol_limite[:, 1], sol_limite[:, 2], color='orange', label='mu.z0 = Q')
        axes[2].plot(sol_pas_stable[:, 1], sol_pas_stable[:, 2], color='red', label = 'mu.z0 > Q')
        axes[2].scatter(pt_stable[1], pt_stable[2], color='green', marker='x', s=80)
        axes[2].scatter(pt_limite[1], pt_limite[2], color='orange', marker='x', s=80)
        axes[2].scatter(pt_pas_stable[1], pt_pas_stable[2], color='red', marker='x', s=80)
        axes[2].annotate('', xy=(sol_stable[i_arrow+1, 1], sol_stable[i_arrow+1, 2]),
                        xytext=(sol_stable[i_arrow, 1], sol_stable[i_arrow, 2]),
                        arrowprops=dict(arrowstyle="->", color='black'))
        axes[2].annotate('', xy=(sol_limite[i_arrow+1, 1], sol_limite[i_arrow+1, 2]),
                        xytext=(sol_limite[i_arrow, 1], sol_limite[i_arrow, 2]),
                        arrowprops=dict(arrowstyle="->", color='black'))
        axes[2].annotate('', xy=(sol_pas_stable[i_arrow+1, 1], sol_pas_stable[i_arrow+1, 2]),
                        xytext=(sol_pas_stable[i_arrow, 1], sol_pas_stable[i_arrow, 2]),
                        arrowprops=dict(arrowstyle="->", color='black'))
        axes[2].set_xlabel('y (réacteur)')
        axes[2].set_ylabel('z (lac)')
        axes[2].set_title('z(y) - '+title)
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()

    #Renvoie les trajectoires 3D des cas stables, pas_stable,limites
    def trajectoire_3D_stable_pas_stable_limite(sol_stable,sol_limite,sol_pas_stable,pt_stable, pt_limite, pt_pas_stable,i_arrow,axes,title=""):

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(sol_stable[:, 0], sol_stable[:, 1], sol_stable[:, 2], label='mu.z0 < Q', color='green')
        ax.plot(sol_limite[:, 0], sol_limite[:, 1], sol_limite[:, 2], label='mu.z0 = Q ', color='orange')
        ax.plot(sol_pas_stable[:, 0], sol_pas_stable[:, 1], sol_pas_stable[:, 2], label='mu.z0 > Q ', color='red')
        ax.scatter(*pt_stable, color='green', marker='x', s=80)
        ax.scatter(*pt_limite, color='orange', marker='x', s=80)
        ax.scatter(*pt_pas_stable, color='red', marker='x', s=80)
        
        # Flèche 3D sur cas limite
        dx = sol_limite[i_arrow+1, 0] - sol_limite[i_arrow, 0]
        dy = sol_limite[i_arrow+1, 1] - sol_limite[i_arrow, 1]
        dz = sol_limite[i_arrow+1, 2] - sol_limite[i_arrow, 2]
        ax.quiver(sol_limite[i_arrow, 0], sol_limite[i_arrow, 1], sol_limite[i_arrow, 2],
                dx, dy, dz, color='black', arrow_length_ratio=0.1)
        
        ax.set_xlabel('x (biomasse)')
        ax.set_ylabel('y (réacteur)')
        ax.set_zlabel('z (lac)')
        ax.set_title('Trajectoires 3D - '+title)
        ax.legend()
        plt.tight_layout()
        plt.show()