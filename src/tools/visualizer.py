import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer:
    def __init__(self, t):
        self.t = t

    def plot_xyz(self, sol, label=None, linestyle='-', linewidth=2):
        """Trace x(t), y(t), z(t)"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(self.t, sol[:, 0], linestyle=linestyle, linewidth=linewidth, label=label)
        axes[0].set_ylabel('x(t)')
        axes[0].legend() if label else None

        axes[1].plot(self.t, sol[:, 1], linestyle=linestyle, linewidth=linewidth)
        axes[1].set_ylabel('y(t)')

        axes[2].plot(self.t, sol[:, 2], linestyle=linestyle, linewidth=linewidth)
        axes[2].set_ylabel('z(t)')
        axes[2].set_xlabel('Time')
        fig.suptitle(f'Trajectories over time', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_3d(self, sol, label='trajectory', color='blue', arrow_index=300, critical_point=None):
        """Affiche une trajectoire 3D avec flèche + point critique éventuel"""
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], label=label, color=color)
        ax.set_xlabel('x (biomass)')
        ax.set_ylabel('y (reactor)')
        ax.set_zlabel('z (lake)')
        ax.set_title('3D Trajectory')

        # Flèche directionnelle
        p0 = sol[arrow_index]
        p1 = sol[arrow_index + 1]
        arrow = p1 - p0
        ax.quiver(*p0, *arrow, color='black', arrow_length_ratio=0.1)

        # Point critique
        if critical_point is not None:
            ax.scatter(*critical_point, marker='x', s=100, color='red', label='Critical point')

        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_projections(self, sol_list, labels, colors, critical_points=None, arrow_index=300):
        """Compare plusieurs solutions sur les projections y(x), z(x), z(y)"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for sol, label, color in zip(sol_list, labels, colors):
            # y(x)
            axes[0].plot(sol[:, 0], sol[:, 1], label=label, color=color)
            axes[0].annotate('', xy=(sol[arrow_index+1, 0], sol[arrow_index+1, 1]),
                             xytext=(sol[arrow_index, 0], sol[arrow_index, 1]),
                             arrowprops=dict(arrowstyle="->", color='black'))
            # z(x)
            axes[1].plot(sol[:, 0], sol[:, 2], color=color)
            axes[1].annotate('', xy=(sol[arrow_index+1, 0], sol[arrow_index+1, 2]),
                             xytext=(sol[arrow_index, 0], sol[arrow_index, 2]),
                             arrowprops=dict(arrowstyle="->", color='black'))
            # z(y)
            axes[2].plot(sol[:, 1], sol[:, 2], color=color)
            axes[2].annotate('', xy=(sol[arrow_index+1, 1], sol[arrow_index+1, 2]),
                             xytext=(sol[arrow_index, 1], sol[arrow_index, 2]),
                             arrowprops=dict(arrowstyle="->", color='black'))

        if critical_points is not None:
            for pt, color in zip(critical_points, colors):
                axes[0].scatter(0, pt[1], color=color, marker='x', s=80)
                axes[1].scatter(0, pt[2], color=color, marker='x', s=80)
                axes[2].scatter(pt[1], pt[2], color=color, marker='x', s=80)

        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title('y(x)')
        axes[0].legend()

        axes[1].set_xlabel('x')
        axes[1].set_ylabel('z')
        axes[1].set_title('z(x)')

        axes[2].set_xlabel('y')
        axes[2].set_ylabel('z')
        axes[2].set_title('z(y)')

        plt.tight_layout()
        plt.show()
