from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, multivariate_normal


def calc_U(x, y, q, a=0, b=1.0, c=0, d=1.0):
    mu_x, mu_y = q

    U = np.sum(norm.logpdf(y, loc=mu_y, scale=1) +
               norm.logpdf(x, loc=mu_x, scale=1) +
               norm.logpdf(mu_y, loc=a, scale=b) +
               norm.logpdf(mu_x, loc=c, scale=d))
    return -U


def grad_U(x, y, q, a=0, b=1.0, c=0, d=1.0):
    mu_x = q[0]
    mu_y = q[1]

    grad_mu_x = np.sum(x - mu_x) - (mu_x - a) / b ** 2
    grad_mu_y = np.sum(y - mu_y) - (mu_y - c) / d ** 2

    return np.array([-1 * grad_mu_x, -1 * grad_mu_y])


def HMC2(U: Callable, grad_U: Callable, epsilon: float, L: int, current_q: np.array, x: np.array, y: np.array):
    """
        This function runs a single trajectory and produces one sample. To build a chain of samples,
        call this function repeatedly.
    """
    q = current_q
    # random flick - p is momentum
    p = np.random.normal(loc=0, scale=1, size=len(q))
    current_p = p

    # Make a half step for momentum at the beginning
    p = p - 0.5 * epsilon * grad_U(x, y, q)

    # initialize bookkeeping - saves trajectory
    q_traj = np.full(shape=(L + 1, len(q)), fill_value=np.nan)
    p_traj = q_traj.copy()

    q_traj[0, :] = current_q
    p_traj[0, :] = p

    # Code 9.9 starts here
    for i in range(L):
        q = q + epsilon * p  # Full step for the position
        q_traj[i + 1, :] = q

        # Make a full step for the momentum, except at the end of trajectory
        if i != L - 1:
            p = p - epsilon * grad_U(x, y, q)
            p_traj[i + 1, :] = p

    # Make a half step for momentum at the end
    p = p - 0.5 * epsilon * grad_U(x, y, q)
    p_traj[L, :] = p

    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -1 * p

    # Evaluate potential and kinetic energies at start and end of trajectory
    current_U = U(x, y, current_q)
    # mass = 1 unit so that KE = 0.5 * p**2
    current_K = np.sum(current_p ** 2) / 2.0
    proposed_U = U(x, y, q)
    proposed_K = np.sum(p ** 2) / 2

    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    accept = False

    if np.log(np.random.uniform()) < current_U - proposed_U + current_K - proposed_K:
        # accept
        new_q = q
        accept = True
    else:
        # reject
        new_q = current_q

    return dict(q=new_q, traj=q_traj, p_traj=p_traj, accept=accept)


def main():
    np.random.seed(42)

    # test data
    real = multivariate_normal([0, 0], np.identity(2))
    x, y = real.rvs(50).T

    Q = {}
    Q["q"] = np.array([-0.1, 0.2])
    pr = 0.3
    step = 0.03
    # L = 11  # 0.03 / 28 for U-turns -- 11 for working example
    n_samples = 4

    _, axs = plt.subplots(1, 2, figsize=[8, 6], constrained_layout=True)

    for L, ax in zip([11, 28], axs):
        ax.scatter(*Q["q"], color="k", marker="x", zorder=3)

        if L == 11:
            ax.text(*Q["q"] + 0.015, "start", weight="bold")

        for i in range(n_samples):
            Q = HMC2(calc_U, grad_U, step, L, Q["q"], x, y)
            ax.scatter(*Q["q"], color="w", marker="o", edgecolor="k", lw=2, zorder=3)

            if n_samples < 10:
                for j in range(L):
                    K0 = np.sum(Q["p_traj"][j, :] ** 2) / 2  # kinetic energy
                    ax.plot(
                        Q["traj"][j: j + 2, 0],
                        Q["traj"][j: j + 2, 1],
                        color="k",
                        lw=1 + 1 * K0,
                        alpha=0.3,
                        zorder=1,
                    )

                ax.scatter(*Q["traj"].T, facecolor="w", edgecolor="gray", lw=1, zorder=2, s=10)

                if L == 11:
                    ax.text(*Q["q"] + [0.02, -0.03], f"{i + 1}", weight="bold")

        ax.set_title(f"2D Gaussian, L = {L}")
        ax.set_xlabel("mux")
        ax.set_ylabel("muy")

        # draw background contours based on real probability defined above
        ax.set_xlim(-pr, pr)
        ax.set_ylim(-pr, pr)
        xs, ys = np.mgrid[-pr:pr:0.01, -pr:pr:0.01]
        p = real.logpdf(np.vstack([xs.flat, ys.flat]).T).reshape(xs.shape)
        ax.contour(xs, ys, p, 4, colors=[(0, 0, 0, 0.3)])
        ax.set_aspect(1)

        plt.savefig('trajectory.png')


if __name__ == '__main__':
    main()
