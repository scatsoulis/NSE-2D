# !/usr/bin/env python3

"""
Created by Sotiris Catsoulis
Outline written by (upd. 10/1/2021):
    Greg Stroot, gstroot@caltech.edu
    Jane Bae (Original code outline)
Copyright 2022 California Institute of Technology
"""

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 14
plt.rcParams["legend.fontsize"] = 14
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def compute_projection_step(
    u_star: (float, float),
    v_star: (float, float),
    dx_fwd: (float, float),
    dx_back: (float, float),
    dy_fwd: (float, float),
    dy_back: (float, float),
    kappa_x_mod_sq: (float, float),
    kappa_y_mod_sq: (float, float),
) -> float:
    """
    Purpose:
        Run projection step to compute pseudo pressure and output next velocities.

    Inputs:
        u_star   - type: float64
                 - pseudo x velocity
        v_star   - type: float64
                 - pseudo y velocity
    Outputs:
        u_next  - type: float64
                - x velocity at next time-step
        v_next  - type: float64
                - y velocity at next time-step
    """

    Nx, Ny = np.shape(u_star)[0], np.shape(u_star)[1]  # noqa: N806

    # Compute (cell-centered) divergence of starred velocities
    du_star_dx, dv_star_dy = np.zeros_like(u_star), np.zeros_like(v_star)
    for j in range(Ny):
        du_star_dx[:, j] = np.matmul(dx_fwd, u_star[:, j])
    for i in range(Nx):
        dv_star_dy[i, :] = np.matmul(dy_fwd, v_star[i, :])
    rhs_p = du_star_dx + dv_star_dy

    # rhs_p = rhs_p

    # Solve Poisson equation
    p = poisson_solver(rhs_p, kappa_x_mod_sq, kappa_y_mod_sq)

    # Compute first derivatives of pseudo-pressure
    dp_dx, dp_dy = np.zeros_like(u_star), np.zeros_like(v_star)
    for j in range(Ny):
        dp_dx[:, j] = np.matmul(dx_back, p[:, j])
    for i in range(Nx):
        dp_dy[i, :] = np.matmul(dy_back, p[i, :])

    return dp_dx, dp_dy


def poisson_solver(
    rhs_p: (float, float),
    kappa_x_mod_sq: (float, float),
    kappa_y_mod_sq: (float, float),
) -> float:
    """
    Goal: Solve the Poisson equation

    Inputs:
    rhs_p           - type: float64
                    - Right-hand-side of poisson equation
    kappa_x_mod_sq  - type: float64
                    - Modified wavenumber in x
    kappa_y_mod_sq  - type: float64
                    - Modified wavenumber in y

    Outputs:
    p           - type: float64
                - Pseudo-pressure
    """

    # Fourier transform of rhs_p
    rhs_p_hat = np.fft.fft2(rhs_p)

    # Solve Poisson equation in Fourier space
    p_hat = np.divide(
        -rhs_p_hat,
        kappa_x_mod_sq + kappa_y_mod_sq,
    )  # division by 0 yields error (ok)
    p_hat[0, 0] = 0.1  # artificial fix

    # Inverse Fourier transform of pseudo-pressure
    p = np.real(np.fft.ifft2(p_hat))

    return p  # noqa RET504


if __name__ == "__main__":
    # TESTING BLOCK

    # Spatial domain and resolution
    Nx = 64
    Ny = Nx
    DOMAIN_X, DOMAIN_Y = 2 * np.pi, 2 * np.pi
    dx, dy = DOMAIN_X / Nx, DOMAIN_Y / Ny

    # 2-D staggered construction grids (for nodes, v, u)
    grid_x, grid_y = np.arange(0.0, DOMAIN_X, dx), np.arange(0.0, DOMAIN_Y, dy)
    grid_x_stagg, grid_y_stagg = (
        np.arange(0.0 + dx / 2, DOMAIN_X + dx / 2, dx),
        np.arange(0.0 + dy / 2, DOMAIN_Y + dy / 2, dy),
    )

    # 2-D grid staggered up (for nodes, u, v, pressure)
    grid_xy_x, grid_xy_y = np.meshgrid(grid_x, grid_y, indexing="ij")
    grid_u_x, grid_u_y = np.meshgrid(grid_x, grid_y_stagg, indexing="ij")
    grid_v_x, grid_v_y = np.meshgrid(grid_x_stagg, grid_y, indexing="ij")
    grid_p_x, grid_p_y = np.meshgrid(grid_x_stagg, grid_y_stagg, indexing="ij")

    # Single derivative forwards, backwards (x, y)
    kxfwd, kxback = [-np.ones(Nx), np.ones(Nx - 1)], [-np.ones(Nx - 1), np.ones(Nx)]
    offset_xfwd, offset_xback = [0, 1], [-1, 0]
    dx_fwd, dx_back = (
        diags(kxfwd, offset_xfwd).toarray(),
        diags(kxback, offset_xback).toarray(),
    )
    dx_fwd, dx_back = (1 / dx) * dx_fwd, (1 / dx) * dx_back
    kyfwd, kyback = [-np.ones(Ny), np.ones(Ny - 1)], [-np.ones(Ny - 1), np.ones(Ny)]
    offset_yfwd, offset_yback = [0, 1], [-1, 0]
    dy_fwd, dy_back = (
        diags(kyfwd, offset_yfwd).toarray(),
        diags(kyback, offset_yback).toarray(),
    )
    dy_fwd[-1, 0], dy_back[0, -1] = 1, -1
    dy_fwd, dy_back = (1 / dy) * dy_fwd, (1 / dy) * dy_back

    # Define test functions
    def u_init(x: float, y: float) -> float:
        return np.cos(x)

    def v_init(x: float, y: float) -> float:
        return np.sin(y)

    # Initialize random field
    u_star, v_star = np.zeros_like(grid_u_x), np.zeros_like(grid_v_y)
    for i in range(Nx):
        for j in range(Ny):
            u_star[i, j] = u_init(grid_u_x[i, j], grid_u_y[i, j])
            v_star[i, j] = v_init(grid_v_x[i, j], grid_v_y[i, j])

    # Run projection step
    u_next, v_next = compute_projection_step(
        u_star,
        v_star,
        dx_fwd,
        dx_back,
        dy_fwd,
        dy_back,
    )

    # Check zero divergence
    du_next_dx, dv_next_dy = np.zeros_like(u_next), np.zeros_like(u_next)
    for j in range(Ny):
        du_next_dx[:, j] = np.matmul(dx_fwd, u_next[:, j])
    for i in range(Nx):
        dv_next_dy[i, :] = np.matmul(dy_fwd, v_next[i, :])
    div_final = du_next_dx + dv_next_dy
    print("div_glob_final: ", div_final)

    # Plot results
    fig, ax1 = plt.subplots()
    image_truth = ax1.imshow(div_final, cmap="coolwarm")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(
        image_truth,
        ax=ax1,
        cax=cax1,
    )  # , ticks=[-3, 0, 3], boundaries=np.linspace(-3, 3, 100))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.show()

    print("Testing block complete.")
