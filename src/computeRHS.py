# !/usr/bin/env python3

"""
Created by Sotiris Catsoulis
Outline written by (upd. 10/1/2021):
    Greg Stroot, gstroot@caltech.edu
    Jane Bae (Original code outline)
Copyright 2022 California Institute of Technology
"""

import numpy as np


def compute_rhs(
    u: (float, float),
    v: (float, float),
    Re: float,  # noqa: N803
    dx_back: (float, float),
    dx_fwd: (float, float),
    dy_back: (float, float),
    dy_fwd: (float, float),
    dxx_cent: (float, float),
    dyy_cent: (float, float),
) -> tuple[float, float]:
    """
    Purpose:
        Compute the RHS of the 2D-NS equations.

        rhs_u = - d(uu)/dx - d(uv)/dy + domain_x/Re*d^2u/dx^2 + domain_x/Re*d^2u/dy^2
        rhs_v = - d(uv)/dx - d(vv)/dy + domain_y/Re*d^2v/dx^2 + domain_y/Re*d^2v/dy^2

    Inputs:
        U        - type: float64
                 - Horizontal velocity
        V        - type: float64
                 - Vertical velocity
        Re       - type: float64
                 - Reynolds number
    Outputs:
        rhs_u   - type: float64
                - RHS of u component of momentum equation
        rhs_v   - type: float64
                - RHS of v component of momentum equation
    """

    Nx, Ny = u.shape[0], u.shape[1]  # noqa: N806

    # Compute Fu
    # Interpolating needed quantities
    u_corner, u_center = np.zeros((Nx, Ny)), np.zeros((Nx, Ny))
    v_corner, v_center = np.zeros((Nx, Ny)), np.zeros((Nx, Ny))
    v_corner[0, :] = 0.5 * (v[-1, :] + v[0, :])  # interpolation at left boundary
    for i in range(Nx - 1):
        u_center[i, :] = 0.5 * (u[i, :] + u[i + 1, :])
        v_corner[i + 1, :] = 0.5 * (v[i, :] + v[i + 1, :])
    u_center[-1, :] = 0.5 * (u[-1, :] + u[0, :])  # interpolation at right boundary

    u_corner[:, 0] = 0.5 * (u[:, -1] + u[:, 0])  # interpolation at bottom boundary
    for j in range(Ny - 1):
        u_corner[:, j + 1] = 0.5 * (u[:, j] + u[:, j + 1])
        v_center[:, j] = 0.5 * (v[:, j] + v[:, j + 1])
    v_center[:, -1] = 0.5 * (v[:, -1] + v[:, 0])  # interpolation at top boundary

    # Compute duu_dx (with central interpolant)
    duu_dx = np.zeros((Nx, Ny))
    for j in range(Ny):
        duu_dx[:, j] = np.matmul(dx_back, np.multiply(u_center[:, j], u_center[:, j]))

    # Compute d(uv)/dy (with corner interpolants)
    duv_dy = np.zeros((Nx, Ny))
    for i in range(Nx):
        duv_dy[i, :] = np.matmul(dy_fwd, np.multiply(u_corner[i, :], v_corner[i, :]))

    # Compute 2*pi/Re*ddu_dxx (no interpolation)
    ddu_dxx = np.zeros((Nx, Ny))
    for j in range(Ny):
        ddu_dxx[:, j] = (1 / Re) * (np.matmul(dxx_cent, u[:, j]))

    # Compute 2*pi/Re*ddu_dyy (no interpolation)
    ddu_dyy = np.zeros((Nx, Ny))
    for i in range(Nx):
        ddu_dyy[i, :] = (1 / Re) * (np.matmul(dyy_cent, u[i, :]))

    # Compute rhs_u
    rhs_u = -duu_dx - duv_dy + ddu_dxx + ddu_dyy

    # Compute dvv_dy (with central interpolants)
    dvv_dy = np.zeros((Nx, Ny))
    for i in range(Nx):
        dvv_dy[i, :] = np.matmul(dy_back, np.multiply(v_center[i, :], v_center[i, :]))

    # Compute duv_dx (with corner interpolants)
    duv_dx = np.zeros((Nx, Ny))
    for j in range(Ny):
        duv_dx[:, j] = np.matmul(dx_fwd, np.multiply(u_corner[:, j], v_corner[:, j]))

    # Compute 2*pi/Re*ddv_dxx (no interpolants)
    ddv_dxx = np.zeros((Nx, Ny))
    for j in range(Ny):
        ddv_dxx[:, j] = (1 / Re) * (np.matmul(dxx_cent, v[:, j]))

    # Compute 2*pi/Re*ddv_dyy (no interpolants)
    ddv_dyy = np.zeros((Nx, Ny))
    for i in range(Nx):
        ddv_dyy[i, :] = (1 / Re) * (np.matmul(dyy_cent, v[i, :]))

    # Compute rhs_v
    rhs_v = -dvv_dy - duv_dx + ddv_dxx + ddv_dyy

    return rhs_u, rhs_v


if __name__ == "__main__":
    # TESTING BLOCK
    Re = 10
    Nx = 16
    Ny = Nx

    DOMAIN_X, DOMAIN_Y = 2 * np.pi, 2 * np.pi
    dx, dy = DOMAIN_X / Nx, DOMAIN_Y / Ny

    # Grid "wall-to-wall"
    grid_x, grid_y = np.arange(0.0, DOMAIN_X, dx), np.arange(0.0, DOMAIN_Y, dy)

    # Grid staggered right (for v, u)
    grid_x_stagg, grid_y_stagg = (
        np.arange(0.0 + dx / 2, DOMAIN_X + dx / 2, dx),
        np.arange(0.0 + dy / 2, DOMAIN_Y + dy / 2, dy),
    )

    # 2-D grid "wall-to-wall"
    grid_xy_x, grid_xy_y = np.meshgrid(grid_x, grid_y, indexing="ij")

    # 2-D grid staggered up (for u)
    grid_u_x, grid_u_y = np.meshgrid(grid_x, grid_y_stagg, indexing="ij")

    # 2-D grid staggered right (for v)
    grid_v_x, grid_v_y = np.meshgrid(grid_x_stagg, grid_y, indexing="ij")

    # Define test function
    def u_init(x: float, y: float) -> float:
        return np.cos(x)

    def v_init(x: float, y: float) -> float:
        return np.sin(y)

    # Define analytical solution of rhs_u, rhs_v
    def rhs_u_analy(x: float, y: float) -> float:
        return 2 * np.cos(x) * np.sin(x) - np.cos(x) * np.cos(y) + (2 * np.pi / Re) * (-np.cos(x))

    def rhs_v_analy(x: float, y: float) -> float:
        return np.sin(x) * np.sin(y) - 2 * np.cos(y) * np.sin(y) + (2 * np.pi / Re) * (-np.sin(y))

    # Run compute_rhs using test function
    rhs_u_num, rhs_v_num = compute_rhs(Re)

    # Compute MSE
    L2_u = np.zeros((Nx, Ny))
    L2_v = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            L2_u[i, j] = (rhs_u_num[i, j] - rhs_u_analy(grid_xy_x[i, j], grid_xy_y[i, j])) ** 2
            L2_v[i, j] = (rhs_v_num[i, j] - rhs_v_analy(grid_xy_x[i, j], grid_xy_y[i, j])) ** 2
    print("MSE_u: " + str(np.sum(L2_u) / (Nx * Ny)))
    print("MSE_v: " + str(np.sum(L2_v) / (Nx * Ny)))

    # Plot convergence (artifical)
    r"""
    grids = [0.09817477042468103, 0.19634954084936207, 0.39269908169872414]
    error_u = [0.0006102351813966211, 0.0025337908271350854, 0.011547308912441639]
    error_v = [0.0006102351813966217, 0.0025337908271351054, 0.011547308912441639]

    fig_u, ax_u = plt.subplots()
    ax_u.loglog(grids, error_u, "o-")
    plt.xlabel(r"Grid spacing $\Delta x$")
    plt.ylabel("MSE")
    plt.savefig("spatialDerivatives/u.pdf")

    fig_v, ax_v = plt.subplots()
    ax_v.loglog(grids, error_v, "o-")
    plt.xlabel(r"Grid spacing $\Delta x$")
    plt.ylabel("MSE")
    plt.savefig("spatialDerivatives/v.pdf")

    plt.show()
    """

    print("Testing block complete")
