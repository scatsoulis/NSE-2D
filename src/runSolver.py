# !/usr/bin/env python3

"""
Created by Sotiris Catsoulis
Outline written by (upd. 10/1/2021):
    Greg Stroot, gstroot@caltech.edu
    Jane Bae (Original code outline)
Copyright 2022 California Institute of Technology
"""

import os
import sys
import numpy as np
from tqdm import tqdm

from scipy.sparse import diags
from timeStepper import time_advance_rk3
from postProcess import interpolate_vel, plot_2D_fields_JPG


from colorama import Fore
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def init_vel(
    Nx: float,  # noqa: N803
    Ny: float,  # noqa: N803
    grid_u_x: (float, float),
    grid_u_y: (float, float),
    grid_v_x: (float, float),
    grid_v_y: (float, float),
) -> tuple[float, float]:
    """
    Purpose:
        Initialize the velocity field

    Inputs:
        Nx      - type: int
                - Grid points in x
        Nx      - type: int
                - Grid points in y
        grid_   - type: numpy.array
                - Coordinates x, y for staggered grids holding u, v

    Outputs:
        u0      - type: float64
                - Initial x velocity
        v0      - type: float64
                - Initial y velocity
    """

    # Define initial velocity functions (Taylor-Green vortex)
    def u0(x: float, y: float) -> float:
        return np.sin(x) * np.cos(y)

    def v0(x: float, y: float) -> float:
        return -np.cos(x) * np.sin(y)

    # Evaluate on staggered grid
    u_init, v_init = np.zeros_like(grid_u_x), np.zeros_like(grid_u_x)
    for i in range(Nx):
        for j in range(Ny):
            u_init[i, j] = u0(grid_u_x[i, j], grid_u_y[i, j])
            v_init[i, j] = v0(grid_v_x[i, j], grid_v_y[i, j])

    return u_init, v_init


def run_solve(Ni: int) -> None:  # noqa: N803
    """
    Purpose:
        Initialize problem and compute solutions
    Inputs:
        N_t     - type: int
                - Number of iterationations
    Outputs:
        The simulation!
    """

    # Initialize velocity field
    u_init, v_init = init_vel(Nx, Ny, grid_u_x, grid_u_y, grid_v_x, grid_v_y)

    # Analytical solution (Taylor-Green):
    def u_analy(x: float, y: float, t: float, Re: float) -> float:  # noqa: N803
        return np.sin(x) * np.cos(y) * np.exp(-2 * t / Re)

    def v_analy(x: float, y: float, t: float, Re: float) -> float:  # noqa: N803
        return -np.cos(x) * np.sin(y) * np.exp(-2 * t / Re)

    def mag_analy_f(x: float, y: float, t: float, Re: float) -> float:  # noqa: N803
        return np.sqrt(
            (np.sin(x) * np.cos(y) * np.exp(-2 * t / Re)) ** 2
            + (-np.cos(x) * np.sin(y) * np.exp(-2 * t / Re)) ** 2,
        )

    # Verify stability
    print("Stability checkpoint - verifying stability...")
    np.max(u_init), np.max(v_init)
    dt_max = min(CFL * dx / np.max(u_init), CFL * dy / np.max(v_init))
    print(f"Maximum allowable time-step: dt_max = {np.round(dt_max, 5)}")
    if dt >= dt_max:
        print(Fore.RED + "WARNING:", "Detected stability issue!", Fore.WHITE)
        print(f"Current time-step dt = {np.round(dt, 5)}")
        sys.exit("Aborting process...")
    print("Advection stability check complete - proceed.")
    print("")

    # Plot IC
    mag_analy = mag_analy_f(grid_p_x, grid_p_y, 0, Re)
    u_inter, v_inter = interpolate_vel(u_init, v_init)
    mag_pred = np.sqrt(np.square(u_inter) + np.square(v_inter))
    err = np.abs(mag_analy - mag_pred)
    plot_2D_fields_JPG(mag_analy, mag_pred, err, Re, 0)

    # Enter time loop
    u_next, v_next = u_init, v_init
    for iteration in tqdm(np.arange(1, Ni)):
        u_current, v_current = u_next, v_next
        u_next, v_next = time_advance_rk3(
            dt,
            u_current,
            v_current,
            Re,
            Dx_back,
            Dx_fwd,
            Dy_back,
            Dy_fwd,
            Dxx_cent,
            Dyy_cent,
            kappa_x_mod_sq,
            kappa_y_mod_sq,
        )

        """ div = np.zeros_like(u_next)
        dk1u_dx, dk1v_dy = np.zeros_like(u_next), np.zeros_like(u_next)
        for j in range(Ny):
            dk1u_dx[:, j] = np.matmul(Dx_fwd, u_next[:, j])
        for i in range(Nx):
            dk1v_dy[i, :] = np.matmul(Dy_fwd, v_next[i, :])
        print(np.max(dk1u_dx + dk1v_dy)) """

        if iteration == int(t_plot / dt):
            # Compute analytical magnitude
            mag_analy = mag_analy_f(grid_p_x, grid_p_y, t_plot, Re)
            u_inter, v_inter = interpolate_vel(u_next, v_next)
            mag_pred = np.sqrt(np.square(u_inter) + np.square(v_inter))

            err = np.abs(mag_analy - mag_pred)
            mse = np.sum(np.square(err)) / (Nx * Ny)

            print(mse)

        if iteration % (int(0.01 / dt)) == 0:
            T = iteration * dt  # noqa: N806
            mag_analy = mag_analy_f(grid_p_x, grid_p_y, dt * iteration, Re)
            u_inter, v_inter = interpolate_vel(u_next, v_next)
            mag_pred = np.sqrt(np.square(u_inter) + np.square(v_inter))
            err = np.abs(mag_analy - mag_pred)
            print(
                "Iteration = {}, T = {}, maxAnalyMag = {}, maxUpred = {}, maxVpred = {}".format(
                    iteration,
                    round(T, 3),
                    round(np.max(mag_analy), 3),
                    round(np.max(u_next), 3),
                    round(np.max(v_next), 3),
                ),
            )

            # Log data
            plot_2D_fields_JPG(mag_analy, mag_pred, err, Re, T)


if __name__ == "__main__":
    # Make dirs
    res_dir = "../results"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # Flow parameter
    Re = 1

    # Spatial extent and resolution
    DOMAIN_X, DOMAIN_Y = 2 * np.pi, 2 * np.pi
    Nx = 64
    Ny = Nx
    dx, dy = DOMAIN_X / Nx, DOMAIN_Y / Ny

    # Time horizon and resolution
    T_max = 0.5
    t_plot = 6
    dt = 0.0001
    Ni = int(T_max / dt)

    # Courant-Friedrich-Lewy number
    CFL = 1.0

    # Grids
    grid_x, grid_y = np.arange(0.0, DOMAIN_X, dx), np.arange(0.0, DOMAIN_Y, dy)
    grid_x_stagg = np.arange(0.0 + dx / 2, DOMAIN_X + dx / 2, dx)
    grid_y_stagg = np.arange(0.0 + dy / 2, DOMAIN_Y + dy / 2, dy)
    grid_xy_x, grid_xy_y = np.meshgrid(grid_x, grid_y, indexing="ij")  # domain
    grid_u_x, grid_u_y = np.meshgrid(grid_x, grid_y_stagg, indexing="ij")  # u
    grid_v_x, grid_v_y = np.meshgrid(grid_x_stagg, grid_y, indexing="ij")  # v
    grid_p_x, grid_p_y = np.meshgrid(grid_x_stagg, grid_y_stagg, indexing="ij")  # p

    # Derivative operators
    # forwards, backwards (x, y)
    kxfwd, kxback = [-np.ones(Nx), np.ones(Nx - 1)], [-np.ones(Nx - 1), np.ones(Nx)]
    offset_xfwd, offset_xback = [0, 1], [-1, 0]
    Dx_fwd, Dx_back = (
        diags(kxfwd, offset_xfwd).toarray(),
        diags(kxback, offset_xback).toarray(),
    )
    Dx_fwd[-1, 0], Dx_back[0, -1] = 1, -1
    Dx_fwd, Dx_back = (1 / dx) * Dx_fwd, (1 / dx) * Dx_back
    kyfwd, kyback = [-np.ones(Ny), np.ones(Ny - 1)], [-np.ones(Ny - 1), np.ones(Ny)]
    offset_yfwd, offset_yback = [0, 1], [-1, 0]
    Dy_fwd, Dy_back = (
        diags(kyfwd, offset_yfwd).toarray(),
        diags(kyback, offset_yback).toarray(),
    )
    Dy_fwd[-1, 0], Dy_back[0, -1] = 1, -1
    Dy_fwd, Dy_back = (1 / dy) * Dy_fwd, (1 / dy) * Dy_back
    # central (x, y)
    kxcent, kycent = (
        [np.ones(Nx - 1), -2 * np.ones(Nx), np.ones(Nx - 1)],
        [np.ones(Ny - 1), -2 * np.ones(Ny), np.ones(Ny - 1)],
    )
    offset_xcent, offset_ycent = [-1, 0, 1], [-1, 0, 1]
    Dxx_cent, Dyy_cent = (
        diags(kxcent, offset_xcent).toarray(),
        diags(kycent, offset_ycent).toarray(),
    )
    Dxx_cent[0, -1], Dyy_cent[0, -1] = 1, 1
    Dxx_cent[-1, 0], Dyy_cent[-1, 0] = 1, 1
    Dxx_cent, Dyy_cent = (1 / (dx**2)) * Dxx_cent, (1 / (dy**2)) * Dyy_cent

    # Compute modified wavenumbers
    kappa_x, kappa_y = np.zeros((Nx, Ny)), np.zeros((Nx, Ny))
    for j in range(Ny):
        kappa_x[:, j] = np.fft.fftshift(
            (2 * np.pi / DOMAIN_X) * np.arange(-Nx / 2, Nx / 2),
        )
    for i in range(Nx):
        kappa_y[i, :] = np.fft.fftshift(
            (2 * np.pi / DOMAIN_Y) * np.arange(-Ny / 2, Ny / 2),
        )
    kappa_x_mod_sq, kappa_y_mod_sq = (
        2 * (1 - np.cos(kappa_x * dx)) / dx**2,
        2 * (1 - np.cos(kappa_y * dy)) / dy**2,
    )

    # Print simulation characteristics
    print("")
    print("--------------------------------------------------------------------------")
    print("Welcome home - this is 2D incompressible NSE")
    print(f"Domain size: [{round(DOMAIN_X, 2)} x {round(DOMAIN_Y, 2)}]")
    print(
        f"Resolution: (Nx, Ny) = ({Nx}, {Ny}) ; (dx, dy) = ({round(dx, 5)}, {round(dy, 5)})",
    )
    print(f"Time-step: dt = {dt}")
    print("--------------------------------------------------------------------------")
    print("")

    # Run solver
    run_solve(Ni)
