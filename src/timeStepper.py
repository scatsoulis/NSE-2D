# !/usr/bin/env python3

"""
Created by Sotiris Catsoulis
Outline written by (upd. 10/1/2021):
    Greg Stroot, gstroot@caltech.edu
    Jane Bae (Original code outline)
Copyright 2022 California Institute of Technology
"""

import numpy as np
import matplotlib.pyplot as plt
from computeRHS import compute_rhs
from computeProjectionStep import compute_projection_step


def time_advance_rk3(
    dt: float,
    u_current: (float, float),
    v_current: (float, float),
    Re: float,  # noqa: N803
    dx_back: (float, float),
    dx_fwd: (float, float),
    dy_back: (float, float),
    dy_fwd: (float, float),
    dxx_cent: (float, float),
    dyy_cent: (float, float),
    kappa_x_mod_sq: (float, float),
    kappa_y_mod_sq: (float, float),
) -> (float, float):
    """
    Purpose:
        Run RK3 for time advancement
    Inputs:
        u_current   - type: float64
                    - current x velocity
        v_current   - type: float64
                    - current y velocity

    Outputs:
        u_current   - type: double
                - updated solution
    """

    # div_final = du_next_dx + dv_next_dy
    # dp_dx1, dp_dy1 = compute_projection_step(
    #     u_star, v_star, dx_back, dx_fwd, dy_back, dy_fwd, kappa_x_mod_sq, kappa_y_mod_sq
    # )
    # print("div_glob_final: ", div_final)
    # print("---")
    # sys.exit()

    """
    div = np.zeros_like(k1u)
    dk1u_dx, dk1v_dy = np.zeros_like(k1u), np.zeros_like(k1u)
    for j in range(Ny):
        dk1u_dx[:, j] = np.matmul(dx_fwd, k1u[:, j])
    for i in range(Nx):
        dk1v_dy[i, :] = np.matmul(dy_fwd, k1v[i, :])
    print(dk1u_dx + dk1v_dy)
    """

    # Nx = u_current.shape[0]  # noqa: N806

    k1u_input, k1v_input = u_current, v_current
    k1u, k1v = compute_rhs(
        k1u_input,
        k1v_input,
        Re,
        dx_back,
        dx_fwd,
        dy_back,
        dy_fwd,
        dxx_cent,
        dyy_cent,
    )
    dp_dx1, dp_dy1 = compute_projection_step(
        k1u,
        k1v,
        dx_fwd,
        dx_back,
        dy_fwd,
        dy_back,
        kappa_x_mod_sq,
        kappa_y_mod_sq,
    )
    k1u, k1v = k1u - dp_dx1, k1v - dp_dy1

    k2u_input, k2v_input = u_current + (dt / 2) * (k1u), v_current + (dt / 2) * (k1v)
    k2u, k2v = compute_rhs(
        k2u_input,
        k2v_input,
        Re,
        dx_back,
        dx_fwd,
        dy_back,
        dy_fwd,
        dxx_cent,
        dyy_cent,
    )
    dp_dx2, dp_dy2 = compute_projection_step(
        k2u,
        k2v,
        dx_fwd,
        dx_back,
        dy_fwd,
        dy_back,
        kappa_x_mod_sq,
        kappa_y_mod_sq,
    )

    k2u, k2v = k2u - dp_dx2, k2v - dp_dy2

    k3u_input, k3v_input = (
        u_current - dt * (k1u) + 2 * dt * (k2u),
        v_current - dt * (k1v) + 2 * dt * (k2v),
    )
    k3u, k3v = compute_rhs(
        k3u_input,
        k3v_input,
        Re,
        dx_back,
        dx_fwd,
        dy_back,
        dy_fwd,
        dxx_cent,
        dyy_cent,
    )
    dp_dx3, dp_dy3 = compute_projection_step(
        k3u,
        k3v,
        dx_fwd,
        dx_back,
        dy_fwd,
        dy_back,
        kappa_x_mod_sq,
        kappa_y_mod_sq,
    )

    k3u, k3v = k3u - dp_dx3, k3v - dp_dy3

    return u_current + (dt / 6) * (k1u + 4 * k2u + k3u), v_current + (dt / 6) * (
        k1v + 4 * k2v + k3v
    )


if __name__ == "__main__":
    # Test time-advancement (full code in other script)

    T = 15
    dt = 0.001

    # Test function: du/dt = -(2+0.01*t^2)*u
    def f(u: float, t: float) -> float:
        return -(2 + 0.01 * t ^ 2) * u

    # Define analytical solution
    u0 = 4

    def u_analytical(t: float) -> float:
        return u0 * np.exp(-(2 * t + (0.01 / 3) * (t**3)))

    u_truth = []

    for t in np.arange(0, T, 0.05):
        u_truth.append(u_analytical(t))

    error = []

    # Run time_advance_rk3 using test function
    for t in np.arange(0, T, dt):
        u_current = time_advance_rk3(t, dt, T)

        error.append(abs(u_current - u_analytical(T)))

    # Plot error
    fig_ts, ax_ts = plt.subplots()
    ax_ts.loglog(np.arange(), u_truth, label="truth", c="green")
    plt.grid(True, which="both")
    plt.xlabel("Time")
    plt.ylabel("v(t)")
    plt.title("Order analysis")
    # plt.savefig("order.pdf")

    print("Testing block complete.")
