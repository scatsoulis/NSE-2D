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
from scipy.sparse import diags
from matplotlib import ticker
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

plt.rcParams["text.usetex"] = True
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.loc"] = "best"
plt.rcParams["legend.edgecolor"] = "1.0"
# plt.rcParams.update({"figure.constrained_layout.use": True})


def fmt(x: str, pos) -> str:
    a, b = f"{x:.2e}".split("e")
    b = int(b)

    return rf"${a} \times 10^{{{b}}}$"


def plot_2D_fields_PDF(  # noqa: N802
    mag_analy: (float, float),
    mag_pred: (float, float),
    err: (float, float),
    Re: float,  # noqa: N803
    T: float,  # noqa: N803
) -> None:
    T = round(T, 3)  # noqa: N806
    Nx = mag_pred.shape[0]  # noqa: N806

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    image_truth = ax0.imshow(mag_analy, cmap="hot")
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig.colorbar(image_truth, ax=ax0, cax=cax0)  # noqa:  F841
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])

    image_pred = ax1.imshow(mag_pred, cmap="hot")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(image_pred, ax=ax1, cax=cax1)  # noqa:  F841
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    image_error = ax2.imshow(err, cmap="Reds")
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(image_error, ax=ax2, cax=cax2, format=ticker.FuncFormatter(fmt))  # noqa:  F841
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    plt.savefig(
        "../results/mag_tot_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(iter) + ".pdf",
    )

    fig_analy, ax0 = plt.subplots()
    image_truth = ax0.imshow(mag_analy, cmap="hot")
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig_analy.colorbar(image_truth, ax=ax0, cax=cax0)  # noqa:  F841
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    plt.savefig(
        "../results/mag_analy_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(iter) + ".pdf",
    )

    fig_pred, ax1 = plt.subplots()
    image_pred = ax1.imshow(mag_analy, cmap="hot")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig_pred.colorbar(image_pred, ax=ax1, cax=cax1)  # noqa:  F841
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.savefig(
        "../results/mag_pred_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(iter) + ".pdf",
    )

    fig_err, ax2 = plt.subplots()
    image_error = ax2.imshow(err, cmap="Reds")
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig_err.colorbar(  # noqa:  F841
        image_error,
        ax=ax2,
        cax=cax2,
        format=ticker.FuncFormatter(fmt),
    )
    plt.ticklabel_format()
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    plt.savefig(
        "../results/mag_err_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(iter) + ".pdf",
    )


def plot_2D_fields_JPG(  # noqa: N802
    mag_analy: (float, float),
    mag_pred: (float, float),
    err: (float, float),
    Re: float,  # noqa: N803
    T: float,  # noqa: N803
) -> None:
    T = round(T, 3)  # noqa: N806
    Nx = mag_pred.shape[0]  # noqa: N806

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

    image_truth = ax0.imshow(mag_analy, cmap="hot")
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig.colorbar(image_truth, ax=ax0, cax=cax0)  # noqa:  F841
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])

    image_pred = ax1.imshow(mag_pred, cmap="hot")
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(image_pred, ax=ax1, cax=cax1)  # noqa:  F841
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    image_error = ax2.imshow(err, cmap="Reds")
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(image_error, ax=ax2, cax=cax2, format=ticker.FuncFormatter(fmt))  # noqa: F841
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    plt.tight_layout()
    plt.savefig("../results/mag_tot_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(T) + ".jpg")

    fig_analy, ax0 = plt.subplots()
    image_truth = ax0.imshow(mag_analy, cmap="hot")
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig_analy.colorbar(image_truth, ax=ax0, cax=cax0)  # noqa:  F841
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(
        "../results/mag_analy_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(T) + ".jpg",
    )

    fig_pred, ax1 = plt.subplots()
    image_pred = ax1.imshow(mag_pred, cmap="hot")  # noqa:  F841
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig_pred.colorbar(image_truth, ax=ax1, cax=cax1)  # noqa:  F841
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.savefig(
        "../results/mag_pred_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(T) + ".jpg",
    )

    fig_err, ax2 = plt.subplots()
    image_error = ax2.imshow(err, cmap="Reds")
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig_err.colorbar(  # noqa:  F841
        image_error,
        ax=ax2,
        cax=cax2,
        format=ticker.FuncFormatter(fmt),
    )
    plt.ticklabel_format()
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    plt.tight_layout()
    plt.savefig("../results/mag_err_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(T) + ".jpg")


def interpolate_vel(u: (float, float), v: (float, float)) -> tuple[float, float]:
    Nx, Ny = np.shape(u)[0], np.shape(u)[1]  # noqa: N806
    kxfwd = [np.ones(Nx), np.ones(Nx - 1)]
    offset_xfwd = [0, 1]
    interp_x_fwd = diags(kxfwd, offset_xfwd).toarray()
    interp_x_fwd[-1, 0] = 1
    interp_x_fwd = 0.5 * interp_x_fwd

    u_cell_cent, v_cell_cent = np.zeros_like(u), np.zeros_like(v)
    for i in range(Nx - 1):
        u_cell_cent[i, :] = 0.5 * (u[i, :] + u[i + 1, :])
    u_cell_cent[-1, :] = 0.5 * (u[-1, :] + u[0, :])

    for j in range(Ny - 1):
        v_cell_cent[:, j] = 0.5 * (v[:, j] + v[:, j + 1])
    v_cell_cent[:, -1] = 0.5 * (v[:, -1] + v[:, 0])

    return u_cell_cent, v_cell_cent


def plot_accuracy() -> None:
    dx_u = [0.09817, 0.04909, 0.02454]
    mse_u = [6.194314140414291e-11, 3.873265818518184e-12, 2.4210755123291014e-13]

    dx_v = [0.09817, 0.04909, 0.02454]
    mse_v = [6.19431414041005e-11, 3.873265818523062e-12, 2.421075512310808e-13]

    dx_mag = [0.09817, 0.04909, 0.02454]
    mse_mag = [6.785780130141549e-07, 4.2422769837072096e-08, 2.651605075095774e-09]

    fig, ax = plt.subplots()
    ax.loglog(dx_u, mse_u, "-o", label="u")
    ax.loglog(dx_v, mse_v, "-x", label="v")
    ax.loglog(dx_mag, mse_mag, "-x", label="Magnitude")
    ax.set_xlabel("Grid spacing")
    ax.set_ylabel("MSE")
    ax.legend()
    plt.savefig("accuracy.pdf")
    plt.show()


def bar_chart() -> None:
    x = ["RHS", "Poisson", "Other"]
    h = [60, 30, 10]
    plt.bar(x, h, hatch="/", edgecolor="white")
    plt.ylim([0, 100])
    plt.ylabel("Compute time (%)")
    plt.savefig("time.pdf")
    plt.show()


if __name__ == "__main__":
    plot_accuracy()
    bar_chart()
