from libraries import *

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def computeVorticity(u, v):
    pass

def plot2DfieldsPDF(sol_analy, sol_pred, err, Re, T):

    T = round(T, 3)
    Nx = sol_pred.shape[0]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    imageTruth  = ax0.imshow(sol_analy, cmap='bwr')
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig.colorbar(imageTruth, ax=ax0, cax=cax0)#
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])

    imagePred  = ax1.imshow(sol_pred, cmap='bwr')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(imagePred, ax=ax1, cax=cax1)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    imageError = ax2.imshow(err, cmap='Reds')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(imageError, ax=ax2, cax=cax2, format=ticker.FuncFormatter(fmt))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    plt.savefig("video/mag_tot_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(iter) + ".pdf")
    # plt.show()

    fig_analy, ax0 = plt.subplots()
    imageTruth  = ax0.imshow(sol_analy, cmap='bwr')
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig_analy.colorbar(imageTruth, ax=ax0, cax=cax0)#
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    plt.savefig("video/sol_analy_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(iter) + ".pdf")

    fig_pred, ax1 = plt.subplots()
    imagePred  = ax1.imshow(sol_analy, cmap='bwr')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig_pred.colorbar(imagePred, ax=ax1, cax=cax1)#
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.savefig("video/sol_pred_" + str(Nx)+ "_Re_" + str(Re) + "_T_" + str(iter) + ".pdf")

    fig_err, ax2 = plt.subplots()
    imageError = ax2.imshow(err, cmap='Reds')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig_err.colorbar(imageError, ax=ax2, cax=cax2, format=ticker.FuncFormatter(fmt))
    plt.ticklabel_format()
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    plt.savefig("video/mag_err_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(iter) + ".pdf")    

    return None

def plot2DfieldsJPG(sol_analy, sol_pred, err, Re, T):

    T = round(T, 3)
    print(T)
    Nx = sol_pred.shape[0]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    imageTruth  = ax0.imshow(sol_analy, cmap='bwr')
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig.colorbar(imageTruth, ax=ax0, cax=cax0)#
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])

    imagePred  = ax1.imshow(sol_pred, cmap='bwr')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(imagePred, ax=ax1, cax=cax1)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])

    imageError = ax2.imshow(err, cmap='Reds')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(imageError, ax=ax2, cax=cax2, format=ticker.FuncFormatter(fmt))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    plt.savefig("video/mag_tot_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(T) + ".jpg")
    # plt.show()

    fig_analy, ax0 = plt.subplots()
    imageTruth  = ax0.imshow(sol_analy, cmap='bwr')
    divider0 = make_axes_locatable(ax0)
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig_analy.colorbar(imageTruth, ax=ax0, cax=cax0)#
    ax0.set_xticklabels([])
    ax0.set_yticklabels([])
    plt.savefig("video/sol_analy_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(T) + ".jpg")

    fig_pred, ax1 = plt.subplots()
    imagePred  = ax1.imshow(sol_analy, cmap='bwr')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig_pred.colorbar(imagePred, ax=ax1, cax=cax1)#
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    plt.savefig("video/sol_pred_" + str(Nx)+ "_Re_" + str(Re) + "_T_" + str(T) + ".jpg")

    fig_err, ax2 = plt.subplots()
    imageError = ax2.imshow(err, cmap='Reds')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig_err.colorbar(imageError, ax=ax2, cax=cax2, format=ticker.FuncFormatter(fmt))
    plt.ticklabel_format()
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    plt.savefig("video/mag_err_" + str(Nx) + "_Re_" + str(Re) + "_T_" + str(T) + ".jpg")    

    return None

def interpolateVel(u, v):

    Nx, Ny = np.shape(u)[0], np.shape(u)[1]
    kxfwd = [np.ones(Nx), np.ones(Nx-1)]
    offset_xfwd = [0, 1]
    interp_x_fwd = diags(kxfwd, offset_xfwd).toarray()
    interp_x_fwd[-1, 0] = 1
    interp_x_fwd = 0.5*interp_x_fwd

    u_cellCent, v_cellCent = np.zeros_like(u), np.zeros_like(v)
    for i in range(Nx-1):
        u_cellCent[i, :] = 0.5*(u[i, :] + u[i+1, :])
    u_cellCent[-1, :] = 0.5*(u[-1, :] + u[0, :])

    for j in range(Ny-1):
        v_cellCent[:, j] = 0.5*(v[:, j] + v[:, j+1])
    v_cellCent[:, -1] = 0.5*(v[:, -1] + v[:, 0])
    
    return u_cellCent, v_cellCent

def plotAccuracy():

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

def barChart():

    x = ["RHS", "Poisson", "Other"]
    h = [60, 30, 10]
    plt.bar(x, h, hatch='/', edgecolor="white")
    plt.ylim([0, 100])
    plt.ylabel("Compute time (%)")
    plt.savefig("time.pdf")
    plt.show()

if __name__ == "__main__":
    barChart()