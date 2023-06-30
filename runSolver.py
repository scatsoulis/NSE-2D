"""
Summary:
    Outline for Ae232a Fall 2022 final project
Written by:
    Greg Stroot, gstroot@caltech.edu
    Jane Bae (Original code outline)
Last updated:
    10/1/2021
"""

from libraries import *

from timeStepper import time_advance_RK3
from postProcess import plot2DfieldsJPG, interpolateVel

def initVel(Nx, Ny, grid_u_x, grid_u_y, grid_v_x, grid_v_y):
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
    u0 = lambda x, y: np.sin(x)*np.cos(y)
    v0 = lambda x, y: -np.cos(x)*np.sin(y)

    # Evaluate on staggered grid
    u_init, v_init = np.zeros_like(grid_u_x), np.zeros_like(grid_u_x)
    for i in range(Nx):
        for j in range(Ny):
            u_init[i, j] = u0(grid_u_x[i, j], grid_u_y[i, j])
            v_init[i, j] = v0(grid_v_x[i, j], grid_v_y[i, j])

    return u_init, v_init

def runSolve(Ni):
    """
    Purpose:
        Initialize problem and compute solutions
    Inputs:
        N_t     - type: int
                - Number of iterations
    Outputs:
        The simulation!
    """

    # Initialize velocity field
    u_init, v_init = initVel(Nx, Ny, grid_u_x, grid_u_y, grid_v_x, grid_v_y)

    # Analytical solution (Taylor-Green):
    u_analy = lambda x, y, t, Re : np.sin(x)*np.cos(y)*np.exp(-2*t/Re)
    v_analy = lambda x, y, t, Re : -np.cos(x)*np.sin(y)*np.exp(-2*t/Re)
    mag_analy_f = lambda x, y, t, Re : np.sqrt((np.sin(x)*np.cos(y)*np.exp(-2*t/Re))**2 + (-np.cos(x)*np.sin(y)*np.exp(-2*t/Re))**2)

    # Verify stability
    print("Stability checkpoint - verifying stability...")
    np.max(u_init), np.max(v_init)
    dt_max = min(CFL*dx/np.max(u_init), CFL*dy/np.max(v_init))
    print("Maximum allowable time-step: dt_max = {}".format((np.round(dt_max, 5))))
    if dt >= dt_max:
        print(Fore.RED + "WARNING:", "Detected stability issue!", Fore.WHITE)
        print("Current time-step dt = {}".format(np.round(dt, 5)))
        sys.exit("Aborting process...")
    print("Advection stability check complete - proceed.")
    print("")

    # Plot IC
    mag_analy = mag_analy_f(grid_p_x, grid_p_y, 0, Re)
    u_inter, v_inter = interpolateVel(u_init, v_init)
    mag_pred = np.sqrt(np.square(u_inter) + np.square(v_inter))
    err = np.abs(mag_analy - mag_pred)
    # plot2DfieldsJPG(mag_analy, mag_pred, err, Re, 0)

    # Enter time loop
    u_next, v_next = u_init, v_init
    for iter in tqdm(np.arange(1, Ni)):
        
        u_current, v_current = u_next, v_next
        u_next, v_next = time_advance_RK3(dt, u_current, v_current, Re, Dx_back, Dx_fwd, Dy_back, Dy_fwd, Dxx_cent, Dyy_cent, kappa_x_mod_sq, kappa_y_mod_sq)

        """ div = np.zeros_like(u_next)
        dk1u_dx, dk1v_dy = np.zeros_like(u_next), np.zeros_like(u_next)
        for j in range(Ny):
            dk1u_dx[:, j] = np.matmul(Dx_fwd, u_next[:, j])
        for i in range(Nx):
            dk1v_dy[i, :] = np.matmul(Dy_fwd, v_next[i, :])
        print(np.max(dk1u_dx + dk1v_dy)) """

        if iter == int(t_plot/dt):
            
            # Compute analytical magnitude
            mag_analy = mag_analy_f(grid_p_x, grid_p_y, t_plot, Re)
            u_inter, v_inter = interpolateVel(u_next, v_next)
            mag_pred = np.sqrt(np.square(u_inter) + np.square(v_inter))

            err = np.abs(mag_analy - mag_pred)
            mse = np.sum(np.square(err))/(Nx*Ny)

            print(mse)

        if iter % (int(0.01/dt)) == 0:
            T = iter*dt
            mag_analy = mag_analy_f(grid_p_x, grid_p_y, dt*iter, Re)
            u_inter, v_inter = interpolateVel(u_next, v_next)
            mag_pred = np.sqrt(np.square(u_inter) + np.square(v_inter))
            err = np.abs(mag_analy - mag_pred)
            print("Iter = {}, T = {}, maxAnalyMag = {}, maxUpred = {}, maxVpred = {}".format(iter, round(T, 3), round(np.max(mag_analy), 3), round(np.max(u_next), 3), round(np.max(v_next), 3)))
            # plot2DfieldsJPG(mag_analy, mag_pred, err, Re, T)
        

    return None

if __name__ == "__main__":

    # Flow parameter
    Re = 1
    
    # Spatial extent and resolution
    DOMAIN_X, DOMAIN_Y = 2*np.pi, 2*np.pi
    Nx = 64
    Ny = Nx
    dx, dy = DOMAIN_X/Nx, DOMAIN_Y/Ny

    # Time horizon and resolution
    T_max = 0.5
    t_plot = 6
    dt = 0.0001
    Ni = int(T_max/dt)

    # Courant-Friedrich-Lewy number
    CFL = 1.0

    # Grids
    grid_x, grid_y = np.arange(0.0, DOMAIN_X, dx), np.arange(0.0, DOMAIN_Y, dy)
    grid_x_stagg = np.arange(0.0 + dx/2, DOMAIN_X + dx/2, dx)
    grid_y_stagg = np.arange(0.0 + dy/2, DOMAIN_Y + dy/2, dy)
    grid_xy_x, grid_xy_y = np.meshgrid(grid_x, grid_y, indexing='ij') # domain
    grid_u_x, grid_u_y = np.meshgrid(grid_x, grid_y_stagg, indexing='ij') # u
    grid_v_x, grid_v_y = np.meshgrid(grid_x_stagg, grid_y, indexing='ij') # v
    grid_p_x, grid_p_y = np.meshgrid(grid_x_stagg, grid_y_stagg, indexing='ij') # p

    # Derivative operators
    # forwards, backwards (x, y)
    kxfwd, kxback = [-np.ones(Nx), np.ones(Nx-1)], [-np.ones(Nx-1), np.ones(Nx)]
    offset_xfwd, offset_xback = [0, 1], [-1, 0]
    Dx_fwd, Dx_back = diags(kxfwd, offset_xfwd).toarray(), diags(kxback, offset_xback).toarray()
    Dx_fwd[-1, 0], Dx_back[0, -1] = 1, -1
    Dx_fwd, Dx_back = (1/dx)*Dx_fwd, (1/dx)*Dx_back
    kyfwd, kyback = [-np.ones(Ny), np.ones(Ny-1)], [-np.ones(Ny-1), np.ones(Ny)]
    offset_yfwd, offset_yback = [0, 1], [-1, 0]
    Dy_fwd, Dy_back = diags(kyfwd, offset_yfwd).toarray(), diags(kyback, offset_yback).toarray()
    Dy_fwd[-1, 0], Dy_back[0, -1] = 1, -1
    Dy_fwd, Dy_back = (1/dy)*Dy_fwd, (1/dy)*Dy_back
    # central (x, y)
    kxcent, kycent = [np.ones(Nx-1), -2*np.ones(Nx), np.ones(Nx-1)], [np.ones(Ny-1), -2*np.ones(Ny), np.ones(Ny-1)]
    offset_xcent, offset_ycent = [-1, 0, 1], [-1, 0, 1]
    Dxx_cent, Dyy_cent = diags(kxcent, offset_xcent).toarray(), diags(kycent, offset_ycent).toarray()
    Dxx_cent[0, -1], Dyy_cent[0, -1] = 1, 1
    Dxx_cent[-1, 0], Dyy_cent[-1, 0] = 1, 1
    Dxx_cent, Dyy_cent  = (1/(dx**2))*Dxx_cent, (1/(dy**2))*Dyy_cent

    # Compute modified wavenumbers
    kappa_x, kappa_y = np.zeros((Nx, Ny)), np.zeros((Nx, Ny))
    for j in range(Ny):
        kappa_x[:, j] = np.fft.fftshift((2*np.pi/DOMAIN_X)*np.arange(-Nx/2, Nx/2))
    for i in range(Nx):
        kappa_y[i, :] = np.fft.fftshift((2*np.pi/DOMAIN_Y)*np.arange(-Ny/2, Ny/2))
    kappa_x_mod_sq, kappa_y_mod_sq = 2*(1-np.cos(kappa_x*dx))/dx**2, 2*(1-np.cos(kappa_y*dy))/dy**2

    # Print simulation characteristics
    print("")
    print("--------------------------------------------------------------------------")
    print("Welcome home - this is 2D incompressible NSE")
    print("Domain size: [{} x {}]".format(round(DOMAIN_X, 2), round(DOMAIN_Y, 2)))
    print("Resolution: (Nx, Ny) = ({}, {}) ; (dx, dy) = ({}, {})".format(Nx, Ny, round(dx, 5), round(dy, 5)))
    print("Time-step: dt = {}".format(dt))
    print("--------------------------------------------------------------------------")
    print("")

    sys.exit()
    # Run solver
    runSolve(Ni)
