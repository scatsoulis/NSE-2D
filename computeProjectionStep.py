from libraries import *

def compute_projection_step(u_star, v_star, Dx_fwd, Dx_back, Dy_fwd, Dy_back, kappa_x_mod_sq, kappa_y_mod_sq):
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

    Nx, Ny = np.shape(u_star)[0], np.shape(u_star)[1]

    # Compute (cell-centered) divergence of starred velocities 
    div = np.zeros_like(u_star)
    du_star_dx, dv_star_dy = np.zeros_like(u_star), np.zeros_like(v_star)
    for j in range(Ny):
        du_star_dx[:, j] = np.matmul(Dx_fwd, u_star[:, j])
    for i in range(Nx):
        dv_star_dy[i, :] = np.matmul(Dy_fwd, v_star[i, :])
    RHS_p = du_star_dx + dv_star_dy

    RHS_p = RHS_p

    # Solve Poisson equation
    p = poissonSolver(RHS_p, kappa_x_mod_sq, kappa_y_mod_sq)
    
    # Compute first derivatives of pseudo-pressure
    dp_dx, dp_dy = np.zeros_like(u_star), np.zeros_like(v_star)
    for j in range(Ny):
        dp_dx[:, j] = np.matmul(Dx_back, p[:, j])
    for i in range(Nx):
        dp_dy[i, :] = np.matmul(Dy_back, p[i, :])

    return dp_dx, dp_dy

def poissonSolver(RHS_p, kappa_x_mod_sq, kappa_y_mod_sq):
    """
    Goal: Solve the Poisson equation

    Inputs:
    RHS_p           - type: float64
                    - Right-hand-side of poisson equation
    kappa_x_mod_sq  - type: float64
                    - Modified wavenumber in x
    kappa_y_mod_sq  - type: float64
                    - Modified wavenumber in y
    
    Outputs:
    p           - type: float64
                - Pseudo-pressure
    """

    # Fourier transform of RHS_p
    RHS_p_hat = np.fft.fft2(RHS_p)

    # Solve Poisson equation in Fourier space
    p_hat = np.divide(-RHS_p_hat, kappa_x_mod_sq + kappa_y_mod_sq) # division by 0 yields error (ok)
    p_hat[0, 0] = 0.1 # artificial fix

    # Inverse Fourier transform of pseudo-pressure
    p = np.real(np.fft.ifft2(p_hat))

    return p

if __name__ == "__main__":

    # Spatial domain and resolution
    #Nx = 64
    Ny = Nx
    DOMAIN_X, DOMAIN_Y = 2*np.pi, 2*np.pi
    dx, dy = DOMAIN_X/Nx, DOMAIN_Y/Ny

    # 2-D staggered construction grids (for nodes, v, u)
    grid_x, grid_y = np.arange(0.0, DOMAIN_X, dx), np.arange(0.0, DOMAIN_Y, dy)
    grid_x_stagg, grid_y_stagg = np.arange(0.0 + dx/2, DOMAIN_X + dx/2, dx), np.arange(0.0 + dy/2, DOMAIN_Y + dy/2, dy)
    
    # 2-D grid staggered up (for nodes, u, v, pressure)
    grid_xy_x, grid_xy_y = np.meshgrid(grid_x, grid_y, indexing='ij')
    grid_u_x, grid_u_y = np.meshgrid(grid_x, grid_y_stagg, indexing='ij')
    grid_v_x, grid_v_y = np.meshgrid(grid_x_stagg, grid_y, indexing='ij')
    grid_p_x, grid_p_y = np.meshgrid(grid_x_stagg, grid_y_stagg, indexing='ij')

    # Single derivative forwards, backwards (x, y)
    kxfwd, kxback = [-np.ones(Nx), np.ones(Nx-1)], [-np.ones(Nx-1), np.ones(Nx)]
    offset_xfwd, offset_xback = [0, 1], [-1, 0]
    Dx_fwd, Dx_back = diags(kxfwd, offset_xfwd).toarray(), diags(kxback, offset_xback).toarray()
    Dx_fwd, Dx_back = (1/dx)*Dx_fwd, (1/dx)*Dx_back
    kyfwd, kyback = [-np.ones(Ny), np.ones(Ny-1)], [-np.ones(Ny-1), np.ones(Ny)]
    offset_yfwd, offset_yback = [0, 1], [-1, 0]
    Dy_fwd, Dy_back = diags(kyfwd, offset_yfwd).toarray(), diags(kyback, offset_yback).toarray()
    Dy_fwd[-1, 0], Dy_back[0, -1] = 1, -1
    Dy_fwd, Dy_back = (1/dy)*Dy_fwd, (1/dy)*Dy_back

    # Define test function
    u_init = lambda x, y : np.cos(x)
    v_init = lambda x, y : np.sin(y)

    # Initialize random field
    u_star, v_star = np.zeros_like(grid_u_x), np.zeros_like(grid_v_y)
    for i in range(Nx):
        for j in range(Ny):
            u_star[i, j] = u_init(grid_u_x[i, j], grid_u_y[i, j])
            v_star[i, j] = v_init(grid_v_x[i, j], grid_v_y[i, j])

    # Run projection step
    u_next, v_next = compute_projection_step(u_star, v_star, Dx_fwd, Dx_back, Dy_fwd, Dy_back)

    # Check zero divergence
    du_next_dx, dv_next_dy = np.zeros_like(u_next), np.zeros_like(u_next)
    for j in range(Ny):
        du_next_dx[:, j] = np.matmul(Dx_fwd, u_next[:, j])
    for i in range(Nx):
        dv_next_dy[i, :] = np.matmul(Dy_fwd, v_next[i, :])
    div_final = du_next_dx + dv_next_dy
    print("div_glob_final: ", div_final)

    fig, ax1 = plt.subplots()
    imageTruth  = ax1.imshow(div_final, cmap='coolwarm')
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(imageTruth, ax=ax1, cax=cax1)#, ticks=[-3, 0, 3], boundaries=np.linspace(-3, 3, 100))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    # plt.savefig('divergence_field.pdf')
    plt.show()

    sys.exit()
    
    raise Exception("No testing written! Please write testing or run 'RunSolver.py' to run the main code")