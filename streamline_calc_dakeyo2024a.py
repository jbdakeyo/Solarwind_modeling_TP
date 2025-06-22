import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


##########################################################################
# Calculation of the Parker's like streamline based on u(r)
##########################################################################

def streamline_calc(r, nsw, Vsw, p_sat, plot_streamline):    
        

    def differential(x):
        N = len(x)
        dxdt = np.zeros_like(x)
        for i in range(N-1):
            dxdt[i] = (x[i+1] - x[i])
            dxdt[-1] = dxdt[-2]            
        return dxdt
    
    def integ(r,dxdr):
        x = np.zeros_like(dxdr)
        dr = differential(r)
        #ind_sel = np.argwhere( (r > r1) & (r < r2) )
        x = np.cumsum( dxdr * dr )   
        return x
    
    def ParkerSpiral_corot(p_sat, r_sat, Rss ,r_mod , u_mod, full_vect, N, r_corot, u_phi_mod):
        
        # Re-sampling of the input vectors
        f_ur = interp1d(r_mod, u_mod)
        f_uphi = interp1d(r_mod, u_phi_mod)
        
        r = np.linspace(Rss, r_sat, int(N) )
        phi = np.zeros_like(r) #+ p_sat            # On se place à p_sat pour debuté la spirale au niveau du satellite
        
        
        # Interpolation of the given speed profile
        u_r = f_ur(r)
        u_phi = f_uphi(r)
        
    
        # Streamline local angle
        phi -=  integ(r,  (omega_sun - u_phi/r) / u_r )
        phi = phi - phi[-1] + p_sat
        if(full_vect != True):
            phi = phi[-1]
    
        return (phi)%(2*np.pi)
    
    
    
    
    # Sattelite coordinates   (in degrees)
    #############################################################
    L = r[-1]
    t_sat = 0
    r_sat = r[-1]
    p_sat = p_sat * np.pi/180
    
    ind_r1au = np.argmin( abs(r - L) )
    u_1au_pop = np.array([345, 391, 445, 486, 609]) 
    num_vent = np.argmin( abs(u_1au_pop - Vsw[ind_r1au]/1e3) )
    
    rss = 2.5 
    nb_pts_spiral = 1000     # Number of points in the Parker Spiral trajectory ( from rss to r_sat )
    
    # Calculation of the tangential speed from modeling
    ##################################################
    # Differential rotation depending on latitude    
    A = 14.713
    B = -2.396
    C = -1.787
    #theta_sat = t_sat / np.pi * 180
    omega_sun = A + B*np.sin(t_sat)**2 + C*np.sin(t_sat)**4  #angular speed in degree per day
    omega_sun = omega_sun / (24* 3600) * np.pi/180    # angular speed in rad per sec
    Br_wind = np.array([2.37, 2.74, 3.18, 3.33 , 3.15]) * 1e-9
    
    # Physical quantities
    mp = 1.67e-27
    me = 9.1e-31
    mu0 = 1.26e-6
    r0 = 6.96e8
        
    
    # Extrapolated magnetic field from spacecraft
    Br_mod_park = Br_wind[num_vent] * (r_sat/r)**2 
    
    # Aflven speed (in absolute value)
    v_alf = abs( Br_mod_park / np.sqrt( mu0 * (mp+me) * nsw*1e6) ) 
    
    # Locate where the wind is equal to the alfven speed
    ind_alf = np.argmin( (abs( Vsw - v_alf ) ) ,axis=0)
    
    
    
    Ma = Vsw / v_alf        # Aflven Mach number 
    u_phi = (omega_sun*r*r0 / (v_alf[ind_alf])) * ( v_alf[ind_alf] - Vsw ) / ( 1 - Ma**2 ) # Tangential speed
    # Around u = v_alf, we approximate the curve to a line (discontinuity because of a pole)
    ind_diverg = np.argwhere( abs(Ma**2 - 1) < 0.15 )[:,0] # pourcentage around which to apply the criteria
    # Treat the zero on the expression of u_phi
    if(ind_diverg.size != 0):
        u_phi[ind_diverg[0]:ind_diverg[-1]] = np.linspace(u_phi[ind_diverg[0]], u_phi[ind_diverg[-1]], ind_diverg[-1]-ind_diverg[0])
    
    
    # Parker spiral calculation
    phi = ParkerSpiral_corot(p_sat, r_sat*r0, 0.99*rss*r0, r*r0, Vsw , True, nb_pts_spiral, 0, u_phi)
    
    
    # Vector from rss to r_sat
    r_phi = np.linspace(rss, r[-1], nb_pts_spiral)
    
    
    # Conversion from radian to degre
    phi_deg = phi / ( 2*np.pi ) * 360
    
    if(plot_streamline):
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(phi, r_phi, color = 'black', linewidth=1.8)
        #ax.plot(phi[:,i] + np.pi, r, color = color_fam[i])
        ax.plot(np.linspace(0, 2*np.pi, 50), rss * np.linspace(1,1,50), color='grey' )
        ax.set_rmax(r_sat)
        #ax.set_rmax(10)
        ax.set_rticks([10, 100, 200])  # Less radial ticks
        ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
        ax.grid(True)
        ax.set_title('Streamline :  $\\phi_{rss}$ =' +str(np.round(phi_deg[0],1))+ 
                     '° | $\\phi_{sat}$ = ' + str(np.round(phi_deg[-1],1))+ '°', va='bottom')
        #ax.set_rscale('log')
        plt.show()
        
        
    
    return(r_phi, phi, v_alf, u_phi)
    
        