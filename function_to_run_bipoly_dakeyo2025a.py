# Package used is the code
import math
import numpy as np
import sys
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import warnings 

  
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 

###############################################################
###### Main function calculating the solar wind solution ######
###############################################################

def function_to_run_bipoly(r0, Tec, Tpc, gamma_p_max, gamma_e_max, ind_r_poly_p, ind_r_poly_e,
                                                                    L ,N, f, bol_poly):
    
    r_transi_p = 0
    r_transi_e = 0
     
    if((any(np.array(gamma_p_max) <= 0)) | (any(np.array(gamma_e_max) <= 0))): 
        print('-------------------------------------------------------')
        print('ERROR : gamma_p or gamma_e <= 0 --> Undefined thermal regime')
        print('-------------------------------------------------------')
        sys.exit()
    elif((Tpc <= 0) | (Tec <= 0)): 
        print('-------------------------------------------------------')
        print('ERROR : Tp0 or Te0 <= 0 --> Unphysical quantities')
        print('-------------------------------------------------------')
        sys.exit()
    elif((ind_r_poly_p == 0) | (ind_r_poly_e == 0)): 
        print('-------------------------------------------------------')
        print('ERROR : r_poly_p or r_poly_e <= rc or <= 0')
        print('--> Misdefined quantities or Undefined solution')
        print('-------------------------------------------------------')
        sys.exit()
        
    if(bol_poly != 1):
        if((ind_r_poly_p >= N-1) | (ind_r_poly_e >= N-1)): 
            print('-------------------------------------------------------')
            print('WARNING : r_poly_p or r_poly_e > the size of the domain')
            print('--> Decrease r_poly_p or r_poly_e')
            print('-------------------------------------------------------')
          
    # Derivative function
    def derive(t,x):
        N = len(x)
        dxdt = np.zeros_like(x)
        for i in range(N-1):
            dxdt[i] = (x[i+1] - x[i]) / (t[i+1] - t[i])
            dxdt[-1] = dxdt[-2]
            
        return dxdt  
        
    # Centered derivative function
    def derive_cen(t,x):
        N = len(x)
        dxdt = np.zeros_like(x)
        for i in range(1, N-1):
            dxdt[i] = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1])
            dxdt[0] = dxdt[1]
            dxdt[-1] = dxdt[-2]
            
        return dxdt
    
    # Differential function
    def differential(x):
        N = len(x)
        dxdt = np.zeros_like(x)
        for i in range(N-1):
            dxdt[i] = (x[i+1] - x[i])
            dxdt[-1] = dxdt[-2]            
        return dxdt
    
    
    ################################################
    ### Forward calculation in finite difference ###
    ################################################
    def grad_forwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e) : 
    
        ### Percentage around rc and uc where the equations are linearized to treat the sonic point calculation
        lim_pourc = 0.005 

        Etot_ref = 1/2*mp*u_h[ind_rc]**2 + gamma_p[ind_rc]/abs(gamma_p[ind_rc]-1)*k*Tpc\
            + gamma_e[ind_rc]/abs(gamma_e[ind_rc]-1)*k*Tec - G*M*mp/r[ind_rc]
        #print(Etot_ref)
        for i in range(ind_start, ind_stop):
            cs_p_i = cs_p[i]
            cs_e_i = cs_e[i]
            ind_rc_i = ind_rc_vect[i]
            
            gamma_p_i = gamma_p[i]
            gamma_e_i = gamma_e[i]
            
            ind_rc_old = ind_rc_vect[ind_start-1]

            ###########################################################
            # Below r_poly_p
            if( (i < ind_r_poly_p) & (i < ind_r_poly_e) ):
                xp = ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_rc_i] * r[ind_rc_i]**2 * f[ind_rc_i]) )**( 1 - gamma_p_i )
                xe = ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_rc_i] * r[ind_rc_i]**2 * f[ind_rc_i]) )**( 1 - gamma_e_i )
            
            # In between r_poly_p and r_poly_e
            elif( (i >= ind_r_poly_e) & ( i < ind_r_poly_p ) ):
                xp = ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_rc_old] * r[ind_rc_old]**2 * f[ind_rc_old]) )**( 1 - gamma_p_i )
                xe_start = xe_vect[ind_r_poly_e] 
                xe = xe_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_r_poly_e] * r[ind_r_poly_e]**2 * f[ind_r_poly_e]) )**( 1 - gamma_e_i )
           
            elif( (i >= ind_r_poly_p) & ( i < ind_r_poly_e ) ):
                xe = ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_rc_old] * r[ind_rc_old]**2 * f[ind_rc_old]) )**( 1 - gamma_e_i )
                xp_start = xp_vect[ind_r_poly_p] 
                xp = xp_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_r_poly_p] * r[ind_r_poly_p]**2 * f[ind_r_poly_p]) )**( 1 - gamma_p_i )
                
            else:
                xp_start = xp_vect[ind_start-1] 
                xp = xp_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_start] * r[ind_start]**2 * f[ind_start]) )**( 1 - gamma_p_i )
                xe_start = xe_vect[ind_start-1] 
                xe = xe_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_start] * r[ind_start]**2 * f[ind_start]) )**( 1 - gamma_e_i )
    
            ###########################################################
            # Treatment of null point(s) --> implicit solving
            ###########################################################
            if( (abs(r[ind_rc_i] - r[i])/r[i] <= lim_pourc)  ):
                pourc_u = 0.01
                N_impl = 50
                u_targ = np.linspace( u_h[i]*(1+0.0001), u_h[i]*(1+pourc_u), N_impl)
                n_targ = (u_h[ind_rc] * r[ind_rc]**2 * f[ind_rc]) / (u_targ * r[i+1]**2 * f[i+1]) # adimensioned
                Tp_targ = Tpc * n_targ**(gamma_p_i)
                Te_targ = Tec * n_targ**(gamma_e_i)
                '''
                F_press_p = ( k / n_targ ) * (Tp_targ*n_targ - Tpc)/(r[i+1]-r[i])
                F_press_e = ( k / n_targ ) * (Te_targ*n_targ - Tec)/(r[i+1]-r[i])

                E_press_p = F_press_p * (r[i+1]-r[i])  
                E_press_e = F_press_e * (r[i+1]-r[i]) 
                '''
                #Etot = 1/2*mp*u_targ**2 + E_press_p + E_press_e - G*M*mp/r[i+1]
                Etot = 1/2*mp*u_targ**2 + gamma_p[i+1]/abs(gamma_p[i+1]-1)*k*Tp_targ\
                    + gamma_e[i]/abs(gamma_e[i+1]-1)*k*Te_targ - G*M*mp/r[i+1]
                #print(Etot)
                ind_best = np.argmin( abs(Etot_ref - Etot) )
                #print(ind_best)
                #print(Etot[ind_best])
                #print('')
                '''
                signe = +1
                pourc = (1 + signe*lim_pourc)
                
                AA = ( 1 - (xp*cs_p_i**2 + xe*cs_e_i**2) / (pourc*(cs_p_i**2 + cs_e_i**2)) )
                BB = ( xp*cs_p_i**2  + xe*cs_e_i**2 ) * ( 2 - f[i]*r[i]*pourc * d_inv_f[i] ) - G*M/(r[i]*pourc)  
                alpha = 1/(u_h[i] * r[i]) * BB/ AA
                if(gamma_p_i < 1.05):
                    alpha =  np.mean(alpha_sauv[i-int(N/1e3) : i]) * (r[i]/r[ind_rc])**(0.5)
                '''
                # Speed forward
                u_h[i+1] = u_targ[ind_best]
                alpha = (u_h[i+1] - u_h[i])/(r[i+1] - r[i])
            ###########################################################
            else: 
                AA = ( 1 - (xp*cs_p_i**2 + xe*cs_e_i**2)/u_h[i]**2)
                BB = (cs_p_i**2 *xp + cs_e_i**2 *xe) * ( 2 - f[i]*r[i] * d_inv_f[i] ) - G*M/r[i]  
                alpha = 1/(u_h[i] * r[i]) * BB/ AA

    
                # Speed forward
                u_h[i+1] = u_h[i] + (r[i+1] - r[i]) * alpha
                ###########################################################
            
            # Local sound speed calculation
            xp_vect[i+1] = xp
            xe_vect[i+1] = xe
            cs_T_vect[i+1] = np.sqrt(cs_p_i**2 *xp + cs_e_i**2 *xe)
            alpha_sauv[i+1] = alpha
    
    #################################################
    ### Backward calculation in finite difference ###
    #################################################
    def grad_backwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e) : 
        
        ### Percentage around rc and uc where the equations are linearized to treat the sonic point calculation
        lim_pourc = 0.002
        
        
        for i in range(ind_start, ind_stop, -1):
            cs_p_i = cs_p[i]
            cs_e_i = cs_e[i]
            ind_rc_i = ind_rc_vect[i]
            
            gamma_p_i = gamma_p[i]
            gamma_e_i = gamma_e[i]
            
            ind_rc_old = ind_rc_vect[ind_start+1]
            gamma_p_old = gamma_p[ind_start+1]
            gamma_e_old = gamma_e[ind_start+1]
            
            # Above rc
            if(i > ind_rc):
                # Above r_poly_max
                if( (i > ind_r_poly_p) & (i > ind_r_poly_e) ):
                    xp_start = xp_vect[ind_start] 
                    xp = xp_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_start] * r[ind_start]**2 * f[ind_start]) )**( 1 - gamma_p_i )
                    xe_start = xe_vect[ind_start] 
                    xe = xe_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_start] * r[ind_start]**2 * f[ind_start]) )**( 1 - gamma_e_i )
                
                # In between r_poly_p and r_poly_e
                elif( (i <= ind_r_poly_e) & ( i > ind_r_poly_p ) ):
                    xp = ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_rc_old] * r[ind_rc_old]**2 * f[ind_rc_old]) )**( 1 - gamma_p_i )
                    xe_start = xe_vect[ind_start+1] 
                    xe = xe_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_r_poly_e] * r[ind_r_poly_e]**2 * f[ind_r_poly_e]) )**( 1 - gamma_e_i )
               
                elif( (i <= ind_r_poly_p) & ( i > ind_r_poly_e ) ):
                    xe = ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_rc_old] * r[ind_rc_old]**2 * f[ind_rc_old]) )**( 1 - gamma_e_i )
                    xp_start = xp_vect[ind_start+1] 
                    xp = xp_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_r_poly_p] * r[ind_r_poly_p]**2 * f[ind_r_poly_p]) )**( 1 - gamma_p_i )
                
    
                elif( (i <= ind_r_poly_p) & (i <= ind_r_poly_e) ):
                    
                    xp_start = xp_vect[ind_start] 
                    xp = xp_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_start] * r[ind_start]**2 * f[ind_start]) )**( 1 - gamma_p_i )
                    xe_start = xe_vect[ind_start] 
                    xe = xe_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_start] * r[ind_start]**2 * f[ind_start]) )**( 1 - gamma_e_i )

            if(i <= ind_rc):
                if( (i > ind_r_poly_p) & (i > ind_r_poly_e) ):
                    xp_start = xp_vect[ind_start] 
                    xp = xp_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_start] * r[ind_start]**2 * f[ind_start]) )**( 1 - gamma_p_i )
                    xe_start = xe_vect[ind_start] 
                    xe = xe_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_start] * r[ind_start]**2 * f[ind_start]) )**( 1 - gamma_e_i )
                
                # In between r_poly_p and r_poly_e
                elif( (i <= ind_r_poly_e) & ( i > ind_r_poly_p ) ):
                    xp = ( (u_h[i] * r[i]**2 * f[i])/(cs_T_vect[ind_rc_old] * r[ind_rc_old]**2 * f[ind_rc_old]) )**( 1 - gamma_p_i )
                    xe_start = xe_vect[ind_start+1] 
                    xe = xe_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_r_poly_e] * r[ind_r_poly_e]**2 * f[ind_r_poly_e]) )**( 1 - gamma_e_i )
               
                elif( (i <= ind_r_poly_p) & ( i > ind_r_poly_e ) ):
                    xe = ( (u_h[i] * r[i]**2 * f[i])/(cs_T_vect[ind_rc_old] * r[ind_rc_old]**2 * f[ind_rc_old]) )**( 1 - gamma_e_i )
                    xp_start = xp_vect[ind_start+1] 
                    xp = xp_start * ( (u_h[i] * r[i]**2 * f[i])/(u_h[ind_start] * r[ind_start]**2 * f[ind_start]) )**( 1 - gamma_p_i )
                
                elif( (i <= ind_r_poly_p) & (i <= ind_r_poly_e) ):
                    xp = ( (cs[ind_rc] * r[ind_rc]**2 * f[ind_rc])/(u_h[i] * r[i]**2 * f[i]) )**( gamma_p_i -1 )
                    xe = ( (u_h[i] * r[i]**2 * f[i])/(cs[ind_rc] * r[ind_rc]**2 * f[ind_rc]) )**( 1 - gamma_e_i )  

                
            
            # Condition around the sonic point and where u sim cs
            if( (abs(r[ind_rc_i] - r[i])/r[i] <= lim_pourc) | (abs(u_h[ind_rc_i] - u_h[i])/u_h[i] <= lim_pourc) ):
                signe = -1
                pourc = (1 + signe*lim_pourc)

                AA = ( 1 - (xp*cs_p_i**2 + xe*cs_e_i**2) /(pourc*u_h[i])**2)
                BB = ( cs_p_i**2 *xp + cs_e_i**2 *xe) * ( 2 - f[i]*r[i]*pourc * d_inv_f[i] ) - G*M/(r[i]*pourc)  
                alpha = 1/(u_h[i] * r[i]) * BB/ AA
                
            else:     
                AA = ( 1 - (xp*cs_p_i**2 + xe*cs_e_i**2)/u_h[i]**2)
                BB = (cs_p_i**2 *xp + cs_e_i**2 *xe) * ( 2 - f[i]*r[i] * d_inv_f[i] ) - G*M/r[i]  
                alpha = 1/(u_h[i] * r[i]) * BB/ AA
                
    
            # Speed backward
            u_h[i-1] = u_h[i] - (r[i] - r[i-1]) * alpha
            if((bol_supersonic) & (u_h[i-1] > cs_T_vect[i]) ):
                u_h[i-1] = cs_T_vect[i]
                
            # Local sound speed calculation
            xp_vect[i-1] = xp
            xe_vect[i-1] = xe
            cs_T_vect[i-1] = np.sqrt(cs_p_i**2 *xp + cs_e_i**2 *xe)
            alpha_sauv[i-1] = alpha
    

    # Initialization of physical quantities
    mp = 1.67e-27
    M = 1.99e30
    G = 6.67e-11
    k = 1.38e-23
    r0 = 6.96e8

    # The radial distance vector is defined logarithmically to reduce time calculation at large distance (slow quantities variation)
    r = np.geomspace(r0, L, N) 
     
    ###########################################################
    # Redefining r_poly for oversampled vector
    r_poly_p = r[ind_r_poly_p]
    r_poly_e = r[ind_r_poly_e]
    
    ind_r_transi_p = np.argmin( abs(r - r_poly_p) ) 
    ind_r_transi_e = np.argmin( abs(r - r_poly_e) ) 

    #########################################
    # Builing gamma_p and gamma_e vectors
    gamma_p0 = gamma_p_max[0]
    gamma_e0 = gamma_e_max[0]
    gamma_p1 = gamma_p_max[1]
    gamma_e1 = gamma_e_max[1]
    
    
    # Gamma isotherme --> gamma = 1
    gamma_poly0_p = np.geomspace(gamma_p0, gamma_p0, ind_r_poly_p - 1 ) 
    gamma_poly0_e = np.geomspace(gamma_e0, gamma_e0, ind_r_poly_e - 1 ) 
    
    # Gamma polytropique --> gamma > 1       
    ind_poly_p = len(r) - (ind_r_poly_p - 1 ) 
    ind_poly_e = len(r) - (ind_r_poly_e - 1 )
    
    ind_r_poly_min = np.min( [ind_r_poly_p, ind_r_poly_e] )
    ind_r_poly_max = np.max( [ind_r_poly_p, ind_r_poly_e] )
    
    gamma_poly1_p = np.geomspace(gamma_p1, gamma_p1, ind_poly_p  )  
    gamma_poly1_e = np.geomspace(gamma_e1, gamma_e1, ind_poly_e  )
    
    gamma_p = np.concatenate( (gamma_poly0_p, gamma_poly1_p), axis=0)
    gamma_e = np.concatenate( (gamma_poly0_e, gamma_poly1_e), axis=0)

    
    # Defining local vectors of cs and rc in case of sonic point assumption
    cs_p = np.sqrt( gamma_p * k * Tpc / mp )
    cs_e = np.sqrt( gamma_e * k * Tec / mp )
    cs = np.sqrt( cs_p**2 + cs_e**2 )
    cs_rc = np.sqrt( gamma_p * k * Tpc / mp  + gamma_e * k * Tec / mp )
    rc = G*M / (2 * cs_rc**2)

    
    if(all(cs) == 0): 
        print('ERROR -- Invalid input coronal temperatures')
        return
    
    
    ###########################################################
    # Variation of expansion factor related terms of momentum equation
    d_inv_f = derive_cen(r, 1/f)
    func_rc = cs**2 * (2 - r*f * d_inv_f) - G*M/r
    # Derivative of func_rc, used for the oversampling
    func_rc_prim = derive_cen(r, func_rc)
    
    
    # Oversampling : Re-sampling of r vector depending super radial expansion derivative (to better treat large gradient with finite difference)
    # Larger variation --> smaller dh
    ######################################################################
    dfdr = derive_cen(r, f)
    derive_lim_f = 0.05/(r[1]-r[0])  # Value of df/dr over which we oversample
    ratio_f = dfdr/derive_lim_f
    derive_lim_func = 100           # Value of func_rc over which we oversample
    ratio_func = abs(func_rc_prim)/derive_lim_func
    ind_fprim = np.argwhere( (ratio_func > 1) | (ratio_f >1) )[:,0] 
    r_new = []

    # Oversampling of r vector 
    for pp in ind_fprim:
        nb_pts = int(ratio_f[pp]) + int(ratio_func[pp])
        for mm in range(1, nb_pts):
            r_new.append(  (1-mm/nb_pts)*r[pp] + (mm/nb_pts)*r[pp+1]  )
    r_conc = np.concatenate((r, np.array(r_new) ),axis=-1)
    r_conc.sort()
    
    r_old = r.copy()
    f_old = f.copy()
    # Interpolation of f to match with r vector
    func_interp_fexp = interp1d( r , f, kind='linear',axis=0)
    f = func_interp_fexp(r_conc)
    r = r_conc.copy()
    
    ind_r_poly_p = np.argmin( abs(r - r_poly_p) )
    ind_r_poly_e = np.argmin( abs(r - r_poly_e) )
    
    
    
    #################################################
    # Illustrating the localized oversampling
    '''
    pol = 14
    xvalues = np.array([1, 10, 100, 200])
    dh_old = differential(r_old)
    dh_opt = differential(r)

    plt.figure()
    plt.plot(r_old/r0, dh_old, label='$dr_{(init)}$')
    plt.plot(r/r0, dh_opt, label='$dr_{(optim)}$')
    plt.xlabel('Radius ($r \: / \: r_0$)', fontsize=pol)
    plt.ylabel('Spatial step of the resolution', fontsize=pol)
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    plt.xticks(xvalues, fontsize= pol)
    plt.yticks(fontsize= pol)
    plt.grid()
    plt.legend(fontsize= 0.9*pol)
    '''
    #################################################
    
    


    
    ###########################################################
    # Redefining r_iso for oversampled vector
    r_poly_p = r[ind_r_poly_p]
    r_poly_e = r[ind_r_poly_e]
    
    ind_r_transi_p = np.argmin( abs(r - (r_poly_p + r_transi_p ) ) ) #int( (r_transi_p + r_poly_p) / dh ) - int(r[0]/ dh ) 
    ind_r_transi_e = np.argmin( abs(r - (r_poly_e + r_transi_e ) ) ) #int( (r_transi_e + r_poly_e) / dh ) - int(r[0]/ dh ) 

    
    #########################################
    # Builing gamma_p and gamma_e vectors
    gamma_p0 = gamma_p_max[0]
    gamma_e0 = gamma_e_max[0]
    gamma_p1 = gamma_p_max[1]
    gamma_e1 = gamma_e_max[1]
    
    # Gamma polytropic 0 --> gamma0
    gamma_poly0_p = np.geomspace(gamma_p0, gamma_p0, ind_r_poly_p - 1 ) 
    gamma_poly0_e = np.geomspace(gamma_e0, gamma_e0, ind_r_poly_e - 1 ) 
    
    # Gamma polytropic 1 --> gamma1
    ind_transi_p =  ind_r_transi_p - ind_r_poly_p + 1
    ind_transi_e =  ind_r_transi_e - ind_r_poly_e + 1
        
    ind_poly_p = len(r) - ( ind_r_transi_p ) 
    ind_poly_e = len(r) - ( ind_r_transi_e )
    
    ind_r_poly_min = np.min( [ind_r_poly_p, ind_r_poly_e] )
    ind_r_poly_max = np.max( [ind_r_poly_p, ind_r_poly_e] )
    
    gamma_transi_p = np.linspace(gamma_p0, gamma_p1, ind_transi_p  ) 
    gamma_transi_e = np.linspace(gamma_e0, gamma_e1, ind_transi_e  )  
    
    gamma_poly1_p = np.geomspace(gamma_p1, gamma_p1, ind_poly_p  )  
    gamma_poly1_e = np.geomspace(gamma_e1, gamma_e1, ind_poly_e  )
    
    if(len(gamma_transi_p) == 0):
        gamma_p = np.concatenate( (gamma_poly0_p, gamma_poly1_p), axis=0)
        gamma_e = np.concatenate( (gamma_poly0_e, gamma_poly1_e), axis=0)
    else:
        gamma_p = np.concatenate( (gamma_poly0_p, gamma_transi_p, gamma_poly1_p), axis=0)
        gamma_e = np.concatenate( (gamma_poly0_e, gamma_transi_e, gamma_poly1_e), axis=0)
    

    cs_p = np.sqrt( gamma_p * k * Tpc / mp )
    cs_e = np.sqrt( gamma_e * k * Tec / mp )
    cs = np.sqrt( cs_p**2 + cs_e**2 )
    cs_rc = np.sqrt( gamma_p * k * Tpc / mp  + gamma_e * k * Tec / mp )
    #cs_rc = np.sqrt( gamma_p[0] * k * Tpc / mp  + gamma_e[0] * k * Tec / mp )
    
    # To save the xp and xe values outside the loop calculation
    xp_vect = np.zeros_like(cs) + 1
    xe_vect = np.zeros_like(cs) + 1
    cs_T_vect = np.zeros_like(cs) + 1
    alpha_sauv = np.zeros_like(cs)
    
    
    
    ###########################################################
    # Numerical search of rc
    d_inv_f = derive(r, 1/f)
    dfdr = derive(r, f)
    # Function to minimize to find rc
    func_rc = cs**2 * (2 - r*f * d_inv_f) - G*M/r
    peaks, _ = find_peaks(- abs(func_rc), height= -5e-2*abs(func_rc[-1]))
    ind_rc_new = peaks[0]
    
    rc_new = r[peaks] 
    rc = rc_new[-1]
    
    func_rc_sauv = func_rc.copy()
    
    ###########################################################
    
    ###########################################################
    # Numerical search of rc of the later gamma interval
    # assuming rc is in the second thermal regime
    # At r_poly_min
    func_rc = cs[ind_r_poly_min]**2 * (2 - r*f * d_inv_f) - G*M/r
    peaks, _ = find_peaks( - abs(func_rc))
    ind_rc_poly_min = peaks[-1]
    ind_rc_new_poly_min = peaks[0]
    # At r_poly_max
    func_rc = cs[ind_r_poly_max]**2 * (2 - r*f * d_inv_f) - G*M/r
    peaks, _ = find_peaks( - abs(func_rc))
    ind_rc_poly_max = peaks[-1]
    ind_rc_new_poly_max = peaks[0]

    
    
    # Determination of the wind solution
    ############################################################
  
    # Initialization of the speed vector at the sonic point
    u_h = np.zeros_like(r)
    ind_rc = np.argmin(abs(r - rc))   

    # Setting vector of rc values depending the thermal regime
    ind_rc_vect = np.zeros_like(u_h, dtype=int)
    ind_rc_new_vect = np.zeros_like(u_h, dtype=int)
    
    if(ind_rc >= ind_r_poly_max):
        ind_rc_vect[:ind_r_poly_min] = ind_rc_poly_min
        ind_rc_vect[ind_r_poly_min: ind_r_poly_max] = ind_rc_poly_max
        ind_rc_vect[ind_r_poly_max:] = ind_rc
        
    if(ind_rc < ind_r_poly_min):
        ind_rc_vect[:ind_r_poly_min] = ind_rc
        ind_rc_vect[ind_r_poly_min: ind_r_poly_max] = ind_rc_poly_min
        ind_rc_vect[ind_r_poly_max:] = ind_rc_poly_max

    ind_rc_new_vect[:ind_r_poly_min] = ind_rc_new
    ind_rc_new_vect[ind_r_poly_min: ind_r_poly_max] = ind_rc_new_poly_min
    ind_rc_new_vect[ind_r_poly_max:] = ind_rc_new_poly_max
    
    # Setting sound speed at the sonic point
    cs_rc = cs[ind_rc]
    cs_T_vect[ind_rc] = cs[ind_rc]
    u_h[ind_rc] = cs[ind_rc].copy()      
    
    
    # Equations physiques discrétisées 
    ###############################################
    
    # Starting from r = rc, we iterate forward and backward to compute u
    bol_supersonic = False
    # Depending if rc above or below all r_poly_s
    if(ind_rc >= ind_r_poly_max): 
        ind_start = ind_rc
        ind_stop = ind_r_poly_max
        grad_backwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e) 
        
        ind_start = ind_r_poly_max
        ind_stop = ind_r_poly_min
        grad_backwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e)
        
        ind_start = ind_r_poly_min
        ind_stop = 0
        grad_backwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e)
    else:
        ind_start = ind_rc
        ind_stop = 0
        grad_backwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e)
    
    if( (any( u_h[:ind_rc] > cs_T_vect[:ind_rc] ) ) | (math.isnan(u_h[0])) | (math.isnan(u_h[-1])) ):
        
        # Test of a positive derivative of func_rc at rc_new
        if(derive_cen(r,func_rc_sauv)[ind_rc_new] > 0): 
        
            rc = rc_new[0].copy()  
            ind_rc = np.argmin(abs(r - rc))   
            u_h[ind_rc] = cs[ind_rc]       
            cs_T_vect[ind_rc] = cs[ind_rc]
            ind_rc_vect = ind_rc_new_vect.copy()
            
            
            bol_supersonic = True 
            # Depending if rc above or below all r_poly_s
            if(ind_rc >= ind_r_poly_max): 
                ind_start = ind_rc
                ind_stop = ind_r_poly_max
                grad_backwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e) 
                
                ind_start = ind_r_poly_max
                ind_stop = ind_r_poly_min
                grad_backwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e)
                
                ind_start = ind_r_poly_min
                ind_stop = 0
                grad_backwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e)
            else:
                ind_start = ind_rc
                ind_stop = 0
                grad_backwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e)
            '''
            print('-----------    -----------')
            print('SOLUTION TYPE : Supersonic before rc')
            print('-----------    -----------')
            '''
        else:
            '''
            print('-----------    -----------')
            print('SOLUTION TYPE : Supersonic at r0')
            print('-----------    -----------')
            '''
            ind_sel = np.argwhere( (u_h[:ind_rc] > cs_T_vect[:ind_rc]) | (u_h[:ind_rc]== float('nan')) )[-1,0]
            u_h[:ind_sel] = cs_T_vect[:ind_sel]
            
            
    # Depending if rc above or below all r_poly_s
    if(ind_rc >= ind_r_poly_max):  
        ind_start = ind_rc
        ind_stop = len(r) - 1
        grad_forwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e)
        
    else:
        ind_start = ind_rc
        ind_stop = ind_r_poly_min
        grad_forwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e)
        
        ind_start = ind_r_poly_min
        ind_stop = ind_r_poly_max
        grad_forwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e)
        
        ind_start = ind_r_poly_max
        ind_stop = len(r) - 1
        grad_forwrad_polytr(r, u_h, ind_rc_vect, ind_start, ind_stop, gamma_p, gamma_e)    

    
    xp_vect[ind_rc] = 1
    xe_vect[ind_rc] = 1
    alpha_sauv[ind_rc] = alpha_sauv[ind_rc-1]
    
    
    # Re-sampling to fit to the initial size of r 
    func_interp_u_h = interp1d( r, u_h, kind='linear', axis=-1)
    func_interp_gp = interp1d( r, gamma_p, kind='linear', axis=-1)
    func_interp_ge = interp1d( r, gamma_e, kind='linear', axis=-1)
    func_interp_cs = interp1d( r, cs_T_vect, kind='linear', axis=-1)
    func_interp_ind_rc = interp1d( r, ind_rc_vect, kind='linear', axis=-1)
    func_interp_xp = interp1d( r, xp_vect, kind='linear', axis=-1)
    func_interp_xe = interp1d( r, xe_vect, kind='linear', axis=-1)
    
    u_h = func_interp_u_h( r_old ) 
    f = f_old.copy()
    gamma_p = func_interp_gp( r_old )
    gamma_e = func_interp_ge( r_old )
    cs_T_vect = func_interp_cs( r_old )
    ind_rc_vect = func_interp_ind_rc( r_old )
    
    xp_vect = func_interp_xp( r_old )
    xe_vect = func_interp_xe( r_old )
    
    ind_r_poly_p = np.argmin( abs(r_old - r_poly_p) ) 
    ind_r_poly_e = np.argmin( abs(r_old - r_poly_e) ) 
    ind_rc = np.argmin( abs(r_old - rc) ) 
    r = r_old
    
    
    # Calcul of the density profile
    ################################################
    n_h = ( u_h[ind_rc] * r[ind_rc]**2 * f[ind_rc] ) / ( u_h * r**2 * f )


    # Calcul of the temperature profiles
    ###############################################
    Tp = np.zeros_like(u_h)
    Te = np.zeros_like(u_h)
    
    # Calculation of Ts from xs, with by construction:  xs = Ts / Tsc 
    Tp = xp_vect * Tpc
    Te = xe_vect * Tec

    ####################################################
    return (r, rc, u_h, n_h, Tp, Te, cs_T_vect, ind_rc, ind_rc_vect, gamma_p, gamma_e, bol_supersonic)










    