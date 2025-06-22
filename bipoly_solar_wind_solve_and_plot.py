# Package used is the code
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from function_to_run_bipoly_dakeyo2025a import function_to_run_bipoly as func_bipoly


def solve_bipoly(N, L, gamma_p_max, gamma_e_max, Tpc, Tec, r_poly_p, r_poly_e
                                      ,fm, r_exp, sig_exp, plot_f, plot_gamma, plot_unT
                                      , plot_energy, data, plot_data):
    
    
    #########################################
    # Initialization of physical quantities
    mp = 1.67e-27
    M = 1.99e30
    G = 6.67e-11
    k = 1.38e-23
    r0 = 6.96e8
    L_min = 10 * r0
    #########################################
    
    def differential(x):
        N = len(x)
        dxdt = np.zeros_like(x)
        for i in range(N-1):
            dxdt[i] = (x[i+1] - x[i])
            dxdt[-1] = dxdt[-2]            
        return dxdt
    
    
    def derive_cen(t,x):
        N = len(x)
        dxdt = np.zeros_like(x)
        for i in range(1, N-1):
            dxdt[i] = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1])
            dxdt[0] = dxdt[1]
            dxdt[-1] = dxdt[-2]
            
        return dxdt
    
    
    if( np.size(gamma_p_max) ==1): gamma_p_max = np.array([1,1])*gamma_p_max
    if( np.size(gamma_e_max) ==1): gamma_e_max = np.array([1,1])*gamma_e_max
    
    # Set the dimension of the input quantities that required it
    bol_poly = True
    if(r_poly_p == float('inf')): 
        r_poly_p = L
        gamma_p_max[1] = gamma_p_max[0]
        bol_poly = True
    if(r_poly_e == float('inf')): 
        r_poly_e = L
        gamma_e_max[1] = gamma_e_max[0]
        bol_poly = True
    
    r_poly_p = r_poly_p * r0
    r_poly_e = r_poly_e * r0
    r_exp = r_exp * r0
    sig_exp = sig_exp * r0
    
    if(L < L_min):
        print('------------------------------------------------------------------------')
        print('WARNING : L too small --> L < 10*r0 not considered')
        print('--> Calculation done with L = 10*r0')
        print('------------------------------------------------------------------------')
        
        
    
    
    
    #########################################
    # Graphic parameters
    ep_trait = 2
    pol = 14
    
    color_reg = [ 'red', 'cyan', 'blue' ]
    xvalues = np.array([1, 10, 100, 200])
    #########################################
    
    N = int(N)
    r = np.geomspace(r0, L, N) 
    
    ind_r_poly_p = np.argmin( abs(r - r_poly_p) ) 
    ind_r_poly_e = np.argmin( abs(r - r_poly_e) ) 
    
    r_poly_min = np.min([r_poly_e, r_poly_p])
    r_poly_max = np.max([r_poly_e, r_poly_p])
    
    
    if(fm<=0): 
        print('------------------------------------------------------------------------')
        print('ERROR : fm <= 0 --> unphysical value')
        print('------------------------------------------------------------------------')
        sys.exit()
    elif(fm<1): 
        print('------------------------------------------------------------------------')
        print('WARNING : fm < 1 --> sub-spherical expansion')
        print('------------------------------------------------------------------------')
    elif( sig_exp<=5e-3*r0 ): 
        print('------------------------------------------------------------------------')
        print('ERROR : sig_exp too small --> increase sig_exp value')
        print('------------------------------------------------------------------------')
        sys.exit()
    elif( r_exp<r0 ): 
        print('------------------------------------------------------------------------')
        print('ERROR : r_exp <= r0 --> unphysical value')
        print('------------------------------------------------------------------------')
        sys.exit()
        
        
    # Calculation of the expansion factor profile (Kopp & Holzer 1976)
    f1 = 1 - (fm -1) * np.exp( (r0 - r_exp)/sig_exp )
    f = ( fm + f1 * np.exp( - ( r - r_exp )/ sig_exp ) ) / ( 1 + np.exp( - (r - r_exp)/sig_exp) )
    
    
    
    
    ##############################################################
    # Computation of the isopoly solutions
    
    (r, rc_iso, u_h, n_h, Tp, Te, cs_T, ind_rc_poly, ind_rc_vect,
         gamma_p, gamma_e, bol_supersonic) = func_bipoly(r0, Tec, Tpc,
                                          gamma_p_max, gamma_e_max, ind_r_poly_p, ind_r_poly_e,
                                          L ,N, f, bol_poly)
    ##############################################################
    
    
    
    # Adjustement of n profile to observations (if no data--> dakeyo2022)
    ##########################################################"  
    ind_r1au = np.argmin( abs(r - L) ) 
    if(plot_data):
        np_med_1au = np.nanmedian((data[data[:,0]>10,0] / (L/r0))**2 * data[data[:,0]>10,2])           
        n_h = n_h / n_h[-1] * np_med_1au
    else:
        np_med_1au_dakeyo2022 = np.array([9.46, 9.59, 6.99, 6.14, 5.37])
        u_1au_pop = np.array([345, 391, 445, 486, 609]) #np.array([ vp_family[-1,:] ])
        
        
        ##################################################
        num_vent = np.argmin( abs(u_1au_pop - u_h[ind_r1au]/1e3) )
        n_h = n_h / n_h[-1] * np_med_1au_dakeyo2022[num_vent]
        ##########################################################"
        
    
    
    u_h = u_h / 1e3
    cs_T = cs_T / 1e3
    r = r / r0
    r_poly_p = r_poly_p / r0
    r_poly_e = r_poly_e / r0
    rc_iso = rc_iso / r0
    
    
    if(plot_unT | plot_gamma ):
        ind_sel_reg = [0] * 3
        ind_sel_reg[0] = np.argwhere( r < r_poly_min/r0 )
        ind_sel_reg[1] = np.argwhere( (r > r_poly_min/r0) & (r < r_poly_max/r0) )
        ind_sel_reg[2] = np.argwhere( r > r_poly_max/r0 )
        
        extrm_u = [float('inf'), 0]
        extrm_n = [float('inf'), 0]
        extrm_T = [float('inf'), 0]
        
        # Remote sensing observations 
        r_cor = np.array([1.5, 2, 3, 4])
        vd_cor = np.array([ 0, 2, 4, 8])  # Bemporad2017
        vu_cor = np.array([ 300, 350, 500, 600])    #Cranmer2002
        Tpd_cor = np.array([ 1e6, 1e6, 1.2e6, 1e6])  # Cranmer 2020 
        Tpu_cor = np.array([ 2.2e6, 3e6, 4.5e6, 6.5e6])  # Grall 1996 from Cranmer 2002
        Ted_cor = np.array([ 1e6, 1e6, 0.85e6, 0.8e6])  # Cranmer 2020
        Teu_cor = np.array([ 1.4e6, 1.7e6, 1.5e6, 1.7e6])  # Cranmer 2020
        nd_cor = np.geomspace( 5e5, 1e4, 4)  # Bemporad 2017
        nu_cor = np.geomspace( 1e9, 1e7, 4)  # Bemporad 2017
        
    
        if(plot_data==0):
            data = np.zeros((1,5))
        extrm_u[0] = np.nanmin([ np.min(u_h), np.nanmin(data[:,1]), extrm_u[0] ]) 
        extrm_u[1] = np.nanmax([ np.max(u_h), np.nanmax(data[:,1]), extrm_u[1] ]) 
        extrm_n[0] = np.nanmin([ np.min(n_h), np.nanmin(data[:,2]), extrm_n[0] ]) 
        extrm_n[1] = np.nanmax([ np.max(n_h), np.nanmax(data[:,2]), extrm_n[1], np.max(nu_cor) ]) 
        extrm_T[0] = np.nanmin([ np.min(Tp), np.min(Te), np.nanmin(data[:,3]), np.nanmin(data[:,4]), extrm_T[0] ]) 
        extrm_T[1] = np.nanmax([ np.max(Tp), np.max(Te), np.nanmax(data[:,3]), np.nanmax(data[:,4]), extrm_T[1], np.max(Tpu_cor) ])
     
    
    
    # Warning from speed derivative variation, if accounting too 
    # much fluctuations
    fluct_u_sol = derive_cen(r, u_h)
    if( np.max(fluct_u_sol[ind_rc_poly:]) > 10*np.max(fluct_u_sol[:ind_rc_poly]) ):
        print('------------------------------------------------------------------------')
        print('WARNING : Anormal fluctuations --> solution potentially unphysical')
        print('--> Increase r_poly_p or r_poly_e, or N')
        print('------------------------------------------------------------------------')
    
    # Preventing breeze wind solution , that are not transonic solar wind solution expected
    if(u_h[-1] < np.max(cs_T[-1])):
        print('------------------------------------------------------------------------')
        print('WARNING : Anormal terminal speed --> Possible breeze wind solution')
        print('--> Increase r_poly_p or r_poly_e, or N, or increase Tpc, Tec')
        print('------------------------------------------------------------------------')
    
    
    # Miscalculation of wind speed solution : NAN in speed vector
    if(math.isnan(u_h[0]) | math.isnan(u_h[-1])):
        print('------------------------------------------------------------------------')
        print('ERROR : Solving issue --> No solution')
        print('--> Increase N may help')
        print('------------------------------------------------------------------------')
        sys.exit()
        

    # Plot velocity
    ######################################
    if(plot_unT):
        plt.figure(figsize=(20,5))
        plt.subplot(1,3,1)
        if(plot_data):
            plt.scatter(data[:,0], data[:,1], s=75, marker='x', color='black', label='$u_{obs}$', zorder=1)
        for i in range(len(ind_sel_reg)):
            plt.plot(r[ind_sel_reg[i]], u_h[ind_sel_reg[i]], color = color_reg[i], linewidth = ep_trait, zorder=2 )
        plt.scatter(rc_iso, cs_T[ind_rc_poly], s=35, marker='o', color='black', label='Sonic point', zorder=3, alpha=0.5)
        plt.fill_between(r_cor, vd_cor, vu_cor, facecolor='green', alpha=0.12, label='$u_{corona}$', zorder=1)
        plt.xlabel('Radius ($r \: / \: r_\\odot$)', fontsize=pol)
        plt.ylabel('u (km/s)', fontsize=pol)
        plt.title('Velocity : $u_0$ = ' + str(np.round(u_h[0],2)) + 
                  ' km/s | $u_{1au}$ = '
                  + str(int(u_h[ind_r1au])) + ' km/s', fontsize=0.9*pol)
        plt.xscale('log')
        plt.xlim([ r[0], 1.15*r[-1] ])
        plt.ylim([0, 1.05*extrm_u[1] ])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.xticks(fontsize= pol)
        plt.yticks(fontsize= pol)
        ax = plt.gca()
        plt.axvspan(1, 2.5, facecolor='silver', alpha=0.5, zorder=0)
        plt.grid()
        plt.legend(fontsize=0.85*pol)
    
    
    
        # Plot density
        ########################################
        plt.subplot(1,3,2)
        if(plot_data):
            plt.scatter(data[:,0], data[:,2], s=75, marker='x', color='black', label='$n_{obs}$', zorder=1)
        for i in range(len(ind_sel_reg)):
            plt.plot(r[ind_sel_reg[i]], n_h[ind_sel_reg[i]], color = color_reg[i], linewidth = ep_trait, zorder=2 )
        plt.fill_between(np.geomspace(1,4, 4), nd_cor, nu_cor, facecolor='green', alpha=0.12, label='$n_{corona}$', zorder=1)
        plt.xlabel('Radius ($r \: / \: r_\\odot$)', fontsize=pol)
        plt.ylabel('n (#/$cm^{-3}$)', fontsize=pol)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Density : $n_0$ = ' + '%.2E' % int(n_h[0]) + ' #/$cm^{-3}$ | $n_{1au}$ = '
                 + str(int(n_h[ind_r1au])) + ' #/$cm^{-3}$' , fontsize=0.9*pol)
        plt.xlim([ r[0], 1.15*r[-1] ])
        plt.ylim([ 0.5 * extrm_n[0], 1.5 * extrm_n[1] ])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.xticks(fontsize= pol)
        plt.yticks(fontsize= pol)
        ax = plt.gca()
        plt.axvspan(1, 2.5, facecolor='silver', alpha=0.5, zorder=0)
        plt.grid()
        if(plot_data):plt.legend(fontsize=0.85*pol)
    
    
        # Plot temperature
        ###############################################
        plt.subplot(1,3,3)
        if(plot_data):
            plt.scatter(data[:,0], data[:,3], s=75, marker='x', color='black', label='$T_{p|obs}$', zorder=1)
            plt.scatter(data[:,0], data[:,4], s=75, marker='d', color='black', label='$T_{e|obs}$', zorder=1)
        plt.plot(r[r<r_poly_p], Tp[r<r_poly_p], color = color_reg[0], linewidth = ep_trait, zorder=2 )
        plt.plot(r[r>=r_poly_p], Tp[r>=r_poly_p], color = color_reg[-1], linewidth = ep_trait, zorder=2 )
        plt.plot(r[r<r_poly_e], Te[r<r_poly_e], '--', color = color_reg[0], linewidth = ep_trait, zorder=2 )
        plt.plot(r[r>=r_poly_e], Te[r>=r_poly_e], '--', color = color_reg[-1], linewidth = ep_trait, zorder=2 )
        plt.fill_between(r_cor, Tpd_cor, Tpu_cor, facecolor='green', alpha=0.12, label='$T_{p|corona}$', zorder=1)
        plt.fill_between(r_cor, Ted_cor, Teu_cor, facecolor='purple', alpha=0.12, label='$T_{e|corona}$', zorder=1)
        plt.xlabel('Radius ($r \: / \: r_\\odot$)', fontsize=pol)
        plt.ylabel('Température (MK)', fontsize=pol)
        plt.title('Temperature : $T_{p|0}$ = ' + '%.2E' % int(Tp[0]) + 
                  'K | $T_{e|0}$ = ' + '%.2E' % int(Te[0]) + 'K ', fontsize=0.9*pol)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([ r[0], 1.15*r[-1] ])
        plt.ylim([ 0.75*extrm_T[0],1.3 * extrm_T[1] ])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.xticks(fontsize= pol)
        plt.yticks(fontsize= pol)
        ax = plt.gca()
        plt.axvspan(1, 2.5, facecolor='silver', alpha=0.5, zorder=0)
        plt.grid()
        if(plot_data):plt.legend(fontsize=0.85*pol)
        plt.show()
        
    
    # Plot f
    #########################################
    if(plot_f):
        plt.figure()
        plt.plot(r, f, color='black', linewidth=1.7)
        plt.xlabel('Radius ($r \: / \: r_\\odot$)', fontsize=pol)
        plt.ylabel('f(r)', fontsize=pol)
        plt.title('Expansion factor profile', fontsize = 0.9*pol)
        #plt.xscale('log')
        plt.yscale('log')
        plt.xlim([1, 5])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.xticks(fontsize= pol)
        plt.yticks(fontsize= pol)
        plt.grid()
        plt.show()
    
    
    # Plot gamma 
    #########################################
    if(plot_gamma):
        plt.figure( figsize=(7,5))
        plt.plot(r[r<=r_poly_p], gamma_p[r<=r_poly_p], color = color_reg[0], linewidth = ep_trait )
        plt.plot(r[r>=0.99*r_poly_p], gamma_p[r>=0.99*r_poly_p], color = color_reg[-1], linewidth = ep_trait, label='$\\gamma_{p|max}$' )
        plt.plot(r[r<=r_poly_e], gamma_e[r<=r_poly_e], '--', color = color_reg[0], linewidth = ep_trait )
        plt.plot(r[r>=0.99*r_poly_e], gamma_e[r>=0.99*r_poly_e], '--', color = color_reg[-1], linewidth = ep_trait, label='$\\gamma_{e|max}$' )
        plt.xlabel('Radius ($r \: / \: r_\\odot$)', fontsize=pol)
        plt.ylabel('$\\gamma$', fontsize=pol)
        plt.xscale('log')
        plt.title('Polytropic indices profiles', fontsize = 0.9*pol)
        plt.xlim([ r[0], 1.15*r[-1] ])
        plt.ylim([0.9 , 1.05 * np.max([gamma_p_max , gamma_e_max])])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.xticks(fontsize= pol)
        plt.yticks(fontsize= pol)
        ax = plt.gca()
        plt.axvspan(1, 2.5, facecolor='silver', alpha=0.5)
        plt.grid()
        plt.legend()
        plt.show()
        
        

    
    # Calculation of the mechanical energy
    ##################################################
    r = r * r0
    u_h = u_h * 1e3
    dh = differential(r) 
    
    
    E_cin = mp / 2. * u_h**2
    E_grav = - G * M * mp / r
    E_press = np.zeros_like(E_cin)
    E_cin_int = np.zeros_like(E_cin)
    
    # Work of pressure force
    n_tilde = n_h / n_h[ind_rc_poly] 
    
    n_tilde_p = n_h / n_h[ind_r_poly_p]
    n_tilde_e = n_h / n_h[ind_r_poly_e]  
    
    F_press = np.zeros_like(E_press)
    F_press_p = np.zeros_like(E_press)
    F_press_e = np.zeros_like(E_press)
    
    
    i = 0
    F_press_p[i] = ( k / n_tilde_p[i]  ) \
        * ( Tp[i+1] * n_tilde_p[i+1] - Tp[i] * n_tilde_p[i] ) / (r[i+1] - r[i]) 
        
    F_press_e[i] = ( k / n_tilde_e[i] ) \
        * ( Te[i+1] * n_tilde_e[i+1] - Te[i] * n_tilde_e[i] ) / (r[i+1] - r[i])
    for i in range(1, N-1):
        F_press_p[i] = ( k / n_tilde_p[i]  ) \
            * ( Tp[i+1] * n_tilde_p[i+1] - Tp[i-1] * n_tilde_p[i-1] ) / (r[i+1] - r[i-1]) 
            
        F_press_e[i] = ( k / n_tilde_e[i] ) \
            * ( Te[i+1] * n_tilde_e[i+1] - Te[i-1] * n_tilde_e[i-1] ) / (r[i+1] - r[i-1])
    
    F_press_p[-1] = ( k / n_tilde_p[i]  ) \
        * ( Tp[i] * n_tilde_p[i] - Tp[i-1] * n_tilde_p[i-1] ) / (r[i] - r[i-1]) 
        
    F_press_e[-1] = ( k / n_tilde_e[i] ) \
        * ( Te[i] * n_tilde_e[i] - Te[i-1] * n_tilde_e[i-1] ) / (r[i] - r[i-1])
                
            
    E_press_p = np.cumsum(F_press_p * dh)  
    E_press_e = np.cumsum(F_press_e * dh) 
    E_press = E_press_p + E_press_e
    
    

    # Adjustment of the thermal energy to be positive at least of the order of gravitational potential 
    # There not influence on the modeled speed, since it's defined to a constant, however the total energy
    # is positive for existing wind solution
    diff = - E_grav[-1]
    E_press = E_press + diff + abs(E_press[-1])

    
    
    
    
    # Calcul par equation de Bernouilli du chauffage hors cas adiabatique
    
    F_press_adiab_p = np.zeros_like(E_press)
    F_press_adiab_e = np.zeros_like(E_press)
    Pp = k * n_h *1e6 * Tp
    Pe = k * n_h *1e6 * Te
    
    
    i = 0
    F_press_adiab_p[i] = Pp[0]/( n_h[0]**(5/3) ) * ( n_h[i+1]**(5/3) - n_h[i]**(5/3)  ) / (r[i+1] - r[i]) * 1/(1e6*n_h[i])
    F_press_adiab_e[i] = Pe[0]/( n_h[0]**(5/3) ) * ( n_h[i+1]**(5/3) - n_h[i]**(5/3)  ) / (r[i+1] - r[i]) * 1/(1e6*n_h[i])
    
    for i in range(1, N-1):
           
        F_press_adiab_p[i] = Pp[0]/( n_h[0]**(5/3) ) * ( n_h[i+1]**(5/3) - n_h[i-1]**(5/3)  ) / (r[i+1] - r[i-1]) * 1/(1e6*n_h[i])
        F_press_adiab_e[i] = Pe[0]/( n_h[0]**(5/3) ) * ( n_h[i+1]**(5/3) - n_h[i-1]**(5/3)  ) / (r[i+1] - r[i-1]) * 1/(1e6*n_h[i])
    
    F_press_adiab_p[-1] = Pp[0]/( n_h[0]**(5/3) ) * ( n_h[i]**(5/3) - n_h[i-1]**(5/3)  ) / (r[i] - r[i-1]) * 1/(1e6*n_h[i])
    F_press_adiab_e[-1] = Pe[0]/( n_h[0]**(5/3) ) * ( n_h[i]**(5/3) - n_h[i-1]**(5/3)  ) / (r[i] - r[i-1]) * 1/(1e6*n_h[i])
    
    E_press_adiab_p = np.cumsum(F_press_adiab_p * dh)  # k * 5/2 * Tp + k * 5/2 * Tp
    E_press_adiab_e = np.cumsum(F_press_adiab_e * dh)
    
    
    
    # Reajustement des energies par rapport aux valeurs modeliser limite (offset a cause de densite)
    
    E_press_adiab_max_p = k * 5/2 * Tp[-1] 
    E_press_adiab_max_e = k * 5/2 * Te[-1]
    
    diff_p_adiab = E_press_adiab_p[-1] - E_press_adiab_max_p
    diff_e_adiab = E_press_adiab_e[-1] - E_press_adiab_max_e
    
    E_press_adiab_p = E_press_adiab_p - diff_p_adiab
    E_press_adiab_e = E_press_adiab_e - diff_e_adiab
    
    E_press_adiab = E_press_adiab_p + E_press_adiab_e
    #E_press_adiab = E_press_adiab - E_press_adiab[-1] + E_press[-1] 
    
    
    E_cin = E_cin /(1.6e-19*1e3)
    E_press = E_press /(1.6e-19*1e3) #+ 4 
    E_press_p = E_press_p /(1.6e-19*1e3) #+ 4 
    E_press_e = E_press_e /(1.6e-19*1e3) #+ 4 
    E_press_adiab = E_press_adiab /(1.6e-19*1e3) #+ 4
    E_press_adiab_p = E_press_adiab_p /(1.6e-19*1e3) #+ 4 
    E_press_adiab_e = E_press_adiab_e /(1.6e-19*1e3) #+ 4  
    E_grav = E_grav /(1.6e-19*1e3)
    #E_tot = E_tot /(1.6e-19*1e3)
    
    E_tot = E_cin + E_press + E_grav
    delta_E = E_press - E_press_adiab
    
    r = r / r0
    
    
    xvalues = np.array([1, 10, 100, 200])
    
    if(plot_energy):
        plt.figure( figsize=(7,5))
        plt.plot(r, E_cin, linewidth=2 , color='blue', label = '$E_c$')
        plt.plot(r, E_grav, linewidth=2 , color='green', label = '$E_g$')
        plt.plot(r, E_press, linewidth=2, color='red' , label = '$E_{th}$')
        #plt.plot(r, E_press_adiab, ':', linewidth=2, color='gray' , label = '$E_{th|5/3}$')
        #plt.plot(r, E_press_adiab_p , label = 'E_press_adiab_p')
        #plt.plot(r, E_press_adiab_e , label = 'E_press_adiab_e')
        #plt.plot(r, E_press - E_press_adiab,'--' , linewidth=2, color='red' , label = '$\Delta$E')
        plt.plot(r, E_tot, '--', linewidth=2, color='black' , label = '$E_{tot}$')
        plt.xlim([ r[0], 1.15*r[-1] ])
        plt.xscale('log')
        plt.ylim([1.15*np.min([E_cin, E_grav, E_press, E_tot]), 1.07*np.max([E_cin, E_grav, E_press, E_tot]) ])
        #plt.title('E_mec')
        plt.xlabel('Radius ($r \: / \: r_\\odot$)', fontsize=0.9*pol)
        plt.ylabel('Energy (KeV)', fontsize=0.9*pol)
        plt.xticks(fontsize= 0.9*pol)
        plt.yticks(fontsize= 0.9*pol)
        plt.grid()
        plt.legend(loc = 1, fontsize= 0.9*pol)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.axvspan(1, 2.5, facecolor='silver', alpha=0.5)
        plt.show()
        
   
    return(r, n_h, u_h, Tp, Te, gamma_p, gamma_e, ind_rc_poly, f, bol_supersonic)
        
        
        
        
        
        
