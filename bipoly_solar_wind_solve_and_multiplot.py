# Package used is the code
import os
import sys
import warnings 
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from function_to_run_bipoly_dakeyo2025a import function_to_run_bipoly as func_bipoly


def multi_solve_plot(N, L, gamma_p_max_vect, gamma_e_max_vect, 
                                        Tp0_vect, Te0_vect, r_iso_p_vect, r_iso_e_vect,
                                        fm_vect, r_exp_vect, sig_exp_vect, data, plot_data, plot_unT):
    
    if(plot_unT ==0):  sys.exit()
    
    def solve_bipoly(N, L, gamma_p_max, gamma_e_max, Tpc, Tec, r_poly_p, r_poly_e
                                          ,fm, r_exp, sig_exp):
        
        
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
        
        
        
        
        ##########################################################"  
        np_med_1au_dakeyo2022 = np.array([9.46, 9.59, 6.99, 6.14, 5.37])
        u_1au_pop = np.array([345, 391, 445, 486, 609]) #np.array([ vp_family[-1,:] ])
        
        # On ajuste n_h_all du modele au mesures complete
        ##################################################
        ind_r1au = np.argmin( abs(r - L) )
        num_vent = np.argmin( abs(u_1au_pop - u_h[ind_r1au]/1e3) )

        n_h = n_h / n_h[-1] * np_med_1au_dakeyo2022[num_vent]
        ##########################################################"
        
        
        
        u_h = u_h / 1e3
        cs_T = cs_T / 1e3
        r = r / r0
        r_poly_p = r_poly_p / r0
        r_poly_e = r_poly_e / r0
        rc_iso = rc_iso / r0
        
        
        
        ind_sel_reg = [0] * 3
        ind_sel_reg[0] = np.argwhere( r < r_poly_min/r0 )
        ind_sel_reg[1] = np.argwhere( (r > r_poly_min/r0) & (r < r_poly_max/r0) )
        ind_sel_reg[2] = np.argwhere( r > r_poly_max/r0 )
        
        
        
        
        
        
        
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
            
        
        return(r, n_h, u_h, Tp, Te, gamma_p, gamma_e, ind_rc_poly, f, bol_supersonic)
            




    # Plot all the three speed curves on the same graph: EXAMPLE
    ####################################################
    plot_unT = False
    plot_energy = False
    r0 = 6.96e8
    #########################################
    # Graphic parameters
    ep_trait = 2
    pol = 14
    
    color_reg = [ 'red', 'cyan', 'blue' ]
    xvalues = np.array([1, 10, 100, 200])
    #########################################
    
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
    
    
    
    plt.figure(figsize=(20,5))
    for j in range(len(gamma_p_max_vect)):
        # Polytropic indexes
        gamma_p_max = gamma_p_max_vect[j]
        gamma_e_max = gamma_e_max_vect[j]

        # Coronal temperature (in Kelvin)
        Tp0 = Tp0_vect[j]
        Te0 = Te0_vect[j]
        
        # Isothermal radius (in solar radii)
        r_iso_p = r_iso_p_vect[j] 
        r_iso_e = r_iso_e_vect[j]
        
        # Expansion factor parameters
        fm = fm_vect[j]
        r_exp = r_exp_vect[j]          # in solar radii
        sig_exp = sig_exp_vect[j]       # in solar radii
        
        
        ###############################################################
        # Running the main function
        (r, n, u, Tp, Te, gamma_p, gamma_e, ind_rc, f, bol_super) = solve_bipoly(
                N, L, gamma_p_max, gamma_e_max, 
                Tp0, Te0, r_iso_p, r_iso_e,
                fm, r_exp, sig_exp)
        ###############################################################
        r_iso_min = np.min([r_iso_e, r_iso_p])
        r_iso_max = np.max([r_iso_e, r_iso_p])
        
        ind_sel_reg = [0] * 3
        ind_sel_reg[0] = np.argwhere( r < r_iso_min )
        ind_sel_reg[1] = np.argwhere( (r > r_iso_min) & (r < r_iso_max) )
        ind_sel_reg[2] = np.argwhere( r > r_iso_max )
        
        
        if(plot_data==0):
            data = np.zeros((1,5))
        extrm_u[0] = np.nanmin([ np.min(u), np.nanmin(data[:,1]), extrm_u[0] ]) 
        extrm_u[1] = np.nanmax([ np.max(u), np.nanmax(data[:,1]), extrm_u[1] ]) 
        extrm_n[0] = np.nanmin([ np.min(n), np.nanmin(data[:,2]), extrm_n[0] ]) 
        extrm_n[1] = np.nanmax([ np.max(n), np.nanmax(data[:,2]), extrm_n[1], np.max(nu_cor) ]) 
        extrm_T[0] = np.nanmin([ np.min(Tp), np.min(Te), np.nanmin(data[:,3]), np.nanmin(data[:,4]), extrm_T[0] ]) 
        extrm_T[1] = np.nanmax([ np.max(Tp), np.max(Te), np.nanmax(data[:,3]), np.nanmax(data[:,4]), extrm_T[1], np.max(Tpu_cor) ])
     
        '''
        if(plot_data):
            print('-------------------')
            print('(u_0, u_1au) = ['+str(np.round(u[0],2))+', '
                  +str(np.round(u[-1],2)) + '] km/s' ) 
            print('(Tp_0, Tp_1au) = ['+str(np.round(Tp[0]/1e6,2))+', '
                  +str(np.round(Tp[-1]/1e6,2)) + '] MK' ) 
            print('(Te_0, Te_1au) = ['+str(np.round(Te[0]/1e6,2))+', '
                  +str(np.round(Te[-1]/1e6,2)) + '] MK' ) 
            print('(n_0, n_1au) = ['+str(np.round(n[0]/1e7,2))+'*1e7, '
                  +str(np.round(n[-1],2)) + '] #/cm3' )
        '''
        

        
        plt.subplot(1,3,1)
        for i in range(len(ind_sel_reg)):
            plt.plot(r[ind_sel_reg[i]], u[ind_sel_reg[i]], color = color_reg[i], linewidth = ep_trait )
        plt.scatter(r[ind_rc], u[ind_rc], s=35, marker='o', color='black', zorder=2, alpha=0.5)
        plt.xlabel('Radius ($r \: / \: r_\\odot$)', fontsize=pol)
        plt.ylabel('u (km/s)', fontsize=pol)
        plt.title('Velocity', fontsize=0.9*pol)
        plt.xscale('log')
        plt.xlim([ r[0], 1.05*r[-1] ])
        plt.ylim([0, 1.05*extrm_u[1] ])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.xticks(fontsize= pol)
        plt.yticks(fontsize= pol)
        ax = plt.gca()
        plt.axvspan(1, 2.5, facecolor='silver', alpha=0.1, zorder=0)

    
    
        # Plot density
        ########################################
        plt.subplot(1,3,2)
        for i in range(len(ind_sel_reg)):
            plt.plot(r[ind_sel_reg[i]], n[ind_sel_reg[i]], color = color_reg[i], linewidth = ep_trait )
        plt.xlabel('Radius ($r \: / \: r_\\odot$)', fontsize=pol)
        plt.ylabel('n (#/$cm^{-3}$)', fontsize=pol)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Density', fontsize=0.9*pol)
        plt.xlim([ r[0], 1.05*r[-1] ])
        plt.ylim([ 0.5 * extrm_n[0], 1.5 * extrm_n[1] ])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.xticks(fontsize= pol)
        plt.yticks(fontsize= pol)
        ax = plt.gca()
        plt.axvspan(1, 2.5, facecolor='silver', alpha=0.1, zorder=0)

    
    
    
        # Plot temperature
        ###############################################
        plt.subplot(1,3,3)
        plt.plot(r[r<r_iso_p], Tp[r<r_iso_p], color = color_reg[0], linewidth = ep_trait )
        plt.plot(r[r>=r_iso_p], Tp[r>=r_iso_p], color = color_reg[-1], linewidth = ep_trait )
        plt.plot(r[r<r_iso_e], Te[r<r_iso_e], '--', color = color_reg[0], linewidth = ep_trait )
        plt.plot(r[r>=r_iso_e], Te[r>=r_iso_e], '--', color = color_reg[-1], linewidth = ep_trait )
        plt.xlabel('Radius ($r \: / \: r_\\odot$)', fontsize=pol)
        plt.ylabel('Température (MK)', fontsize=pol)
        plt.title('Temperature', fontsize=0.9*pol)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([ r[0], 1.05*r[-1] ])
        plt.ylim([ 0.75*extrm_T[0],1.15 * extrm_T[1] ])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        plt.xticks(fontsize= pol)
        plt.yticks(fontsize= pol)
        ax = plt.gca()
        plt.axvspan(1, 2.5, facecolor='silver', alpha=0.1, zorder=0)

    plt.subplot(1,3,1)
    if(plot_data):
        plt.scatter(data[:,0], data[:,1], s=75, marker='x', color='black', label='$u_{obs}$', alpha=0.45)
        plt.fill_between(r_cor, vd_cor, vu_cor, facecolor='green', alpha=0.12, label='$u_{corona}$', zorder=1)
    plt.grid()
    plt.legend()
    plt.subplot(1,3,2)
    if(plot_data):
        plt.scatter(data[:,0], data[:,2], s=75, marker='x', color='black', label='$n_{obs}$', alpha=0.45)
        plt.fill_between(np.geomspace(1,4, 4), nd_cor, nu_cor, facecolor='green', alpha=0.12, label='$n_{corona}$', zorder=1)
    plt.grid()
    plt.legend()
    plt.subplot(1,3,3)
    if(plot_data):
        plt.scatter(data[:,0], data[:,3], s=75, marker='x', color='black', label='$T_{p|obs}$', alpha=0.45)
        plt.scatter(data[:,0], data[:,4], s=75, marker='d', color='black', label='$T_{e|obs}$', alpha=0.45)
        plt.fill_between(r_cor, Tpd_cor, Tpu_cor, facecolor='green', alpha=0.12, label='$T_{p|corona}$', zorder=1)
        plt.fill_between(r_cor, Ted_cor, Teu_cor, facecolor='purple', alpha=0.12, label='$T_{e|corona}$', zorder=1)
    plt.grid()
    plt.legend()
    plt.show()    
    
    
    
    ####################################################
    
    
    
    
    
 
      
        
        
        
        
