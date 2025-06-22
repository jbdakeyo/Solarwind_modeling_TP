# Importation required to run this code
import bipoly_solar_wind_solve_and_plot as bpsw
import streamline_calc_dakeyo2024a as stream 
import matplotlib.pyplot as plt
import numpy as np

#########################################
# Data to plot : Rivera et al. 2025
'''
r_data_RA  = [13.7, 130.6]
u_data_RA = [ 311, 451]
n_data_RA = [1043, 11]
Tp_data_RA = [ 9e5, 2.2e5]
Te_data_RA = [ 4.7e5, 1.7e5]

data = np.column_stack(( r_data_RA , u_data_RA, n_data_RA, Tp_data_RA, Te_data_RA ))
'''
##########################################################"
# Loading observations HELIOS and PSP

data = np.loadtxt('HELIOS_PSP_med_profile_5_vents_r_v_Tp_Te_ne.txt')
data[data==0] = np.nan

nb_vent = 5
choix_fam = 4
r_data = data[:, 0]
u_data = data[:, 1 : nb_vent + 1] 
Tp_data = data[:, 1*nb_vent +1: 2*nb_vent +1]
Te_data = data[:, 2*nb_vent +1: 3*nb_vent +1]
n_data = data[:, 3*nb_vent +1: 4*nb_vent +1]

data = np.column_stack((r_data , u_data[:,choix_fam], n_data[:,choix_fam],
                        Tp_data[:,choix_fam], Te_data[:,choix_fam]))










#########################################
# Inputs of the model  
#########################################

# Length of the output model
N = 5e3         # (Note : the number of points N has been inscreased compared to previous cases)
L = 1.496e11      # set to 1au by default

# Polytropic indexes
gamma_p_values = [1]
gamma_e_values = [1]

# Coronal temperature
Tpc = 1e6
Tec = 1e6

# Isothermal radius (in solar radii)
r_poly_p = float('inf')
r_poly_e = float('inf')

# Expansion factor parameters
fm = 1
r_exp = 1.9          # in solar radii
sig_exp = 0.1       # in solar radii
#########################################
# Plotting option 
plot_f = False
plot_gamma = False

plot_unT = True
plot_energy = False
plot_data = True


###############################################################
# Running the main function
(r, n, u, Tp, Te, gamma_p, gamma_e, ind_rc, f, bol_super) = bpsw.solve_bipoly(
                                        N, L, gamma_p_values, gamma_e_values, 
                                        Tpc, Tec, r_poly_p, r_poly_e,
                                        fm, r_exp, sig_exp, plot_f, 
                                        plot_gamma, plot_unT, 
                                        plot_energy, data, plot_data)
###############################################################

























#########################################
# Streamline tracing 
#########################################
stream_calc = False
plot_streamline = False
# Probe location for streamline tracing
phi_sat = 10     # in degrees

# Streamline calculation
if(stream_calc):
    (r_phi, phi, v_alf, u_phi) = stream.streamline_calc(r, n, u, phi_sat, plot_streamline)  
###############################################################
plt.show()


