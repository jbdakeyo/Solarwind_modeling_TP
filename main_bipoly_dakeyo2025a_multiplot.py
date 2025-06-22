# Importation required to run this code
import bipoly_solar_wind_solve_and_multiplot as mbpsw
import numpy as np


#########################################
# Data to plot : Loading in-situ observations from HELIOS and PSP
########################################
data_insitu = np.loadtxt('HELIOS_PSP_med_profile_5_vents_r_v_Tp_Te_ne.txt')
data_insitu[data_insitu==0] = np.nan

nb_vent = 5
r_data_all = data_insitu[:, 0]
ns = nb_vent * len(r_data_all) 
u_data_all = np.resize(data_insitu[:, 1 : nb_vent + 1].T, (ns))
Tp_data_all = np.resize(data_insitu[:, 1*nb_vent +1: 2*nb_vent +1].T, (ns))
Te_data_all = np.resize(data_insitu[:, 2*nb_vent +1: 3*nb_vent +1].T, (ns))
n_data_all = np.resize(data_insitu[:, 3*nb_vent +1: 4*nb_vent +1].T, (ns))
for i in range(nb_vent-1):
    r_data_all = np.concatenate((r_data_all, data_insitu[:, 0]))
                   

data_all = np.column_stack((r_data_all , u_data_all, n_data_all
                            , Tp_data_all, Te_data_all))






#########################################
# Inputs of the model  
#########################################
nb_curves = 4

# Length of the output model
N = 5e3
L = 1.496e11      # set to 1au by default

# Polytropic indexes
gamma_p_max = np.ones((nb_curves))
gamma_e_max = gamma_p_max

# Coronal temperature (in Kelvin)
Tpc = [1e6, 1.1e6, 1.2e6, 1.3e6]
Tec = Tpc

# Isothermal radius (in solar radii)
r_iso_p = np.ones((nb_curves)) + float('inf')
r_iso_e = r_iso_p 

# Expansion factor parameters
fm = np.ones((nb_curves))
r_exp = np.ones((nb_curves))          # in solar radii
sig_exp = np.ones((nb_curves))       # in solar radii

plot_data = True
plot_unT = True


###############################################################
# Running the main function
mbpsw.multi_solve_plot(N, L, gamma_p_max, gamma_e_max, 
                       Tpc, Tec, r_iso_p, r_iso_e,
                       fm, r_exp, sig_exp, data_all, plot_data, plot_unT)
###############################################################

