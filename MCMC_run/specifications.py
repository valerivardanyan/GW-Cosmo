### The experimental and technical specifications should be given in this file ###

### --- imports --- ###
import numpy as np
import time

from colossus.cosmology import cosmology
import astropy.cosmology
### --- ####### --- ###

# This function specifies your fiducial model
def fiducial_ratios(z_lst):
    ### --- Fiducial DL-ratio and bias ratio --- ###
    xi_tmp = 1.4
    n = 1.
    DL_ratio_fiducial = (xi_tmp + (1. - xi_tmp)/(1. + z_lst)**n)#np.ones(len(z_lst))

    bias_fiducial = np.ones(len(z_lst))
    ################################################

    return [DL_ratio_fiducial, bias_fiducial]

################# Specifications #################

ell_lst = np.arange(10., 101., 1.)

### --- Comoving number densities of mergers and galaxies --- ###
n_bar_GW_z = 3.*1e-6 # T_obs ndot_GW = 3e-6h^3Mpc^-3
n_bar_gal_z = 1e-3
#################################################################

### --- Bins --- ###
### Note that you should pass the
### central locations of the bins,
### and the (const) bin width
N_gal = 12
z_min, z_max = 0.1, 2.95
delta_z = (z_max - z_min)/N_gal
z_bin_centers = np.arange(z_min + delta_z/2., z_max, delta_z)

cosmo = cosmology.setCosmology('planck18')
N_DL = 8
minmax = np.array([0.1, 3.])
DL_min, DL_max = fiducial_ratios(minmax)[0]*cosmo.luminosityDistance(minmax)  # in Mpc/h
delta_DL = (DL_max - DL_min)/N_DL
DL_bin_centers = np.arange(DL_min + delta_DL/2., DL_max, delta_DL)

survey_params = {"sigma_lnD":0.05, \
                 "ell_lst":ell_lst, \
                 "f_sky":0.5, \
                 "z_bin_centers":z_bin_centers, \
                 "delta_z":delta_z, \
                 "DL_bin_centers":DL_bin_centers, \
                 "delta_DL":delta_DL, \
                 "comoving_number_densities":np.array([n_bar_gal_z, n_bar_GW_z])}
####################



w0_fid = cosmo.w0
H0_fid = cosmo.H0
Om0_fid = cosmo.Om0
sigma_w0 = w0_fid*5e-2
sigma_H0 = H0_fid*1e-2
sigma_Om0 = Om0_fid*1e-2




### --- This block specifies the technical parameters neccessary for the run -- ###
# Location and name of the chain
chain_storage = "Data/chains/var_cosmo_reconstruction_ET_a.h5"
# Location and name of the mock data file
data_file = "Data/mock_data.npy"
# If you are worried about having window functions with too steep slopes set this to True
redshift_array_validation = False
# Redshift range
z_min, z_max = 1e-3, 3.
z_array_length = 1e3
z_lst = np.linspace(z_min, z_max, z_array_length)

# Sometimes you would like to use EXACTLY the same mock data from a different run.
# E.g. you want to have the same run with a different ell_max.
# In such a case you should set this to False
generate_new_data = False
# This is the observable mode.
# Use "combined" for using all the spectra
# Use "GWGC_only" for using only the cross spectra
# Use "GWGW_only" for using only the GW auto spectra
# Use "GCGC_only" for using only the GC auto spectra
observable_mode = "combined"
###################################################################################
