### --- imports --- ###
import sys
from os import path
sys.path.append(path.dirname(path.dirname( path.abspath(__file__))))

from initialize import *
from multiprocessing import Pool

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import emcee
### --- ####### --- ###



filename = chain_storage
n_cores = 27
max_n = 100000


C_ell_instance = C_ells(survey_params, z_integration_lst = z_lst, verbose = False)

mock_data = input_data.get("C_ell_mock")
inv_Cov_mat = input_data.get("inv_Cov_mat")



################# -------- #################



def GP(training_pos, training_vals, correlation_length, z_lst):

    kernel = RBF(correlation_length, (correlation_length, correlation_length))
    gp = GaussianProcessRegressor(kernel=kernel)

    gp.fit(np.array(training_pos)[:,np.newaxis], training_vals)

    return gp.predict(z_lst[:, np.newaxis])

def lnlike(params):
    DLR_val_2, DLR_val_3, DLR_val_4, DLR_val_5, corr_length_DLR, \
    BR_val_1, BR_val_2, BR_val_3, BR_val_4, corr_length_BR, w0, H0, Om0 = params # We vary these params
#, H0, Om0

    # Note that we always have a fixed node at z = 0 w/ value = 1
    training_pos_DLR = [min(z_lst), 0.5, 1.25, 2., 2.75]
    training_vals_DLR = [0., DLR_val_2, DLR_val_3, DLR_val_4, DLR_val_5]
    DLR_gp = GP(training_pos_DLR, training_vals_DLR, corr_length_DLR, z_lst) + 1.

    training_pos_BR = [min(z_lst), 1., 2., 3.]
    training_vals_BR = [BR_val_1, BR_val_2, BR_val_3, BR_val_4]
    BR_gp = GP(training_pos_BR, training_vals_BR, corr_length_BR, z_lst) + 1.



    if True in (DLR_gp<=0) or True in (BR_gp<=0) or np.all(np.diff(C_ell_instance.DL_lst*DLR_gp) >= 0.) == False:
        return -np.inf
    else:
        '''
        NOTE: Remember that the first C_ell entry should the DL-ratio while the second one is bias ratio.
        NOTE: Varying cosmology is now trivial. Simply pass the desired value of the parameter in the argument of C_ell (e.g. w0 = -0.9, wa = 0.1).
                Supported parameters are: 'H0' (default: 67.66)
                                          'Om0' (default: 0.3111)
                                          'Ob0' (default: 0.049)
                                          'sigma8': (default: 0.8102)
                                          'ns': (default: 0.9665)
                                          'w0': (default: -1.0)
                                          'wa': (default: 0.0)
        '''


        if observable_mode == "combined":
            C_ell_theory = C_ell_instance.C_ell(DLR_gp, BR_gp, \
                        w0 = w0, H0 = H0, Om0 = Om0, output_mode = "combined")["C_ell_combined"]
        if observable_mode == "GWGC_only":
            C_ell_theory = C_ell_instance.C_ell(DLR_gp, BR_gp, output_mode = "separate")["GWGC"]
        if observable_mode == "GWGW_only":
            C_ell_theory = C_ell_instance.C_ell(DLR_gp, BR_gp, output_mode = "separate")["GWGW"]
        if observable_mode == "GCGC_only":
            C_ell_theory = C_ell_instance.C_ell(DLR_gp, BR_gp, output_mode = "separate")["GCGC"]

        diff = (C_ell_theory - mock_data)

        tmp_1 = (inv_Cov_mat*diff.T).sum(axis = 1)
        tmp_2 = (diff.T*tmp_1).sum(axis = 0)

        loglikelihood = - 0.5*tmp_2.sum()
        return loglikelihood

def lnpost(params):
    DLR_val_2, DLR_val_3, DLR_val_4, DLR_val_5, corr_length_DLR, \
    BR_val_1, BR_val_2, BR_val_3, BR_val_4, corr_length_BR, w0, H0, Om0 = params # We vary these params
#, H0, Om0
    training_vals_DLR = np.array([0., DLR_val_2, DLR_val_3, DLR_val_4, DLR_val_5])
    training_vals_BR = np.array([BR_val_1, BR_val_2, BR_val_3, BR_val_4])

    prior_1 = np.all((training_vals_BR > -1.)&(training_vals_BR < 10.)) and (1. < corr_length_BR < 10.)
    prior_2 = np.all((training_vals_DLR > -1.)&(training_vals_DLR < 10.)) and (1. < corr_length_DLR < 10.)
    prior_3 = (w0 > -3.)&(w0 < -0.1)&(H0 > 20.)&(H0 < 150.)&(Om0 > 0.1)&(Om0 < 0.9)

#
    if prior_1 and prior_2 and prior_3:
        #print(lnlike(params))
        return lnlike(params) - (w0 - w0_fid)**2./2./(sigma_w0**2.) - \
                                (H0 - H0_fid)**2./2./(sigma_H0**2.) - \
                                (Om0 - Om0_fid)**2./2./(sigma_Om0**2.)
    return -np.inf


print("initialize chain")
p0 = np.zeros(13)
p0[4] = 2.
p0[9] = 2.
p0[10], p0[11], p0[12] = -1., 65., 0.3
ndim, nwalkers = len(p0), n_cores - 1
pos = [p0 + 1.*np.random.randn(ndim)*1e-3 for i in range(nwalkers)]
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)


print("start sampling")
pool = Pool(n_cores)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, backend = backend, pool = pool)

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf

# Now we'll sample for up to max_n steps
for sample in sampler.sample(pos, iterations=max_n, progress=True):
    # Only check convergence every 500 steps
    if sampler.iteration % 500:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.025)
    if converged:
        break

    print(np.abs(old_tau - tau) / tau)
    print(tau, np.abs(old_tau - tau) / tau < 0.025)

    old_tau = tau
