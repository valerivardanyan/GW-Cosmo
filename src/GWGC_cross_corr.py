'''
    GWGC_cross_corr: The source code for calculating tomographic cross-correlation
    angular power spectra between resolved gravitational wave souces (GW) and galaxy catalogues (GC).

    C_ells: This is the main class. Its methods return GW and GC window functions, and angular power spectra.
    C_ells.W_gal(): Returns a list of redshift arrays for desired GC window functions.
    C_ells.W_gw(): Returns a list of redshift arrays for desired GC window functions.
    C_ells.C_ell(): Returns the actual angular power spectra (both cross and auto).

    statistics: The methods of this class are producing all the necessary statistical utilities.
    statistics.covariance(): Returns the covariance matrix.
    statistics.mock_data(): Generate and returns the mock data based on fiducial model
                            and the corresponding covariance matrix.
'''

### --- general imports --- ###
import numpy as np
import scipy
from colossus.cosmology import cosmology
import astropy.cosmology
###############################

class C_ells:
    def __init__(self, survey_params, z_integration_lst, **kwargs):
        '''
            survey_params: a dictionary containing survey specifications.
                survey_params["sigma_lnD"]: error on ln(D_GW)
                survey_params["f_sky"]: fraction of the sky covered in the survey
                survey_params["z_bin_centers"]: central positions of GC bins
                survey_params["delta_z"]: (const) width of GC bins
                survey_params["DL_bin_centers"]: central positions of GW bins
                survey_params["delta_DL"]: (const) width of GW bins
                survey_params["comoving_number_densities"]: comoving number densities of GW and GC sources
                survey_params["ell_lst"]: the list of used multipoles
            cosmo_model: "CPL" w/ Planck2018
            z_integration_lst: a fixed z-array used throughout this file
        '''

        #### --- START: Basic initialization --- ###
        self.survey_params = survey_params
        self.z_lst = z_integration_lst
        self.dz = self.z_lst[1] - self.z_lst[0]

        self.sigma_lnD = self.survey_params["sigma_lnD"]
        self.f_sky = self.survey_params["f_sky"]

        self.N_z = len(self.survey_params["z_bin_centers"])
        self.N_DL = len(self.survey_params["DL_bin_centers"])

        self.gal_comoving_number_density, self.gw_comoving_number_density = survey_params["comoving_number_densities"]

        self.ell_lst = survey_params["ell_lst"]

        ### verbosity can be set through verbose = True
        self.kwargs = kwargs
        if "verbose" in self.kwargs and self.kwargs["verbose"] == True:
            print("Done: initialization of the survey specifications")
        #### --- END: Basic initialization --- ###

        #### --- START: Setting up cosmology --- ###
        cosmo_model = "CPL"
        if cosmo_model == "CPL":
            self.params = cosmology.cosmologies['planck18']
            self.params['de_model'], self.params['w0'], self.params['wa'] = 'w0wa', -1., 0.
            self.cosmo = cosmology.setCosmology('planck_w0wa', self.params)
        if cosmo_model == "fixed":
            self.cosmo = cosmology.setCosmology('planck18')


        ### Calculate the Hubble parameter, Hubble function, comoving distance and the EM luminosity distance.
        self.h = self.cosmo.Hz(0.)/100.
        self.H_z = self.cosmo.Hz(self.z_lst)/astropy.constants.c.to('km/s').value/self.h # in h/Mpc
        self.chi_lst = self.cosmo.comovingDistance(np.zeros(len(self.z_lst)), self.z_lst + 1e-5, transverse=True) # in Mpc/h
        self.DL_lst = self.cosmo.luminosityDistance(self.z_lst) # in Mpc/h
        self.DL_gw_lst = np.copy(self.DL_lst) # in Mpc/h

        if "verbose" in self.kwargs and self.kwargs["verbose"] == True:
            print("Current cosmology: ", self.params)
            print("Done: setting up the cosmology")
        #### --- END: Setting up cosmology --- ###

    def W_gal(self):

        '''
            This function returns N_galaxy_bins x len(z_lst) dimensional array of galaxy window functions,
            together with number of galaxies in the given bin (N_galaxy_bins dimensional array).
        '''


        z_bin_centers = self.survey_params["z_bin_centers"]
        delta_z = self.survey_params["delta_z"]

        tmp_1, tmp_2 = np.meshgrid(self.z_lst, z_bin_centers)

        top_hat = (np.tanh((tmp_1 - (tmp_2 - delta_z/2.))*1000.) + 1.)*\
                    (-np.tanh((tmp_1 - (tmp_2 + delta_z/2.))*1000.) + 1.)/4.


        tmp = self.gal_comoving_number_density*top_hat*self.chi_lst**2/self.H_z
        normalization = np.trapz(tmp, self.z_lst, axis = 1)

        return [np.transpose(np.transpose(tmp)/normalization), normalization]

    def W_gw(self):

        '''
            This function returns N_GW_bins x len(DL_gw_lst) dimensional array of galaxy window functions,
            together with number of galaxies in the given bin (N_GW_bins dimensional array).
        '''

        GW_DL_bin_centers = self.survey_params["DL_bin_centers"]
        delta_DL_GW = self.survey_params["delta_DL"]

        tmp_1, tmp_2 = np.meshgrid(self.DL_gw_lst, GW_DL_bin_centers)

        x_min = (np.log(tmp_2 - delta_DL_GW/2.) - np.log(tmp_1))/np.sqrt(2.)/self.sigma_lnD
        x_max = (np.log(tmp_2 + delta_DL_GW/2.) - np.log(tmp_1))/np.sqrt(2.)/self.sigma_lnD

        S_z_lst = (scipy.special.erfc(x_min) - scipy.special.erfc(x_max))/2.

        tmp = self.gw_comoving_number_density*S_z_lst*self.chi_lst**2/self.H_z/(1. + self.z_lst)
        normalization = np.trapz(tmp, self.z_lst, axis = 1)

        return [np.transpose(np.transpose(tmp)/normalization), normalization]


    def C_ell(self, DL_gw_ratio, bias, output_mode = "combined_tilde", **kwargs):

        '''
        DL_gw_ratio: redshift array of D_L^GW/D_L^EM
        bias: typically you want to pass a redshift array of b^GW/b^gal. Sometimes, however, you want to pass a single parameters
              in b_0*(1 + 1/D(z)). You can do either. When doing the latter make sure you pass an array of a single number, i.e. [b_0].
        output_mode: "separate" -- outputs a dictionary of power
                                   spectra with keys: "GWGC" (only GWGC cross spectra)
                                                      "GWGW" (only GWGW auto spectra)
                                                      "GCGC" (only GCGC auto spectra)
                                                      "GW_noise" (GW shot-noise in each GW bin)
                                                      "GC_noise" (GC shot-noise in each GC bin)

                     "combined" -- outputs a dictionary of all power
                           spectra combined together with a key: "C_ell_combined" (both cross and auto spectra)

                     "combined_tilde" -- outputs a dictionary of all power
                           spectra combined together, including the corresponding shot noise terms.
                           Key: "C_ell_combined_tilde" (both cross and auto spectra, w/ shot noise included)

        Cosmo parameters can be changed via kwargs (simply add, e.g. w0 = -0.95 in the C_ell call).
        Possible parameters are: 'H0' (default: 67.66)
                                 'Om0' (default: 0.3111)
                                 'Ob0' (default: 0.049)
                                 'sigma8': (default: 0.8102)
                                 'ns': (default: 0.9665)
                                 'w0': (default: -1.0)
                                 'wa': (default: 0.0)
        '''


        #### --- START: updating the cosmology --- ###
        if kwargs != {}:
            for param_tmp in kwargs:
                if param_tmp == "H0":
                    self.params[param_tmp] = kwargs[param_tmp]
                    self.cosmo.H0 = kwargs[param_tmp]
                elif param_tmp == "Om0":
                    self.params[param_tmp] = kwargs[param_tmp]
                    self.cosmo.Om0 = kwargs[param_tmp]
                elif param_tmp == "Ob0":
                    self.params[param_tmp] = kwargs[param_tmp]
                    self.cosmo.Ob0 = kwargs[param_tmp]
                elif param_tmp == "sigma8":
                    self.params[param_tmp] = kwargs[param_tmp]
                    self.cosmo.sigma8 = kwargs[param_tmp]
                elif param_tmp == "w0":
                    self.params[param_tmp] = kwargs[param_tmp]
                    self.cosmo.w0 = kwargs[param_tmp]
                elif param_tmp == "wa":
                    self.params[param_tmp] = kwargs[param_tmp]
                    self.cosmo.wa = kwargs[param_tmp]
                else:
                    print("ERROR: Unknown cosmological parameter: ", param_tmp)
                self.cosmo.checkForChangedCosmology()

            self.h = self.cosmo.Hz(0.)/100.
            self.H_z = self.cosmo.Hz(self.z_lst)/astropy.constants.c.to('km/s')/self.h # in h/Mpc
            self.chi_lst = self.cosmo.comovingDistance(np.zeros(len(self.z_lst)), self.z_lst + 1e-5, transverse=True) # in Mpc/h
            self.DL_lst = self.cosmo.luminosityDistance(self.z_lst) # in Mpc/h
        #### --- END: updating the cosmology --- ###

        if len(DL_gw_ratio) != len(self.z_lst):
            print("Error: The length of GW luminosity distance array does not match with redshift array!")
        else:
            self.DL_gw_lst = DL_gw_ratio*self.DL_lst

        b_gal = 1. + 1./self.cosmo.growthFactor(self.z_lst)

        if len(bias) != 1:
            if len(bias) != len(self.z_lst):
                print("Error: The length of GW bias array does not match with redshift array!")
            else:
                b_gw = bias*b_gal
        else:
            b_gw = bias[0]*(1. + 1./self.cosmo.growthFactor(self.z_lst))


        n_bar_gal_z, n_bar_GW_z = self.gal_comoving_number_density, self.gw_comoving_number_density

        ### ---> We call the GC and GW window functions
        W_gal_lst, gal_normalization_lst = self.W_gal()
        W_gw_lst, gw_normalization_lst = self.W_gw()

        N_GWGC = self.N_DL*self.N_z
        N_GWGW = self.N_DL
        N_GCGC = self.N_z

        ### ---> We compute the window kernels for all possible GW x GC cross correlations, as well as auto-corr
        GWGC_Window_Kernel_tmp = np.array([W_gw_lst[indx_DL]*W_gal_lst for indx_DL in range(self.N_DL)])
        GWGC_Window_Kernel = GWGC_Window_Kernel_tmp.reshape(self.N_DL*self.N_z, len(self.z_lst))
        # (Note that GWGC_Window_Kernel is organized as follows: it has self.N_DL blocks, the FIRST block of size
        # self.N_z corresponds to the FIRST GW bin multiplications w/ all gal-bins, the SECOND block of size
        # self.N_z corresponds to the SECOND GW bin multiplications w/ all gal-bins, etc...)

        GWGW_Window_Kernel_tmp = np.array([W_gw_lst[indx_DL]*W_gw_lst for indx_DL in range(self.N_DL)])
        GWGW_Window_Kernel = GWGW_Window_Kernel_tmp.reshape(self.N_DL*self.N_DL, len(self.z_lst))
        GCGC_Window_Kernel = W_gal_lst**2
        #GWGW_Window_Kernel = W_gw_lst**2


        ### ---> Preparing the power spectra entering the Limber approximated C_ell integrals
        chi_2D, ell_2D = np.meshgrid(self.chi_lst, self.ell_lst)
        k_ell_lst = (ell_2D + 0.5)/chi_2D
        # E.g. k_ell_lst[0] = (ell_2D[0] + 0.5)/chi_2D, k_ell_lst[1] = (ell_2D[1] + 0.5)/chi_2D, etc...

        P_m = self.cosmo.matterPowerSpectrum(k_ell_lst, 0.)*self.cosmo.growthFactor(self.z_lst)**2
        # Note that len(P_m) == len(self.ell_lst), while len(P_m[ith_ell]) == len(self.z_lst)

        ### ---> We are now ready to set up the GW x GC integrand and to perform the integral.
        tmp = b_gw*b_gal*GWGC_Window_Kernel*self.H_z/self.chi_lst**2
        # Note that len(tmp) == self.N_DL*self.N_z, while len(tmp[i]) == len(self.z_lst)

        C_ell_GWGC = np.matmul(P_m, tmp.T)*self.dz
        # len(C_ell_GWGC) == len(self.ell_lst), and len(C_ell_GWGC[ith_ell]) == self.N_DL*self.N_z

        ### ---> The same for GWGW and GCGC autocorrelations.
        tmp = b_gal*b_gal*GCGC_Window_Kernel*self.H_z/self.chi_lst**2
        C_ell_GCGC = np.matmul(P_m, tmp.T)*self.dz
        # len(C_ell_GCGC) == len(self.ell_lst), and len(C_ell_GCGC[ith_ell]) == self.N_z
        tmp_for_cov_mat = b_gw*b_gw*GWGW_Window_Kernel*self.H_z/self.chi_lst**2
        tmp = tmp_for_cov_mat[np.diag([True]*N_GWGW).flatten()]
        C_ell_GWGW_for_cov_mat = np.matmul(P_m, tmp_for_cov_mat.T)*self.dz
        C_ell_GWGW = np.matmul(P_m, tmp.T)*self.dz
        # C_ell_GWGW_for_cov_mat is used for the covariance matrix calculations only and is not considered as a signal
        # len(C_ell_GWGW) == len(self.ell_lst), and len(C_ell_GWGW[ith_ell]) == self.N_DL


        if output_mode == "separate":

            C_ell_dictionary = {"GWGC":C_ell_GWGC, "GWGW":C_ell_GWGW, "GCGC":C_ell_GCGC, \
                            "GW_noise":1./gw_normalization_lst, "GC_noise":1./gal_normalization_lst}

            return C_ell_dictionary

        elif output_mode == "combined":

            C_ell_combined = np.zeros((len(self.ell_lst), N_GWGC + N_GWGW + N_GCGC))

            C_ell_combined[:,0:N_GWGC] = C_ell_GWGC
            C_ell_combined[:,N_GWGC:N_GWGC + N_GWGW] = C_ell_GWGW
            C_ell_combined[:,N_GWGC + N_GWGW:N_GWGC + N_GWGW + N_GCGC] = C_ell_GCGC
            # Note that the first N_GWGC entries are the cross-spectra,
            # then GW auto spectra (N_GWGW entries), then GC auto spectra (N_GCGC entries)

            return {"C_ell_combined":C_ell_combined}

        elif output_mode == "combined_tilde":
            C_ell_tilde_combined = np.zeros((len(self.ell_lst), N_GWGC + N_GWGW*N_GWGW + N_GCGC))

            C_ell_tilde_combined[:,0:N_GWGC] = C_ell_GWGC

            GW_shot_noise = np.diag(1./gw_normalization_lst).flatten()
            C_ell_tilde_combined[:,N_GWGC:N_GWGC + N_GWGW*N_GWGW] = C_ell_GWGW_for_cov_mat + GW_shot_noise
            C_ell_tilde_combined[:,N_GWGC + N_GWGW*N_GWGW:N_GWGC + N_GWGW*N_GWGW + N_GCGC] = C_ell_GCGC + 1./gal_normalization_lst
            # Note that the first N_GWGC entries are the cross-spectra,
            # then GW auto spectra (including cross-bins: N_GWGW*N_GWGW entries), then GC auto spectra (N_GCGC entries)

            return {"C_ell_combined_tilde":C_ell_tilde_combined}

class statistics():

    def __init__(self, C_ells_instance):
        self.C_ells = C_ells_instance

    def covariance(self, DL_gw_ratio, bias):
        '''
        DL_gw_ratio: redshift array of D_L^GW/D_L^EM
        bias: typically you want to pass a redshift array of b^GW/b^gal. Sometimes, however, you want to pass a single parameters
              in b_0*(1 + 1/D(z)). You can do either. When doing the latter make sure you pass an array of a single number, i.e. [b_0].
        '''

        N_DL = self.C_ells.N_DL
        N_z = self.C_ells.N_z

        # We get the combined array of ALL C_ell's, including the shot noise when applicable.
        # Remember that the first N_GWGC entries are the cross-spectra,
        # then GW auto spectra (N_DL*N_DL entries), then GC auto spectra (N_z entries)
        C_ell_tilde = self.C_ells.C_ell(DL_gw_ratio, bias, output_mode = "combined_tilde")["C_ell_combined_tilde"]

        # We transpose it, so now len(C_ell_GWGC) == self.N_DL*self.N_z + self.N_DL*self.N_DL + self.N_z,
        # and len(C_ell_GWGC[ith_spectrum]) == len(self.ell_lst)
        C_ell_tilde = C_ell_tilde.T


        # Initialize the full covariance matrix
        # While we take the cross-GW-bin correlations into account, we don't use them as a signal.
        Cov_full = np.zeros((N_DL*N_z + N_DL + N_z, N_DL*N_z + N_DL + N_z, len(self.C_ells.ell_lst)))

        # Cov[C^ij(ell), C^mn(ell)] = (C_tilde^im C_tilde^jn + C_tilde^in C_tilde^jm)/(2 ell + 1)/f_sky
        # I_indx is a collective index for {ij}, and J_indx is a collective index for {mn}
        for I_indx in range(N_DL*N_z + N_DL + N_z):

            # Identify in which category I_indx belonds to
            if I_indx < N_DL*N_z:
                I_observable  = "GWGC"

                i_indx = int(I_indx/N_z)
                j_indx = I_indx - i_indx*N_z

            if N_DL*N_z <= I_indx < N_DL*N_z + N_DL:
                # i = gw, j = gw: Note that we don't have cross-bin correlations as a signal
                I_observable = "GWGW"

                i_indx = j_indx = I_indx - N_DL*N_z

            if N_DL*N_z + N_DL <= I_indx:
                # i = gal, j = gal: Note that we don't have cross-bin correlations as a signal
                I_observable = "GCGC"

                i_indx = j_indx = I_indx - N_DL*N_z - N_DL


            #print("I_observable = ", I_observable, "I_indx = ", I_indx, "i_indx = ", i_indx, "j_indx = ", j_indx)

            for J_indx in range(I_indx, N_DL*N_z + N_DL + N_z):
                # We only loop over upper diagonal (including the diagonal itself)

                if J_indx < N_DL*N_z:
                    J_observable  = "GWGC"

                    m_indx = int(J_indx/N_z)
                    n_indx = J_indx - m_indx*N_z
                if N_DL*N_z <= J_indx < N_DL*N_z + N_DL:
                    # m = gw, n = gw: Note that we don't have cross-bin correlations as a signal
                    J_observable = "GWGW"

                    m_indx = n_indx = J_indx - N_DL*N_z


                if N_DL*N_z + N_DL <= J_indx:
                    # m = gal, n = gal: Note that we don't have cross-bin correlations as a signal
                    J_observable = "GCGC"

                    m_indx = n_indx = J_indx - N_DL*N_z - N_DL
                #print("J_observable = ", J_observable, "J_indx = ", J_indx, "m_indx = ", m_indx, "n_indx = ", n_indx)


                # Finally, let's calculate the covariance elements for each combination
                if I_observable == "GWGC" and J_observable == "GWGC":
                    # j_indx and n_indx should be the same because these are galaxy bins and we assume C_ell_tilde^jn == 0 if j!=n
                    if (j_indx == n_indx):
                        cov_tmp_1 = C_ell_tilde[N_DL*N_z + i_indx*N_DL + m_indx]*C_ell_tilde[N_DL*N_z + N_DL*N_DL + j_indx]
                    else:
                        cov_tmp_1 = np.zeros(len(self.C_ells.ell_lst))

                    cov_tmp_2 = C_ell_tilde[i_indx*N_z + n_indx]*C_ell_tilde[m_indx*N_z + j_indx]

                if I_observable == "GWGC" and J_observable == "GWGW":
                    cov_tmp_1 = C_ell_tilde[N_DL*N_z + i_indx*N_DL + m_indx]*C_ell_tilde[n_indx*N_z + j_indx]
                    cov_tmp_2 = C_ell_tilde[N_DL*N_z + i_indx*N_DL + n_indx]*C_ell_tilde[m_indx*N_z + j_indx]

                if I_observable == "GWGC" and J_observable == "GCGC":
                    # m_indx == n_indx in this case by assumption
                    # j_indx and m_indx should be the same because these are galaxy bins and we assume C_ell_tilde^jm == 0 if j!=m
                    if (j_indx == m_indx):
                        cov_tmp_1 = C_ell_tilde[i_indx*N_z + m_indx]*C_ell_tilde[N_DL*N_z + N_DL*N_DL + n_indx]
                        cov_tmp_2 = C_ell_tilde[i_indx*N_z + n_indx]*C_ell_tilde[N_DL*N_z + N_DL*N_DL + m_indx]
                    else:
                        cov_tmp_1 = np.zeros(len(self.C_ells.ell_lst))
                        cov_tmp_2 = np.zeros(len(self.C_ells.ell_lst))

                if I_observable == "GWGW" and J_observable == "GWGW":
                    cov_tmp_1 = C_ell_tilde[N_DL*N_z + i_indx*N_DL + m_indx]*C_ell_tilde[N_DL*N_z + j_indx*N_DL + n_indx]
                    cov_tmp_2 = C_ell_tilde[N_DL*N_z + i_indx*N_DL + n_indx]*C_ell_tilde[N_DL*N_z + j_indx*N_DL + m_indx]

                if I_observable == "GWGW" and J_observable == "GCGC":
                    # m_indx == n_indx in this case by definition
                    cov_tmp_1 = C_ell_tilde[i_indx*N_z + m_indx]*C_ell_tilde[j_indx*N_z + n_indx]
                    cov_tmp_2 = C_ell_tilde[i_indx*N_z + n_indx]*C_ell_tilde[j_indx*N_z + m_indx]

                # Finally
                if I_observable == "GCGC" and J_observable == "GCGC":
                    # m_indx == n_indx and i_indx == j_indx in this case by definition
                    # i_indx and m_indx should be the same because these are galaxy bins and we assume C_ell_tilde^im == 0 if i!=m
                    if (i_indx == m_indx):
                        cov_tmp_1 = C_ell_tilde[N_DL*N_z + N_DL*N_DL + i_indx]**2
                        cov_tmp_2 = C_ell_tilde[N_DL*N_z + N_DL*N_DL + i_indx]**2
                    else:
                        cov_tmp_1 = np.zeros(len(self.C_ells.ell_lst))
                        cov_tmp_2 = np.zeros(len(self.C_ells.ell_lst))

                Cov_full[I_indx, J_indx] = cov_tmp_1 + cov_tmp_2
                Cov_full[J_indx, I_indx] = cov_tmp_1 + cov_tmp_2

        return Cov_full/(2.*self.C_ells.ell_lst + 1.)/self.C_ells.f_sky

    def mock_data(self, DL_gw_ratio, bias, observable_mode):
        '''
        DL_gw_ratio: redshift array of D_L^GW/D_L^EM
        bias: typically you want to pass a redshift array of b^GW/b^gal. Sometimes, however, you want to pass a single parameters
              in b_0*(1 + 1/D(z)). You can do either. When doing the latter make sure you pass an array of a single number, i.e. [b_0].

        observable_mode: "combined" (if you are using all the data)
                         "GWGC_only" (if you are only using the cross-spectra)
                         "GWGW_only" (if you are only using GW auto spectra)
                         "GCGC_only" (if you are only using GC auto spectra)
        '''

        N_DL = self.C_ells.N_DL
        N_z = self.C_ells.N_z

        # Irrespective of which observable mode you are interested in,
        # it is the best to get the full covariance matrix and take the necessary slices
        Cov_full = self.covariance(DL_gw_ratio, bias)


        if observable_mode == "combined":
            # Covariance and its inverse
            Cov_mat = np.copy(Cov_full)
            inv_Cov_mat = np.copy(Cov_mat)

            C_ell_fiducial = self.C_ells.C_ell(DL_gw_ratio, bias, output_mode = "combined")["C_ell_combined"]

        if observable_mode == "GWGC_only":
            # Covariance and its inverse
            Cov_mat = Cov_full[0:N_DL*N_z, 0:N_DL*N_z]
            inv_Cov_mat = np.copy(Cov_mat)

            C_ell_fiducial = self.C_ells.C_ell(DL_gw_ratio, bias, output_mode = "separate")["GWGC"]

        if observable_mode == "GWGW_only":
            # Covariance and its inverse
            Cov_mat = Cov_full[N_DL*N_z:N_DL*N_z + N_DL, N_DL*N_z:N_DL*N_z + N_DL]
            inv_Cov_mat = np.copy(Cov_mat)

            C_ell_fiducial = self.C_ells.C_ell(DL_gw_ratio, bias, output_mode = "separate")["GWGW"]

        if observable_mode == "GCGC_only":
            # Covariance and its inverse
            Cov_mat = Cov_full[N_DL*N_z + N_DL:N_DL*N_z + N_DL + N_z, N_DL*N_z + N_DL:N_DL*N_z + N_DL + N_z]
            inv_Cov_mat = np.copy(Cov_mat)

            C_ell_fiducial = self.C_ells.C_ell(DL_gw_ratio, bias, output_mode = "separate")["GCGC"]

        # Initialize an array for storing the generated mock data
        C_ell_mock = np.copy(C_ell_fiducial)

        for ell_indx in np.arange(len(self.C_ells.ell_lst)):
            Cov_tmp = Cov_mat[:,:,ell_indx]

            # We invert the cov matrix
            rhs = np.diag([1.]*len(np.diagonal(Cov_tmp)))
            inv_Cov_mat[:,:,ell_indx] = np.linalg.solve(Cov_tmp, rhs)

            C_ell_tmp = C_ell_fiducial[ell_indx]

            C_ell_mock[ell_indx] = np.random.multivariate_normal(C_ell_tmp, Cov_tmp)

        output_dict = {"survey_params":self.C_ells.survey_params, \
                       "C_ell_mock":C_ell_mock, \
                       "Cov_mat":Cov_mat, \
                       "inv_Cov_mat":inv_Cov_mat}

        return output_dict
