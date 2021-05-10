import os
import sys

file_dir = os.path.split(os.getcwd())[0]
if file_dir not in sys.path:
    sys.path.append(file_dir)

from src.GWGC_cross_corr import *
from specifications import *

# Find the optimal redshift array for internal integrations,
# or use the default one with z_array_length = 1e3 (which is a safe option for normal runs)
if redshift_array_validation:

    z_array_length = 1e4
    z_lst = np.linspace(z_min, z_max, z_array_length)

    DL_ratio_fiducial, bias_fiducial = fiducial_ratios(z_lst)
    C_ell_instance = C_ells(survey_params, z_integration_lst = z_lst, verbose = True)

    C_ell_old = C_ell_instance.C_ell(DL_ratio_fiducial, \
                    bias_fiducial, output_mode = "combined")["C_ell_combined"]
    tol = 1e-3
    change = 0.

    while change < tol:
            z_array_length = int(z_array_length/2)
            print("Current z-array length: ", z_array_length)
            z_lst = np.linspace(z_min, z_max, z_array_length)

            DL_ratio_fiducial, bias_fiducial = fiducial_ratios(z_lst)
            C_ell_instance = C_ells(survey_params, z_integration_lst = z_lst, verbose = True)
            C_ell_new = C_ell_instance.C_ell(DL_ratio_fiducial, \
                        bias_fiducial, output_mode = "combined")["C_ell_combined"]

            diff = np.sum(C_ell_old - C_ell_new)
            sum_new = np.sum(C_ell_new)
            change = np.abs(diff/sum_new)
            print("The calculated relative change is: ", change)
            C_ell_old = C_ell_new

    z_array_length = int(z_array_length*2)
    z_lst = np.linspace(z_min, z_max, z_array_length)
    print("The used z-array length: ", z_array_length)

    DL_ratio_fiducial, bias_fiducial = fiducial_ratios(z_lst)
    C_ell_instance = C_ells(survey_params, z_integration_lst = z_lst, verbose = True)
else:
    # Use the defaul from the specifications.py: z_array_length = 1e3

    DL_ratio_fiducial, bias_fiducial = fiducial_ratios(z_lst)
    C_ell_instance = C_ells(survey_params, z_integration_lst = z_lst, verbose = True)

if generate_new_data:
    ### --- Make a statistics instance --- ###
    stat = statistics(C_ell_instance)

    ### --- This writes the covariance matrix, its inverse, and a generated mock data into files --- ###
    data_dict = stat.mock_data(DL_ratio_fiducial, bias_fiducial, observable_mode = observable_mode)
    np.save(data_file, data_dict)
    input_data = data_dict

    # The remaining lines output the total chi^2 for the fiducial model
    mock_data, Cov_mat, inv_Cov_mat = data_dict["C_ell_mock"], data_dict["Cov_mat"], data_dict["inv_Cov_mat"]

    if observable_mode == "combined":
        C_ell_theory = C_ell_instance.C_ell(DL_ratio_fiducial, bias_fiducial, output_mode = "combined")["C_ell_combined"]
    if observable_mode == "GWGC_only":
        C_ell_theory = C_ell_instance.C_ell(DL_ratio_fiducial, bias_fiducial, output_mode = "separate")["GWGC"]
    if observable_mode == "GWGW_only":
        C_ell_theory = C_ell_instance.C_ell(DL_ratio_fiducial, bias_fiducial, output_mode = "separate")["GWGW"]
    if observable_mode == "GCGC_only":
        C_ell_theory = C_ell_instance.C_ell(DL_ratio_fiducial, bias_fiducial, output_mode = "separate")["GCGC"]

    diff = C_ell_theory - mock_data

    tmp_1 = (inv_Cov_mat*diff.T).sum(axis = 1)
    tmp_2 = (diff.T*tmp_1).sum(axis = 0)

    print("chi^2_fid = ", tmp_2.sum())
else:
    # You decided to use a pre-existing data file,
    # but we will check to be sure it was generated
    # with parameters compatible with the current ones.
    try:
        input_data = np.load(data_file, allow_pickle = True).item()
    except FileNotFoundError:
        print("ERROR: Data file does not exist. You should proceed with enabling the generate_new_data option!")
    specs_from_file = input_data.get("survey_params")
    specs_current = survey_params

    for key in specs_current:

        if np.all(specs_current[key] == specs_from_file[key]) != True:
            print("WARNING: The specs you specified in specifications.py do not match those of the data you are using. Proceed with caution!")
