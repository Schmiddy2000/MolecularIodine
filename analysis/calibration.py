# -*- coding: utf-8 -*-

import numpy as np
from scipy import odr
import matplotlib.pyplot as plt


def calibrate_pixel(p):
    parameters = [395.12645299492544, 0.18992098893696846, -1.0893407346103385e-05, -1.6973034224672095e-09]
    param_std_dev = [0.12977807032153638, 0.0012283171854946735, 2.2947774732871085e-06, 1.1992220641201165e-09]
    cov_matrix = np.array([[2.04470541e-02, -1.67582159e-04, 2.83597587e-07, -1.35114241e-10],
                           [-1.67582159e-04, 1.83167821e-06, -3.33363893e-09, 1.64974004e-12],
                           [2.83597587e-07, -3.33363893e-09, 6.39306733e-12, -3.29154754e-15],
                           [-1.35114241e-10, 1.64974004e-12, -3.29154754e-15, 1.74593207e-18]])

    # Compute the calibrated wavelength
    wavelength = parameters[0] + parameters[1] * p + parameters[2] * p ** 2 + parameters[3] * p ** 3

    # Partial derivatives with respect to parameters
    partials = np.array([
        1,
        p,
        p ** 2,
        p ** 3
    ])

    # Variance of the calibrated wavelength
    variance = np.dot(partials, np.dot(cov_matrix, partials))

    # Standard deviation (uncertainty)
    wavelength_uncertainty = np.sqrt(variance)

    return wavelength, wavelength_uncertainty


def run_code():
    # Pixel positions (independent variable)
    pixel2 = np.array([50.4, 217, 841, 1027.3, 1040.3, 1102.8, 1106.4])

    # Known wavelengths (dependent variable)
    wavel2 = np.array([404.65650, 435.83350, 546.07500, 576.96100, 579.06700, 588.995095, 589.592424])

    # Uncertainties in pixel positions
    delta_pixel2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    # Uncertainties in wavelengths
    delta_wavel2 = np.array([0.00010, 0.00010, 0.00010, 0.00010, 0.00010, 0.000003, 0.000003])

    # Define the third-order polynomial function
    def third_order_poly(beta, p):
        I, C1, C2, C3 = beta
        return I + C1 * p + C2 * p ** 2 + C3 * p ** 3

    # Create a Model object
    model = odr.Model(third_order_poly)

    # Create a RealData object
    data = odr.RealData(pixel2, wavel2, sx=delta_pixel2, sy=delta_wavel2)

    # Initial parameter estimates
    beta0 = [400, 0.1, 1e-6, 1e-9]

    # Create an ODR object
    odr_instance = odr.ODR(data, model, beta0=beta0)

    # Run the regression
    output = odr_instance.run()

    # Extract parameters and uncertainties
    parameters = output.beta
    param_std_dev = output.sd_beta
    cov_matrix = output.cov_beta  # Covariance matrix of the parameters
    #denominator_matrix = np.outer(param_std_dev, param_std_dev)

    #orrelation_matrix = cov_matrix / denominator_matrix
    print("Parameter Standard Deviations:")
    print(f'parameters = {list(parameters)}')
    print(f'param_std_dev = {list(param_std_dev)}')

    print("Covariance Matrix:")
    print(cov_matrix)

    # Display the results
    print("Fitted Parameters and Uncertainties:")
    print(f"I   = {parameters[0]:.6f} +- {param_std_dev[0]:.6f}")
    print(f"C1  = {parameters[1]:.6f} +- {param_std_dev[1]:.6f}")
    print(f"C2  = {parameters[2]:.6e} +- {param_std_dev[2]:.6e}")
    print(f"C3  = {parameters[3]:.6e} +- {param_std_dev[3]:.6e}")

    # Assess fit quality
    print(f"\nSum of squares of residuals: {output.sum_square}")
    print(f"Reduced chi-squared: {output.res_var}")
    fitted_wavel2 = third_order_poly(parameters, pixel2)

    ss_res = np.sum((wavel2 - fitted_wavel2) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((wavel2 - np.mean(wavel2)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)

    print(f"R: {r_squared:.6f}")

    # Generate a range of pixel values for plotting
    # p_fit = np.linspace(min(pixel2), max(pixel2), 500)
    p_fit = np.linspace(0, 2050, 500)
    w_fit = third_order_poly(parameters, p_fit)
    #wavelength, wavelength_uncertainty = calibrate_pixel()
    w_uncertainties = []
    for p in p_fit:
        _, unc = calibrate_pixel(p)
        w_uncertainties.append(unc)
    w_uncertainties = np.array(w_uncertainties)

    plt.figure(figsize=(12, 4))
    # Plot data points with error bars
    plt.errorbar(pixel2, wavel2, xerr=delta_pixel2, yerr=delta_wavel2, fmt='o', label='Data with uncertainties',
                 ecolor='k')

    # Plot the fitted curve
    plt.plot(p_fit, w_fit, 'r-', label='ODR Fit')
    plt.fill_between(p_fit, w_fit - w_uncertainties, w_fit + w_uncertainties, color='r', alpha=0.2, label='Uncertainty')

    plt.xlabel('Channel', fontsize=13)
    plt.ylabel('Wavelength in [nm]', fontsize=13)
    plt.legend()
    plt.title('ODR Fit of Wavelength vs. Channels', fontsize=16)
    plt.xlim(0, 2050)
    plt.tight_layout()
    plt.savefig('channel_to_wave_length_transformation.png', dpi=200)
    plt.show()

    return None


# run_code()
