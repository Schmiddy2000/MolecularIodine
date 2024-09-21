import numpy as np
from scipy import odr
import matplotlib.pyplot as plt

# Pixel positions (independent variable)
pixel2 = np.array([ 50.4, 217, 841, 1027.3, 1040.3, 1102.8, 1106.4])

# Known wavelengths (dependent variable)
wavel2 = np.array([404.65650, 435.83350, 546.07500, 576.96100, 579.06700,588.995095,589.592424])

# Uncertainties in pixel positions
delta_pixel2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

# Uncertainties in wavelengths
delta_wavel2 = np.array([0.00010, 0.00010, 0.00010, 0.00010, 0.00010, 0.000003,0.000003])

# Define the third-order polynomial function
def third_order_poly(beta, p):
    I, C1, C2, C3 = beta
    return I + C1 * p + C2 * p**2 + C3 * p**3

# Create a Model object
model = odr.Model(third_order_poly)

# Create a RealData object
data = odr.RealData(pixel2, wavel2, sx=delta_pixel2, sy=delta_wavel2)

# Initial parameter estimates
beta0 = [400, 0.1, 0, 0]

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
print(param_std_dev)

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


def calibrate_pixel(p):
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


# Generate a range of pixel values for plotting
p_fit = np.linspace(min(pixel2), max(pixel2), 500)
w_fit = third_order_poly(parameters, p_fit)

# Plot data points with error bars
plt.errorbar(pixel2, wavel2, xerr=delta_pixel2, yerr=delta_wavel2, fmt='o', label='Data with uncertainties')

# Plot the fitted curve
plt.plot(p_fit, w_fit, 'r-', label='ODR Fit')

plt.xlabel('Pixel')
plt.ylabel('Wavelength (nm)')
plt.legend()
plt.title('ODR Fit of Wavelength vs. Pixel')
plt.show()
