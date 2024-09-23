import numpy as np
import matplotlib.pyplot as plt

# Given constants
mu = 1.05e-25          # Reduced mass in kg
c = 2.99792458e10      # Speed of light in cm/s
h = 6.62607015e-34     # Planck's constant in Js
x_e = 2.666e-10        # Equilibrium distance in meters

# Your measurements
omega_0 = 122.1          # Measured value of omega_0 in cm^-1
delta_omega_0 = 1.8      # Uncertainty of omega_0
x0 = 45.33 / 8065             # Measured value of x0 (dimensionless)
delta_x0 = 1.01 / 8065      # Uncertainty of x0
cov_omega0_x0 = 0.001  # Covariance between omega_0 and x0

# Calculation of De
De = omega_0 / (4 * x0) * h * c  # in Joules

dDe_domega_0 = (1 / (4 * x0)) * h * c
dDe_dx_0 = (-omega_0 / (4 * x0**2)) * h * c
delta_De = np.sqrt((dDe_domega_0 * delta_omega_0)**2 + (dDe_dx_0 * delta_x0)**2 +
                   2 * dDe_domega_0 * dDe_dx_0 * cov_omega0_x0)

# Calculation of k
omega_0_SI = 2 * np.pi * c * omega_0
delta_omega_0_SI = 2 * np.pi * c * delta_omega_0

k = mu * omega_0_SI**2
delta_k = 2 * mu * omega_0_SI * delta_omega_0_SI

print(f'De = {De} +- {delta_De} Joules')
print(f'k = {k} +- {delta_k} N/m')

# Calculation of a
a = np.sqrt(k / (2 * De))  # in 1/m
da_dk = 1 / (4 * a * De)
da_dDe = -k / (4 * a * De**2)

delta_a = np.sqrt((da_dk * delta_k)**2 + (da_dDe * delta_De)**2)

print(f'a = {a} +- {delta_a} 1/m')

# Range of x-values around x_e
x = np.linspace(x_e - 1e-10, x_e + 5e-10, 500)  # in meters

# Calculation of the Morse potential
V = De * (1 - np.exp(-a * (x - x_e)))**2  # in Joules

V_cm1 = V / (h * c)
De_cm1 = De / (h * c)

# Convert x for plotting
x_angstrom = x * 1e10
x_e_angstrom = x_e * 1e10

# Plotting the Morse potential
plt.figure(figsize=(8, 6))
plt.plot(x_angstrom, V_cm1, label='Morse potential')

# Adding De (horizontal line)
plt.axhline(y=De_cm1, color='r', linestyle='--', label='$D_e$')

# Adding x_e (vertical line)
plt.axvline(x=x_e_angstrom, color='g', linestyle='--', label='$x_e$')


plt.xlabel(r'Distance $x$ in [$\AA$]')
plt.ylabel('Potential energy $V(x)$ in [cm$^{-1}$]')
plt.title('Morse potential for the I$_2$ molecule')

plt.grid(True)
plt.legend()
plt.show()
