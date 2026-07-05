import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# --- Parameters ---
Delta = 1.5e-3      # gap (eV)
temp = 0.5          # temperature (K)
kB = 8.617e-5       # Boltzmann constant (eV/K)

Z_list = [0, 0.5, 1, 2]

# Voltage and energy grids
v = np.linspace(-3*Delta, 3*Delta, 300)
e = np.linspace(-6*Delta, 6*Delta, 3000)

# --- Fermi derivative ---
def dfdE(e, v):
    x = (e - v) / (2 * kB * temp)
    return 1 / (4 * kB * temp * np.cosh(x)**2)

# --- BTK A and B ---
def AB(e, Z):
    e_abs = np.abs(e)
    
    A = np.zeros_like(e)
    B = np.zeros_like(e)
    
    # Inside gap
    mask1 = e_abs <= Delta
    e1 = e_abs[mask1]
    
    denom1 = e1**2 + (Delta**2 - e1**2) * (1 + 2*Z**2)**2
    A[mask1] = Delta**2 / denom1
    B[mask1] = 1 - A[mask1]
    
    # Outside gap
    mask2 = e_abs > Delta
    e2 = e_abs[mask2]
    
    root = np.sqrt(e2**2 - Delta**2)
    u2 = 0.5 * (1 + root / e2)
    v2 = 1 - u2
    
    denom2 = (u2 + Z**2 * (u2 - v2))**2
    
    A[mask2] = (u2 * v2) / denom2
    B[mask2] = ((u2 - v2)**2 * Z**2 * (1 + Z**2)) / denom2
    
    return A, B

# --- Plot ---
plt.figure(figsize=(8,6))

colors = ['blue', 'red', 'green', 'purple']

for i, Z in enumerate(Z_list):
    
    TN = 1 / (1 + Z**2)
    
    A, B = AB(e, Z)
    sigma = (1 + A - B) / TN
    
    G = []
    
    for vv in v:
        kernel = dfdE(e, vv)
        G.append(simpson(sigma * kernel, x=e))
    
    plt.plot(v/Delta, G, color=colors[i], label=f'Z = {Z}')

# --- Formatting ---
plt.axvline(1, linestyle='--', color='gray', alpha=0.5)
plt.axvline(-1, linestyle='--', color='gray', alpha=0.5)
plt.axhline(1, linestyle=':', color='gray', alpha=0.5)

plt.xlabel(r'$eV / \Delta$')
plt.ylabel(r'$G/G_N$')
plt.title('BTK Conductance (Normalized)')
plt.legend()
plt.grid(alpha=0.3)

plt.show()
