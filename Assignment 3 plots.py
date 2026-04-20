import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

Theta_D = 276
lambda_ = 0.33  # N(0)V


#############################################         Q1          #########################################
def gap_equation(Delta, T):

    def integrand(x):
        if Delta < 1e-8:
            return np.tanh(x / (2*T)) / x
        else:
            E = np.sqrt(x**2 + Delta**2)
            return np.tanh(E / (2*T)) / E

    integral, _ = quad(integrand, 1e-6, Theta_D)
    return lambda_ * integral - 1

# temperature range
T_vals = np.linspace(0.0, 16, 100)
Delta_vals = []

for T in T_vals:
    try:
        sol = root_scalar(gap_equation, args=(T,),
                          bracket=[-5, 50], method='brentq')
        Delta_vals.append(sol.root)
    except:
        Delta_vals.append(0)


print(min(Delta_vals))
plt.plot(T_vals, Delta_vals)
plt.xlabel("Temperature (K)")
plt.ylabel("Delta(T)")
plt.title("BCS Gap vs Temperature")
plt.show()




#############################################         Q2          #########################################
def lambda_inv_sq(T, Delta):
    def integrand(x):
        E = np.sqrt(x**2 + Delta**2)
        return 1 / np.cosh(E/(2*T))**2

    integral, _ = quad(integrand, 0, 200)
    return 1 - (1/(2*T)) * integral

lambda_vals = []

for T, Delta in zip(T_vals, Delta_vals):
    if Delta == 0:
        lambda_vals.append(0)
    else:
        lambda_vals.append(lambda_inv_sq(T, Delta))

plt.plot(T_vals, lambda_vals)
plt.xlabel("Temperature (K)")
plt.ylabel("1 / lambda^2(T)")
plt.title("Superfluid Density vs Temperature")
plt.show()



#############################################         Q3           #########################################

# --- Input from Q1 ---
# T_vals and Delta_vals 


# Interpolation function for Delta(T)
def Delta_T(T):
    return np.interp(T, T_vals, Delta_vals)

# BCS density of states (normalized)
def Ns(E, Delta):
    if abs(E) < Delta:
        return 0.0
    return abs(E) / np.sqrt(E**2 - Delta**2)

# derivative of Fermi function
def dfdE(E, V, T):
    return (1/(4*T)) * (1/np.cosh((E - V)/(2*T)))**2

# Conductance (normalized to G_N)
def conductance(V, T):
    Delta = Delta_T(T)
    
    if Delta == 0:
        return 1.0  # normal state
    
    def integrand(E):
        return Ns(E, Delta) * dfdE(E, V, T)
    
    # symmetric limits for stability
    integral, _ = quad(integrand, -10*max(Delta, T), 10*max(Delta, T), limit=200)
    
    return integral

# Voltage range (in same units as Delta, typically k_B T units)
V_vals = np.linspace(-100, 100, 300)

# Choose 5 temperatures approaching Tc
Tc = max(T_vals)
temps = Tc*np.linspace(0.1,0.99,10)

# Plot
plt.figure()

for T in temps:
    G_vals = [conductance(V, T) for V in V_vals]
    plt.plot(V_vals, G_vals, label=f"T = {T:.2f} K")

plt.xlabel("Voltage (V)")
plt.ylabel("G(V) / G_N")
plt.title("Conductance (Nb–Ag Junction)")
plt.legend()
plt.grid()
plt.show()



#############################################         Q4          #########################################


kB = 1.0  # set k_B = 1 (natural units)

Tc = 15  # Nb Tc (can be changed)
T_vals = np.linspace(0.01, Tc, 200)

Delta0 = 1.76 * kB * Tc
Delta_vals = Delta0 * np.tanh(1.74 * np.sqrt(Tc / T_vals - 1))

# Interpolation
Delta_func = interp1d(T_vals, Delta_vals, fill_value="extrapolate")

# Numerical derivative dDelta/dT
dDelta_dT_vals = np.gradient(Delta_vals, T_vals)
dDelta_dT_func = interp1d(T_vals, dDelta_dT_vals, fill_value="extrapolate")


# ---- Specific heat integrand ----
def integrand(xi, T):
    Delta = Delta_func(T)
    dDelta_dT = dDelta_dT_func(T)
    
    E = np.sqrt(xi**2 + Delta**2)
    beta = 1.0 / (kB * T)
    
    sech2 = 1.0 / np.cosh(beta * E / 2.0)**2
    
    return sech2 * (E**2 - T * Delta * dDelta_dT)


# ---- Compute C(T) ----
def C_of_T(T):
    xi_max = 10 * Delta0  # cutoff
    
    integral, _ = quad(integrand, 0, xi_max, args=(T,))
    
    return integral / (kB * T**2)


# ---- Evaluate over temperature ----
C_vals = np.array([C_of_T(T) for T in T_vals])


# ---- Plot ----
plt.figure()
plt.plot(T_vals, C_vals)
plt.xlabel("T")
plt.ylabel("C(T)")
plt.title("Specific Heat vs Temperature")
plt.grid()
plt.show()
