import numpy as np

# ==========================================
# PHYSICAL CONSTANTS & PARAMETERS
# ==========================================
G = 9.81         # Gravity (m/s^2)
RHO = 1.225      # Air density at sea level (kg/m^3)

# Tennis Ball Parameters
MASS = 0.0577    # kg
RADIUS = 0.033   # m
AREA = np.pi * (RADIUS**2)
CD = 0.55        # Drag Coefficient
CL = 0.22        # Lift Coefficient (Depends on spin)

# Wind Vector [wx, wy, wz]
WIND = np.array([0, 0, 0.0])

# Spin Vector [wx, wy, wz] in rad/s
OMEGA = np.array([0.0, 2000, 100])  # Example: topspin around Y-axis and some sidespin around Z-axis
