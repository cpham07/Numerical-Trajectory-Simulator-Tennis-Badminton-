import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from constants import G, RHO

# ==========================================
# PHYSICS ENGINE
# ==========================================
def get_derivatives(state, t, wind, omega):
    pos = state[0:3]
    vel = state[3:6]
    v_rel = vel - wind
    v_mag = np.linalg.norm(v_rel)

    if v_mag == 0:
        return np.array([vel[0], vel[1], vel[2], 0.0, 0.0, -G])

    area = np.pi * (radius**2)
    F_drag = -0.5 * RHO * area * cd * v_mag * v_rel
    K = 0.5 * RHO * area * cl * (radius / v_mag)
    F_magnus = K * np.cross(omega, v_rel)
    accel = (F_drag + F_magnus) / mass
    accel[2] -= G

    return np.concatenate((vel, accel))

def rk4_step(state, t, dt, wind, omega):
    k1 = get_derivatives(state, t, wind, omega)
    k2 = get_derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, wind, omega)
    k3 = get_derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, wind, omega)
    k4 = get_derivatives(state + dt * k3, t + dt, wind, omega)
    next_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return next_state

def eulers(state, t, dt, wind, omega):
    next_state = state + get_derivatives(state, t, wind, omega) * dt
    return next_state

def improved_eulers(state, t, dt, wind, omega):
    next_state = state + (dt/2) * (get_derivatives(state, t, wind, omega) +
                                   get_derivatives(eulers(state, t, dt, wind, omega), t + dt, wind, omega))
    return next_state

def simulate_all_methods(speed, theta, phi, wind, omega, dt=0.05):
    """Run all three integration methods with given parameters"""
    vx = speed * np.cos(np.radians(theta)) * np.cos(np.radians(phi))
    vy = speed * np.cos(np.radians(theta)) * np.sin(np.radians(phi))
    vz = speed * np.sin(np.radians(theta))
    initial_state = np.array([0.0, 0.0, 3, vx, vy, vz])

    # Initialize trajectories
    traj_rk4 = [initial_state.copy()]
    traj_euler = [initial_state.copy()]
    traj_imp_euler = [initial_state.copy()]

    state_rk4 = initial_state.copy()
    state_euler = initial_state.copy()
    state_imp_euler = initial_state.copy()

    t = 0.0

    # Run all three methods simultaneously
    while (state_rk4[2] > 0 and state_euler[2] > 0 and state_imp_euler[2] > 0 and
           len(traj_rk4) < 1000):  # Safety limit
        state_rk4 = rk4_step(state_rk4, t, dt, wind, omega)
        state_euler = eulers(state_euler, t, dt, wind, omega)
        state_imp_euler = improved_eulers(state_imp_euler, t, dt, wind, omega)

        traj_rk4.append(state_rk4.copy())
        traj_euler.append(state_euler.copy())
        traj_imp_euler.append(state_imp_euler.copy())
        t += dt

    return np.array(traj_rk4), np.array(traj_euler), np.array(traj_imp_euler)

def show_error_analysis():
    speed = speed_slider.val
    theta = theta_slider.val
    phi = phi_slider.val
    wind = np.array([wind_x_slider.val, wind_y_slider.val, wind_z_slider.val])
    omega = np.array([spin_x_slider.val, spin_y_slider.val, spin_z_slider.val])
    dt = dt_slider.val

    traj_rk4, traj_euler, traj_imp_euler = simulate_all_methods(
        speed, theta, phi, wind, omega, dt)

    # Trim to shortest trajectory length
    min_len = min(len(traj_rk4), len(traj_euler), len(traj_imp_euler))
    traj_rk4 = traj_rk4[:min_len]
    traj_euler = traj_euler[:min_len]
    traj_imp_euler = traj_imp_euler[:min_len]

    time_steps = np.arange(min_len) * dt

    # Euclidean distance from RK4 at each timestep (position only)
    err_euler = np.linalg.norm(traj_euler[:, 0:3] - traj_rk4[:, 0:3], axis=1)
    err_imp_euler = np.linalg.norm(traj_imp_euler[:, 0:3] - traj_rk4[:, 0:3], axis=1)

    # Open new window
    err_fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    err_fig.suptitle(f'Error vs RK4 Over Time (dt={dt:.3f}s, {current_mode})',
                     fontsize=13, fontweight='bold')

    # Top plot — position error over time
    axes[0].plot(time_steps, err_euler, 'r-', label='Euler Error', linewidth=2)
    axes[0].plot(time_steps, err_imp_euler, 'g-', label='Improved Euler Error', linewidth=2)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Position Error vs RK4 (m)")
    axes[0].set_title("Cumulative Position Divergence from RK4")
    axes[0].legend()
    axes[0].grid()

    # Bottom plot — X, Y, Z error separately for Euler
    err_x = np.abs(traj_euler[:, 0] - traj_rk4[:, 0])
    err_y = np.abs(traj_euler[:, 1] - traj_rk4[:, 1])
    err_z = np.abs(traj_euler[:, 2] - traj_rk4[:, 2])

    axes[1].plot(time_steps, err_x, 'r-', label='Euler X error', linewidth=2)
    axes[1].plot(time_steps, err_y, 'b-', label='Euler Y error', linewidth=2)
    axes[1].plot(time_steps, err_z, 'orange', label='Euler Z error', linewidth=2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Absolute Error (m)")
    axes[1].set_title("Euler Error Breakdown by Axis")
    axes[1].legend()
    axes[1].grid()

    plt.tight_layout()
    plt.show()

def error_per_meter(traj_rk4, traj_other):
    # Distance traveled along RK4 path at each step
    deltas = np.diff(traj_rk4[:, 0:3], axis=0)
    dist_traveled = np.concatenate(([0], np.cumsum(np.linalg.norm(deltas, axis=1))))
    
    # Positional error at each step
    min_len = min(len(traj_rk4), len(traj_other))
    error = np.linalg.norm(
        traj_other[:min_len, 0:3] - traj_rk4[:min_len, 0:3], 
        axis=1
    )
    
    return dist_traveled[:min_len], error

def get_error_rate(speed, theta, phi, wind, omega, dt, method='euler'):
    traj_rk4, traj_euler, traj_imp_euler = simulate_all_methods(speed, theta, phi, wind, omega, dt)
    if method == 'euler':
        dist, err = error_per_meter(traj_rk4, traj_euler)
    elif method == 'improved_euler':
        dist, err = error_per_meter(traj_rk4, traj_imp_euler)
    else:
        return 0
    if len(dist) > 1:
        coeffs = np.polyfit(dist[1:], err[1:], 1)
        return coeffs[0]  # m error per m traveled
    return 0

def show_accuracy_analysis():
    speed = speed_slider.val
    theta = theta_slider.val
    phi = phi_slider.val
    wind = np.array([wind_x_slider.val, wind_y_slider.val, wind_z_slider.val])
    omega = np.array([spin_x_slider.val, spin_y_slider.val, spin_z_slider.val])
    dt_current = dt_slider.val

    dt_values = [0.1, 0.05, 0.025, 0.01, 0.005]
    euler_rates = []
    imp_euler_rates = []
    euler_landing_errors = []
    imp_euler_landing_errors = []

    for dt in dt_values:
        traj_rk4, traj_euler, traj_imp_euler = simulate_all_methods(speed, theta, phi, wind, omega, dt)
        print(f"dt={dt:.3f}: RK4 steps={len(traj_rk4)}, Euler steps={len(traj_euler)}, Imp Euler steps={len(traj_imp_euler)}")
        
        # Landing position error — just the final point
        euler_landing_err = np.linalg.norm(traj_euler[-1, 0:3] - traj_rk4[-1, 0:3])
        imp_euler_landing_err = np.linalg.norm(traj_imp_euler[-1, 0:3] - traj_rk4[-1, 0:3])
        
        euler_landing_errors.append(euler_landing_err)
        imp_euler_landing_errors.append(imp_euler_landing_err)
        
        euler_rates.append(get_error_rate(speed, theta, phi, wind, omega, dt, 'euler'))
        imp_euler_rates.append(get_error_rate(speed, theta, phi, wind, omega, dt, 'improved_euler'))

    # Fit slopes for empirical order
    log_dt = np.log(dt_values)
    log_euler = np.log(euler_rates)
    log_imp = np.log(imp_euler_rates)
    slope_euler = np.polyfit(log_dt, log_euler, 1)[0]
    slope_imp = np.polyfit(log_dt, log_imp, 1)[0]

    # For landing errors
    euler_land_below = [dt for dt, err in zip(dt_values, euler_landing_errors) if err < 0.0038]
    imp_land_below = [dt for dt, err in zip(dt_values, imp_euler_landing_errors) if err < 0.0038]
    max_dt_euler_land = max(euler_land_below) if euler_land_below else None
    max_dt_imp_land = max(imp_land_below) if imp_land_below else None

    # Console summary
    print("=== Accuracy Analysis Summary ===")
    print(f"Euler error rates (m/m): {euler_rates}")
    print(f"Improved Euler error rates (m/m): {imp_euler_rates}")
    print(f"Empirical order - Euler: O(dt^{slope_euler:.2f}), Improved Euler: O(dt^{slope_imp:.2f})")
    print(f"Maximum dt for Euler to meet 3.8mm threshold (landing): {max_dt_euler_land}")
    print(f"Maximum dt for Improved Euler to meet 3.8mm threshold (landing): {max_dt_imp_land}")
    if max_dt_euler_land and max_dt_imp_land:
        print("Both methods can meet officiating tolerances with sufficiently small dt.")
    elif max_dt_imp_land:
        print("Only Improved Euler meets officiating tolerances.")
    else:
        print("Neither method meets officiating tolerances within tested dt range.")

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f'Accuracy Analysis ({current_mode}) - Speed: {speed:.1f}m/s, Theta: {theta:.1f}°, Phi: {phi:.1f}°', fontsize=14)

    # Panel 1: Error vs Distance for current dt
    traj_rk4, traj_euler, traj_imp_euler = simulate_all_methods(speed, theta, phi, wind, omega, dt_current)
    dist_e, err_e = error_per_meter(traj_rk4, traj_euler)
    dist_ie, err_ie = error_per_meter(traj_rk4, traj_imp_euler)
    axes[0].plot(dist_e, err_e, 'r-', label='Euler', linewidth=2)
    axes[0].plot(dist_ie, err_ie, 'g-', label='Improved Euler', linewidth=2)
    axes[0].set_xlabel('Distance Traveled (m)')
    axes[0].set_ylabel('Positional Error (m)')
    axes[0].set_title(f'Error vs Distance (dt={dt_current:.3f}s)')
    axes[0].legend()
    axes[0].grid()

    # Panel 2: Log-log convergence
    axes[1].loglog(dt_values, euler_rates, 'r-o', label=f'Euler (slope={slope_euler:.2f})')
    axes[1].loglog(dt_values, imp_euler_rates, 'g-s', label=f'Improved Euler (slope={slope_imp:.2f})')
    axes[1].set_xlabel('dt (s)')
    axes[1].set_ylabel('Error Rate (m/m)')
    axes[1].set_title('Convergence Analysis')
    axes[1].legend()
    axes[1].grid()

    # Panel 3: Summary table
    axes[2].axis('off')
    table_data = [
        ['Method', 'Empirical Order', 'Max dt (landing)'],
        ['Euler', f'O(dt^{slope_euler:.2f})', str(max_dt_euler_land) if max_dt_euler_land else 'N/A'],
        ['Improved Euler', f'O(dt^{slope_imp:.2f})', str(max_dt_imp_land) if max_dt_imp_land else 'N/A']
    ]
    table = axes[2].table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    axes[2].set_title('Summary')

    # Panel 4: Landing position error vs dt
    axes[3].loglog(dt_values, euler_landing_errors, 'r-o', label='Euler', linewidth=2)
    axes[3].loglog(dt_values, imp_euler_landing_errors, 'g-s', label='Improved Euler', linewidth=2)
    axes[3].axhline(y=0.0038, color='k', linestyle='--', label='Hawk-Eye Threshold (3.8mm)')
    axes[3].set_xlabel('dt (s)')
    axes[3].set_ylabel('Landing Position Error vs RK4 (m)')
    axes[3].set_title('Landing Error vs dt')
    axes[3].legend()
    axes[3].grid()

    # Annotate max dt below threshold for each method
    if max_dt_euler_land:
        axes[3].annotate(f'Euler max dt: {max_dt_euler_land}s',
                         xy=(max_dt_euler_land, 0.0038),
                         xytext=(max_dt_euler_land * 1.2, 0.0038 * 2),
                         color='red', fontsize=9)
    if max_dt_imp_land:
        axes[3].annotate(f'Imp. Euler max dt: {max_dt_imp_land}s',
                         xy=(max_dt_imp_land, 0.0038),
                         xytext=(max_dt_imp_land * 0.5, 0.0038 * 0.4),
                         color='green', fontsize=9)

    plt.tight_layout()
    plt.show()

# ==========================================
# INTERACTIVE VISUALIZATION - COMPARE METHODS
# ==========================================
# Initial parameters
initial_speed = 64
initial_theta = -8.66
initial_phi = 0
initial_wind = np.array([0.0, 0.0, 0.0])
initial_omega = np.array([0.0, 300.0, 0.0])
initial_dt = 0.05

# Create figure and subplots
fig = plt.figure(figsize=(16, 8))

# Adjust subplot to make room for sliders
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9)

# 3D plot
ax1 = fig.add_subplot(131, projection='3d')

# Side view (XZ plane)
ax2 = fig.add_subplot(132)

# Top view (XY plane)
ax3 = fig.add_subplot(133)

# Mode-specific physical constants and court dimensions
current_mode = 'Tennis'
mode_constants = {
    'Tennis': {
        'mass': 0.0577,
        'radius': 0.033,
        'cd': 0.55,
        'cl': 0.22,
        'court_length': 23.77,
        'court_width': 8.23,
        'net_height': 0.91
    },
    'Badminton': {
        'mass': 0.0052,
        'radius': 0.032,
        'cd': 0.6,
        'cl': 0.0,
        'court_length': 13.4,
        'court_width': 6.1,
        'net_height': 1.55
    }
}

court_artists = []
court_artists2 = []
court_artists3 = []


def draw_court():
    for artist in court_artists:
        try:
            artist.remove()
        except Exception:
            pass
    court_artists.clear()

    for artist in court_artists2:
        try:
            artist.remove()
        except Exception:
            pass
    court_artists2.clear()

    for artist in court_artists3:
        try:
            artist.remove()
        except Exception:
            pass
    court_artists3.clear()

    left = -court_width / 2
    right = court_width / 2
    net_x = court_length / 2

    # Court boundaries for ax1 (3D)
    court_artists.extend(ax1.plot([0, court_length], [right, right], [0, 0], color='green', linestyle='--'))
    court_artists.extend(ax1.plot([0, court_length], [left, left], [0, 0], color='green', linestyle='--'))
    court_artists.extend(ax1.plot([0, 0], [left, right], [0, 0], color='green', linestyle='--'))
    court_artists.extend(ax1.plot([court_length, court_length], [left, right], [0, 0], color='green', linestyle='--'))

    # Net for ax1
    court_artists.extend(ax1.plot([net_x, net_x], [left, left], [0, court_net_height], color='red', linestyle='-'))
    court_artists.extend(ax1.plot([net_x, net_x], [right, right], [0, court_net_height], color='red', linestyle='-'))
    court_artists.extend(ax1.plot([net_x, net_x], [left, right], [court_net_height, court_net_height], color='red', linestyle='-'))

    # For ax2 (side view XZ): Net
    court_artists2.extend(ax2.plot([net_x, net_x], [0, court_net_height], color='red', linestyle='-', linewidth=2))

    # For ax3 (top view XY): Court boundaries and net
    court_artists3.extend(ax3.plot([0, court_length], [right, right], color='green', linestyle='--', linewidth=1))
    court_artists3.extend(ax3.plot([0, court_length], [left, left], color='green', linestyle='--', linewidth=1))
    court_artists3.extend(ax3.plot([court_length, court_length], [left, right], color='green', linestyle='--', linewidth=1))
    court_artists3.extend(ax3.plot([0, 0], [left, right], color='green', linestyle='--', linewidth=1))
    court_artists3.extend(ax3.plot([net_x, net_x], [left, right], color='red', linestyle='-', linewidth=2))

    if current_mode == 'Tennis':
        service_line = 6.40
        alley_offset = 1.37
        court_artists.extend(ax1.plot([service_line, service_line], [left, right], [0, 0], color='green', linestyle='--'))
        court_artists.extend(ax1.plot([court_length - service_line, court_length - service_line], [left, right], [0, 0], color='green', linestyle='--'))
        court_artists.extend(ax1.plot([service_line, court_length - service_line], [0, 0], [0, 0], color='green', linestyle='--'))
        court_artists.extend(ax1.plot([0, court_length], [right - alley_offset, right - alley_offset], [0, 0], color='green', linestyle='--'))
        court_artists.extend(ax1.plot([0, court_length], [left + alley_offset, left + alley_offset], [0, 0], color='green', linestyle='--'))

        # For ax3, add service lines
        court_artists3.extend(ax3.plot([service_line, service_line], [left, right], color='green', linestyle='--', linewidth=1))
        court_artists3.extend(ax3.plot([court_length - service_line, court_length - service_line], [left, right], color='green', linestyle='--', linewidth=1))
        court_artists3.extend(ax3.plot([service_line, court_length - service_line], [0, 0], color='green', linestyle='--', linewidth=1))


def set_mode(mode):
    global current_mode, mass, radius, cd, cl, court_length, court_width, court_net_height
    current_mode = mode
    params = mode_constants[mode]
    mass = params['mass']
    radius = params['radius']
    cd = params['cd']
    cl = params['cl']
    court_length = params['court_length']
    court_width = params['court_width']
    court_net_height = params['net_height']

    ax1.set_xlim(-2, court_length + 2)
    ax1.set_ylim(-court_width / 2 - 1, court_width / 2 + 1)
    ax1.set_zlim(0, max(4, court_net_height * 2))
    ax2.set_xlim(-2, court_length + 2)
    ax2.set_ylim(0, max(4, court_net_height * 2))
    ax3.set_xlim(-2, court_length + 2)
    ax3.set_ylim(-court_width / 2 - 1, court_width / 2 + 1)
    ax1.set_title(f"3D Trajectories on Court ({current_mode})")
    draw_court()

set_mode(current_mode)

# Initial simulation
traj_rk4, traj_euler, traj_imp_euler = simulate_all_methods(
    initial_speed, initial_theta, initial_phi, initial_wind, initial_omega, initial_dt)

# Plot initial trajectories
rk4_3d, = ax1.plot(traj_rk4[:, 0], traj_rk4[:, 1], traj_rk4[:, 2], 'b-', label='RK4', linewidth=2)
euler_3d, = ax1.plot(traj_euler[:, 0], traj_euler[:, 1], traj_euler[:, 2], 'r--', label='Euler', linewidth=2)
imp_euler_3d, = ax1.plot(traj_imp_euler[:, 0], traj_imp_euler[:, 1], traj_imp_euler[:, 2], 'g:', label='Improved Euler', linewidth=2)

rk4_xz, = ax2.plot(traj_rk4[:, 0], traj_rk4[:, 2], 'b-', label='RK4', linewidth=2, marker='o', markersize=3, markevery=5)
euler_xz, = ax2.plot(traj_euler[:, 0], traj_euler[:, 2], 'r--', label='Euler', linewidth=2, marker='s', markersize=3, markevery=5)
imp_euler_xz, = ax2.plot(traj_imp_euler[:, 0], traj_imp_euler[:, 2], 'g:', label='Improved Euler', linewidth=2, marker='^', markersize=3, markevery=5)

rk4_xy, = ax3.plot(traj_rk4[:, 0], traj_rk4[:, 1], 'b-', label='RK4', linewidth=2)
euler_xy, = ax3.plot(traj_euler[:, 0], traj_euler[:, 1], 'r--', label='Euler', linewidth=2)
imp_euler_xy, = ax3.plot(traj_imp_euler[:, 0], traj_imp_euler[:, 1], 'g:', label='Improved Euler', linewidth=2)

# Formatting - set fixed court limits like court_visualization.py
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.set_title(f"3D Trajectories on Court ({current_mode})")
ax1.legend()
ax1.set_xlim(-2, court_length + 2)
ax1.set_ylim(-court_width / 2 - 1, court_width / 2 + 1)
ax1.set_zlim(0, max(4, court_net_height * 2))

ax2.set_xlabel("Distance X (m)")
ax2.set_ylabel("Height Z (m)")
ax2.set_title("Side View (XZ Plane)")
ax2.legend()
ax2.grid()
ax2.set_xlim(-2, court_length + 2)
ax2.set_ylim(0, max(4, court_net_height * 2))

ax3.set_xlabel("X (m)")
ax3.set_ylabel("Y (m)")
ax3.set_title("Top View (XY Plane)")
ax3.legend()
ax3.grid()
ax3.set_xlim(-2, court_length + 2)
ax3.set_ylim(-court_width / 2 - 1, court_width / 2 + 1)

# ==========================================
# SLIDERS
# ==========================================
# Speed slider
ax_speed = plt.axes([0.1, 0.25, 0.8, 0.03])
speed_slider = Slider(ax_speed, 'Speed (m/s)', 1, 100, valinit=initial_speed)

# Theta slider
ax_theta = plt.axes([0.1, 0.20, 0.8, 0.03])
theta_slider = Slider(ax_theta, r'$\theta$ (deg)', -90,90, valinit=initial_theta)

# Phi slider
ax_phi = plt.axes([0.1, 0.15, 0.8, 0.03])
phi_slider = Slider(ax_phi, r'$\phi$ (deg)', -90, 90, valinit=initial_phi)

# Wind sliders
ax_wind_x = plt.axes([0.1, 0.10, 0.35, 0.03])
wind_x_slider = Slider(ax_wind_x, 'Wind X (m/s)', -200, 200, valinit=initial_wind[0])

ax_wind_y = plt.axes([0.6, 0.10, 0.35, 0.03])
wind_y_slider = Slider(ax_wind_y, 'Wind Y (m/s)', -200, 200, valinit=initial_wind[1])

ax_wind_z = plt.axes([0.1, 0.05, 0.35, 0.03])
wind_z_slider = Slider(ax_wind_z, 'Wind Z (m/s)', -200, 200, valinit=initial_wind[2])

# Spin sliders
ax_spin_x = plt.axes([0.55, 0.05, 0.15, 0.03])
spin_x_slider = Slider(ax_spin_x, 'Spin X (rad/s)', -500, 500, valinit=initial_omega[0])

ax_spin_y = plt.axes([0.1, 0.00, 0.35, 0.03])
spin_y_slider = Slider(ax_spin_y, 'Spin Y (rad/s)', -500, 500, valinit=initial_omega[1])

ax_spin_z = plt.axes([0.55, 0.00, 0.15, 0.03])
spin_z_slider = Slider(ax_spin_z, 'Spin Z (rad/s)', -500, 500, valinit=initial_omega[2])

# Time step slider
ax_dt = plt.axes([0.75, 0.00, 0.15, 0.03])
dt_slider = Slider(ax_dt, 'dt (s)', 0.005, 0.1, valinit=initial_dt)

# Sport mode selector
ax_mode = plt.axes([0.02, 0.88, 0.09, 0.08], facecolor='lightgoldenrodyellow')
mode_radio = RadioButtons(ax_mode, ('Tennis', 'Badminton'), active=0)

def on_mode_change(label):
    set_mode(label)
    update(None)

mode_radio.on_clicked(on_mode_change)

# Update function
def update(val):
    speed = speed_slider.val
    theta = theta_slider.val
    phi = phi_slider.val
    wind = np.array([wind_x_slider.val, wind_y_slider.val, wind_z_slider.val])
    omega = np.array([spin_x_slider.val, spin_y_slider.val, spin_z_slider.val])
    dt = dt_slider.val

    # Run new simulations
    traj_rk4, traj_euler, traj_imp_euler = simulate_all_methods(speed, theta, phi, wind, omega, dt)

    # Update all plots
    rk4_3d.set_data_3d(traj_rk4[:, 0], traj_rk4[:, 1], traj_rk4[:, 2])
    euler_3d.set_data_3d(traj_euler[:, 0], traj_euler[:, 1], traj_euler[:, 2])
    imp_euler_3d.set_data_3d(traj_imp_euler[:, 0], traj_imp_euler[:, 1], traj_imp_euler[:, 2])

    rk4_xz.set_data(traj_rk4[:, 0], traj_rk4[:, 2])
    euler_xz.set_data(traj_euler[:, 0], traj_euler[:, 2])
    imp_euler_xz.set_data(traj_imp_euler[:, 0], traj_imp_euler[:, 2])

    rk4_xy.set_data(traj_rk4[:, 0], traj_rk4[:, 1])
    euler_xy.set_data(traj_euler[:, 0], traj_euler[:, 1])
    imp_euler_xy.set_data(traj_imp_euler[:, 0], traj_imp_euler[:, 1])

    # Update axis limits dynamically
    max_x = max(traj_rk4[-1, 0], traj_euler[-1, 0], traj_imp_euler[-1, 0]) * 1.1
    max_y = max(abs(traj_rk4[:, 1].max()), abs(traj_euler[:, 1].max()), abs(traj_imp_euler[:, 1].max())) * 1.2
    max_z = max(traj_rk4[:, 2].max(), traj_euler[:, 2].max(), traj_imp_euler[:, 2].max()) * 1.1

    # Keep court limits but extend if trajectory goes beyond
    ax1.set_xlim(-2, max(max_x, court_length + 2))
    ax1.set_ylim(-max(max_y, court_width + 3), max(max_y, court_width + 3))
    ax1.set_zlim(0, max(max_z, 4))

    ax2.set_xlim(-2, max(max_x, court_length + 2))
    ax2.set_ylim(0, max(max_z, 4))

    ax3.set_xlim(-2, max(max_x, court_length + 2))
    ax3.set_ylim(-max(max_y, court_width + 3), max(max_y, court_width + 3))

    # Update title with landing differences
    rk4_landing = traj_rk4[-1, 0]
    euler_landing = traj_euler[-1, 0]
    imp_euler_landing = traj_imp_euler[-1, 0]

    diff_euler = abs(euler_landing - rk4_landing)
    diff_imp = abs(imp_euler_landing - rk4_landing)

    fig.suptitle(f'Integration Method Comparison - Landing Differences: Euler: {diff_euler:.2f}m, Improved Euler: {diff_imp:.2f}m',
                 fontsize=12, fontweight='bold')

    fig.canvas.draw_idle()

# Connect sliders to update function
speed_slider.on_changed(update)
theta_slider.on_changed(update)
phi_slider.on_changed(update)
wind_x_slider.on_changed(update)
wind_y_slider.on_changed(update)
wind_z_slider.on_changed(update)
spin_x_slider.on_changed(update)
spin_y_slider.on_changed(update)
spin_z_slider.on_changed(update)
dt_slider.on_changed(update)

# Reset button handler
def reset_sliders(event):
    speed_slider.set_val(initial_speed)
    theta_slider.set_val(initial_theta)
    phi_slider.set_val(initial_phi)
    wind_x_slider.set_val(initial_wind[0])
    wind_y_slider.set_val(initial_wind[1])
    wind_z_slider.set_val(initial_wind[2])
    spin_x_slider.set_val(initial_omega[0])
    spin_y_slider.set_val(initial_omega[1])
    spin_z_slider.set_val(initial_omega[2])
    dt_slider.set_val(initial_dt)
    update(None)

# ax_mode = plt.axes([0.02, 0.88, 0.09, 0.08], facecolor='lightgoldenrodyellow')
ax_err_btn = plt.axes([0.02, 0.78, 0.09, 0.08])
err_button = Button(ax_err_btn, 'Error Analysis', color='lightyellow', hovercolor='yellow')
err_button.on_clicked(lambda event: show_error_analysis())

ax_acc_btn = plt.axes([0.02, 0.68, 0.09, 0.08])
acc_button = Button(ax_acc_btn, 'Accuracy Analysis', color='lightblue', hovercolor='blue')
acc_button.on_clicked(lambda event: show_accuracy_analysis())

ax_reset_btn = plt.axes([0.02, 0.58, 0.09, 0.08])
reset_button = Button(ax_reset_btn, 'Reset Sliders', color='lightgreen', hovercolor='green')
reset_button.on_clicked(reset_sliders)

plt.show()