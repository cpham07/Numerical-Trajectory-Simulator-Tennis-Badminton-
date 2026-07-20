# Numerical Trajectory Simulator (Tennis/Badminton)

Compares three numerical integration methods (RK4, Euler, Improved Euler) for simulating projectile motion with drag and Magnus force, evaluating accuracy against real-world officiating tolerances (Hawk-Eye). Includes an interactive 3D visualizer with adjustable speed, spin, and wind.

## Overview
Originally prototyped in 3D Desmos, this projectile motion project 
incorporates drag and the Magnus effect into a sports simulation. It 
compares three numerical integration methods (Euler's, Improved 
Euler's, and RK4) to see how much accuracy each one sacrifices for 
speed, using tennis and badminton trajectories as test cases.

## Features
- Interactive 3D + 2D (side/top view) trajectory visualization
- Adjustable sliders: speed, launch angle (theta/phi), wind (x/y/z), 
  spin (x/y/z), timestep (dt)
- Tennis / Badminton mode toggle (different court dimensions, net height)
- Error Analysis: plots Euler and Improved Euler divergence from RK4 
  over time, broken down by axis
- Accuracy Analysis: empirical convergence order estimation across 
  multiple dt values, landing-position error comparison

## Methodology
- RK4, Euler, and Improved Euler (Heun's method) implemented
- RK4 treated as baseline for error comparison
- Note: UI features (interactive sliders, sport-mode toggle) were specified by me; the 
Hawk-Eye tolerance framing and much of the code implementation were 
AI-assisted.

## Findings
Using RK4 as baseline, the key question was whether the cheaper 
methods can meet Hawk-Eye's real-world tolerance of 3.8mm.

- **Euler's method didn't meet the threshold within tested step 
  sizes** — its 1st-order convergence means it would need a step size 
  far smaller than what's practical for real-time use to close the gap.
- **Improved Euler's method passes, but the required step size 
  depends on speed:**

  | Serve speed | Max dt within 3.8mm |
  |---|---|
  | 64 m/s (standard) | dt ≤ 0.025s |
  | 100 m/s (high speed) | dt ≤ 0.01s |
  | 20 m/s (low speed) | dt ≤ 0.08s |

  Faster shots need a finer timestep to hit the same accuracy bar.

**Bottom line:** Euler's 1st-order convergence means it needs a much 
smaller step size than Improved Euler's to reach Hawk-Eye's 3.8mm 
tolerance — a difference explained directly by their differing 
convergence orders (O(dt) vs O(dt²))

## Requirements
- Python 3.x
- numpy
- matplotlib

## Credits
Built by me as a Differential Equations class project.
