# scratchpadfr

Load-capacity check for a Fairino FR20 (on a swivel) mounted on a Yuhesen FR-max chassis.

## Layout
- `vendor/fr-max-gazebo/` – the manufacturer's Gazebo (ROS1) package as shipped: xacro/URDF, meshes, controller config, cmd_vel bridge.
- `sim/urdf2mjcf.py` – converts the vendor xacro 1:1 into a MuJoCo model (same masses, inertias, joints, collision shapes, actuator limits). Gazebo/ROS isn't needed.
- `sim/run_sim.py` – runs the vendor demo (spawn, then drive through the vendor's `cmdvel2gazebo.py` Ackermann mapping), logs wheel loads and pose, and renders stills plus an mp4.

## Run
```
pip install mujoco pillow numpy   # plus ffmpeg and libosmesa6 for headless rendering
MUJOCO_GL=osmesa python3 sim/run_sim.py out
```
