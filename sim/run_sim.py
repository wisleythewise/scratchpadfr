"""Run the vendor FR-max model in MuJoCo (headless) and render it.

Usage:  MUJOCO_GL=osmesa python3 sim/run_sim.py [outdir]

Reproduces the vendor Gazebo demo: spawn on an empty plane, then drive via
/cmd_vel through the same Ackermann mapping as scripts/cmdvel2gazebo.py.
"""
import json
import math
import os
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import urdf2mjcf  # noqa: E402

L, T, R_WHEEL, MAXSTEER = 0.85, 0.665, 0.21, math.radians(25)
WHEELS = ["left_front_wheel", "right_front_wheel", "left_rear_wheel", "right_rear_wheel"]


def cmdvel2gazebo(vx, wz):
    """Port of vendor scripts/cmdvel2gazebo.py -> (steerL, steerR, rearL, rearR)."""
    x = vx / R_WHEEL
    z = max(-MAXSTEER, min(MAXSTEER, wz))
    if z == 0:
        return 0.0, 0.0, x, x
    r = L / math.tan(abs(z))
    s = 1 if z > 0 else -1
    rl, rr = r - s * T / 2, r + s * T / 2
    sl, sr = math.atan(L / rl) * s, math.atan(L / rr) * s
    inner = min(abs(sl), abs(sr))
    if inner > 0.52333333:
        sl, sr = sl * 0.52333333 / inner, sr * 0.52333333 / inner
    return sl, sr, x * rl / r, x * rr / r


def wheel_loads(m, d):
    out = {w: 0.0 for w in WHEELS}
    f = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        for g in (c.geom1, c.geom2):
            b = m.body(m.geom_bodyid[g]).name
            if b in out:
                mujoco.mj_contactForce(m, d, i, f)
                out[b] += f[0]
    return out


def chassis_contact(m, d):
    return any("base_link" in (m.geom(d.contact[i].geom1).name + m.geom(d.contact[i].geom2).name)
               for i in range(d.ncon))


def make_camera(lookat, dist, az, el):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    return cam


def render(m, d, renderer, cam, collision=False):
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    opt.geomgroup[3 if collision else 2] = 1
    if collision:
        opt.geomgroup[2] = 1
    renderer.update_scene(d, camera=cam, scene_option=opt)
    return renderer.render()


def run(outdir, payloads=(), tag="stock", program=None):
    os.makedirs(outdir, exist_ok=True)
    xml = urdf2mjcf.build_mjcf(payloads=payloads)
    with open(os.path.join(outdir, f"fr_max_{tag}.xml"), "w") as fh:
        fh.write(xml)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    W, H = 1280, 720
    renderer = mujoco.Renderer(m, H, W)
    dt = m.opt.timestep

    # cmd_vel program: (t_start, vx, wz)
    program = program or [(0.0, 0.0, 0.0), (2.0, 1.0, 0.0), (8.0, 1.0, 0.4363), (16.0, 1.0, -0.4363),
                          (24.0, 0.0, 0.0), (27.0, None, None)]
    t_end = program[-1][0]

    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "23", os.path.join(outdir, f"drive_{tag}.mp4")], stdin=subprocess.PIPE)
    log = []
    fps_every = int(round(1 / 30 / dt))
    log_every = int(round(0.05 / dt))
    cam = make_camera([0, 0, 0.3], 3.4, 135, -22)
    step = 0
    chassis_hit = False
    while d.time < t_end:
        seg = [p for p in program if p[0] <= d.time][-1]
        vx, wz = seg[1] or 0.0, seg[2] or 0.0
        sl, sr, rl, rr = cmdvel2gazebo(vx, wz)
        d.actuator("left_steer").ctrl = sl
        d.actuator("right_steer").ctrl = sr
        d.actuator("left_drive").ctrl = rl
        d.actuator("right_drive").ctrl = rr
        mujoco.mj_step(m, d)
        step += 1
        chassis_hit |= chassis_contact(m, d)
        if step % log_every == 0:
            q = d.qpos[3:7]
            pitch = math.degrees(math.asin(max(-1, min(1, 2 * (q[0] * q[2] - q[3] * q[1])))))
            roll = math.degrees(math.atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
            v_body = d.qvel[:3]
            loads = wheel_loads(m, d)
            log.append({"t": round(d.time, 3), "cmd_vx": vx, "cmd_wz": wz,
                        "x": float(d.qpos[0]), "y": float(d.qpos[1]),
                        "speed": float(np.linalg.norm(v_body[:2])), "pitch": pitch, "roll": roll,
                        "drive_torque": [float(d.actuator("left_drive").force[0]),
                                         float(d.actuator("right_drive").force[0])],
                        "steer": [float(d.qpos[m.joint("left_steering_hinge_joint").qposadr[0]]),
                                  float(d.qpos[m.joint("right_steering_hinge_joint").qposadr[0]])],
                        "loads": loads})
        if step % fps_every == 0:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.3]
            ff.stdin.write(render(m, d, renderer, cam).tobytes())
    ff.stdin.close()
    ff.wait()

    # stills on a freshly settled model
    mujoco.mj_resetData(m, d)
    for _ in range(1500):
        mujoco.mj_step(m, d)
    static = wheel_loads(m, d)
    com = d.sensor("com").data.copy()
    views = {"iso": ([0, 0, 0.3], 2.7, 135, -25), "side": ([0, 0, 0.28], 3.0, 90, -2),
             "front": ([0, 0, 0.28], 2.6, 0, -2), "top": ([0, 0, 0.2], 3.2, 90, -89)}
    for name, (la, dist, az, el) in views.items():
        Image.fromarray(render(m, d, renderer, make_camera(la, dist, az, el))).save(
            os.path.join(outdir, f"{tag}_{name}.png"))
    Image.fromarray(render(m, d, renderer, make_camera([0, 0, 0.3], 2.7, 135, -25), collision=True)).save(
        os.path.join(outdir, f"{tag}_collision.png"))

    total_mass = float(sum(m.body_mass))
    summary = {
        "tag": tag, "total_mass_kg": total_mass, "weight_N": total_mass * 9.81,
        "com_world_m": com.tolist(), "static_wheel_loads_N": static,
        "static_sum_N": sum(static.values()),
        "chassis_touched_ground_while_driving": bool(chassis_hit),
        "max_speed_mps": max(r["speed"] for r in log),
        "max_abs_pitch_deg": max(abs(r["pitch"]) for r in log),
        "max_abs_roll_deg": max(abs(r["roll"]) for r in log),
        "drive_torque_saturated_frac": float(np.mean([max(abs(t) for t in r["drive_torque"]) > 9.99
                                                      for r in log if r["cmd_vx"] != 0])),
        "min_wheel_load_N": min(min(r["loads"].values()) for r in log),
    }
    with open(os.path.join(outdir, f"summary_{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(outdir, f"log_{tag}.json"), "w") as fh:
        json.dump(log, fh)
    return summary


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "out")
    print(json.dumps(run(out), indent=2))
