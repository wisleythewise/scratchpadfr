"""Convert the vendor FR-max Gazebo xacro into a MuJoCo MJCF model.

The conversion is 1:1 with the vendor file: same links, masses, inertias,
joint origins/limits, collision primitives and visual meshes. Gazebo-specific
settings are mapped as follows:
  * rear wheels: velocity control, torque capped at the URDF effort limit
    (gazebo_ros_control's DefaultRobotHWSim uses <limit effort> as fmax)
  * steering hinges: position PID p=100 from smart_control_config.yaml
  * contact friction mu1 from fr-max.gazebo
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, "..", "vendor", "fr-max-gazebo")
XACRO = os.path.join(PKG, "urdf", "fr-max.xacro")
MESHDIR = os.path.join(PKG, "meshes")

# mu1 from urdf/fr-max.gazebo
FRICTION = {
    "base_link": 0.0,
    "left_rear_wheel": 2.0,
    "right_rear_wheel": 2.0,
    "left_front_wheel": 1.0,
    "right_front_wheel": 1.0,
}
STEER_KP = 100.0  # config/smart_control_config.yaml
DRIVE_KV = 100.0


def rpy_to_quat(r, p, y):
    # URDF rpy: fixed-axis X, then Y, then Z  ->  R = Rz(y) Ry(p) Rx(r)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _vec(s, default="0 0 0"):
    return [float(v) for v in (s or default).split()]


def _origin(el):
    o = el.find("origin") if el is not None else None
    if o is None:
        return [0, 0, 0], [0, 0, 0]
    return _vec(o.get("xyz")), _vec(o.get("rpy"))


def _fmt(v):
    return " ".join(f"{x:.6g}" for x in v)


def _pose_attrs(xyz, rpy):
    a = {"pos": _fmt(xyz)}
    if any(rpy):
        a["quat"] = _fmt(rpy_to_quat(*rpy))
    return a


def parse_urdf(path=XACRO):
    root = ET.parse(path).getroot()
    links = {l.get("name"): l for l in root.findall("link")}
    joints = root.findall("joint")
    return links, joints


def build_mjcf(payloads=(), floor_tilt_deg=0.0, extra_worldbody=""):
    """Return MJCF xml string.

    payloads: list of dicts {name, mass, pos(base_link frame), size(box half-extents),
              inertia(optional diag), rgba} rigidly welded to base_link.
    """
    links, joints = parse_urdf()
    children = {}
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j)

    mj = ET.Element("mujoco", model="fr_max")
    ET.SubElement(mj, "compiler", angle="radian", meshdir=MESHDIR, autolimits="true")
    ET.SubElement(mj, "option", timestep="0.001", integrator="implicitfast", cone="elliptic")
    vis = ET.SubElement(mj, "visual")
    ET.SubElement(vis, "global", offwidth="1600", offheight="1000")
    ET.SubElement(vis, "quality", shadowsize="4096")
    ET.SubElement(vis, "headlight", ambient="0.35 0.35 0.35", diffuse="0.6 0.6 0.6")
    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1=".82 .84 .86", rgb2=".74 .76 .78", width="512", height="512")
    ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="8 8", reflectance="0.0")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1=".95 .96 .98", rgb2=".80 .84 .90", width="256", height="256")
    meshes = set()
    for l in links.values():
        for m in l.iter("mesh"):
            meshes.add(os.path.basename(m.get("filename")))
    for m in sorted(meshes):
        ET.SubElement(asset, "mesh", name=os.path.splitext(m)[0], file=m)

    default = ET.SubElement(mj, "default")
    dv = ET.SubElement(default, "default", {"class": "visual"})
    ET.SubElement(dv, "geom", contype="0", conaffinity="0", group="2")
    dc = ET.SubElement(default, "default", {"class": "collision"})
    # robot geoms collide with the world only (Gazebo default: self_collide off)
    ET.SubElement(dc, "geom", group="3", rgba="0.2 0.6 1 0.3", condim="3", contype="2", conaffinity="1")

    wb = ET.SubElement(mj, "worldbody")
    ET.SubElement(wb, "light", pos="2 -2 4", dir="-0.5 0.5 -1", directional="true", castshadow="true",
                  diffuse="0.7 0.7 0.7")
    tilt = math.radians(floor_tilt_deg)
    ET.SubElement(wb, "geom", name="floor", type="plane", size="0 0 0.05", material="grid",
                  quat=_fmt(rpy_to_quat(0, -tilt, 0)), friction="1 0.005 0.0001")
    if extra_worldbody:
        for el in ET.fromstring(f"<w>{extra_worldbody}</w>"):
            wb.append(el)

    colors = {"base_link": "0.80 0.82 0.86 1"}

    def add_link(parent_el, lname, xyz, rpy, joint=None):
        body = ET.SubElement(parent_el, "body", name=lname, **_pose_attrs(xyz, rpy))
        if joint is None:
            ET.SubElement(body, "freejoint", name="root")
        elif joint.get("type") in ("revolute", "continuous"):
            ja = {"name": joint.get("name"), "type": "hinge",
                  "axis": _fmt(_vec(joint.find("axis").get("xyz"), "1 0 0")),
                  "damping": "0.05", "armature": "0.01"}
            lim = joint.find("limit")
            if joint.get("type") == "revolute" and lim is not None:
                ja["range"] = f"{lim.get('lower')} {lim.get('upper')}"
            ET.SubElement(body, "joint", ja)
        l = links[lname]
        inert = l.find("inertial")
        if inert is not None:
            ixyz, irpy = _origin(inert)
            I = inert.find("inertia")
            ia = {"mass": inert.find("mass").get("value"), "pos": _fmt(ixyz),
                  "fullinertia": _fmt([float(I.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")])}
            ET.SubElement(body, "inertial", ia)
        for i, v in enumerate(l.findall("visual")):
            g = v.find("geometry")[0]
            vxyz, vrpy = _origin(v)
            ga = {"class": "visual", "name": f"{lname}_vis{i}", **_pose_attrs(vxyz, vrpy)}
            if g.tag == "mesh":
                ga.update(type="mesh", mesh=os.path.splitext(os.path.basename(g.get("filename")))[0])
                ga["rgba"] = colors.get(lname, "0.12 0.12 0.12 1")
            else:
                continue  # tiny hinge markers
            ET.SubElement(body, "geom", ga)
        for i, c in enumerate(l.findall("collision")):
            g = c.find("geometry")[0]
            cxyz, crpy = _origin(c)
            ga = {"class": "collision", "name": f"{lname}_col{i}", **_pose_attrs(cxyz, crpy)}
            if g.tag == "box":
                ga.update(type="box", size=_fmt(np.array(_vec(g.get("size"))) / 2))
            elif g.tag == "cylinder":
                ga.update(type="cylinder", size=f"{g.get('radius')} {float(g.get('length')) / 2}")
            else:
                continue
            if lname.endswith("steering_hinge"):
                continue  # 1cm marker discs, never touch anything
            mu = FRICTION.get(lname, 1.0)
            ga["friction"] = f"{max(mu, 1e-4)} 0.005 0.0001"
            ET.SubElement(body, "geom", ga)
        if lname == "base_link":
            for p in payloads:
                pb = ET.SubElement(body, "body", name=p["name"], pos=_fmt(p["pos"]))
                ia = {"mass": str(p["mass"]), "pos": "0 0 0"}
                if "inertia" in p:
                    ia["diaginertia"] = _fmt(p["inertia"])
                else:
                    h = np.array(p["size"]) * 2
                    m = p["mass"]
                    ia["diaginertia"] = _fmt([m / 12 * (h[1] ** 2 + h[2] ** 2), m / 12 * (h[0] ** 2 + h[2] ** 2),
                                               m / 12 * (h[0] ** 2 + h[1] ** 2)])
                ET.SubElement(pb, "inertial", ia)
                ET.SubElement(pb, "geom", {"class": "visual", "type": "box", "size": _fmt(p["size"]),
                                           "rgba": p.get("rgba", "0.9 0.4 0.1 1")})
        for j in children.get(lname, []):
            jxyz, jrpy = _origin(j)
            add_link(body, j.find("child").get("link"), jxyz, jrpy, j)

    # base_footprint is the URDF root; it has no inertia so we merge it into the
    # free body by starting the tree at base_link with the footprint offset.
    fp_joint = next(j for j in joints if j.find("parent").get("link") == "base_footprint")
    fxyz, _ = _origin(fp_joint)
    add_link(wb, "base_link", [0, 0, fxyz[2] + 0.002], [0, 0, 0])

    act = ET.SubElement(mj, "actuator")
    eff = {j.get("name"): float(j.find("limit").get("effort")) for j in joints if j.find("limit") is not None}
    for side in ("left", "right"):
        jn = f"{side}_rear_wheel_joint"
        ET.SubElement(act, "velocity", name=f"{side}_drive", joint=jn, kv=str(DRIVE_KV),
                      forcerange=f"{-eff[jn]} {eff[jn]}")
    for side in ("left", "right"):
        jn = f"{side}_steering_hinge_joint"
        ET.SubElement(act, "position", name=f"{side}_steer", joint=jn, kp=str(STEER_KP), kv="10",
                      forcerange=f"{-eff[jn]} {eff[jn]}")

    sens = ET.SubElement(mj, "sensor")
    ET.SubElement(sens, "framepos", name="base_pos", objtype="body", objname="base_link")
    ET.SubElement(sens, "subtreecom", name="com", body="base_link")
    ET.SubElement(mj, "contact")
    ET.indent(mj)
    return ET.tostring(mj, encoding="unicode")


if __name__ == "__main__":
    print(build_mjcf())
