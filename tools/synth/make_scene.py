"""tools/synth/make_scene.py — deterministic HDR test scene with exact camera ground truth.

Builds a procedural room in Blender (8 x 6 m, 3 m high, one side open to an HDRI sky), lights it with a tiny very
bright pendant lamp, a weak area fill and the sky, and renders N views on an arc to linear float EXR with Cycles.
The scene spans ~14 stops: a dark alcove (albedo 0.04) far from the opening sits below 0.005, the sky around 5-20,
the lamp's surface at ~400. The bracket generator (`make_brackets.py`) turns the EXRs into -2/0/+2 EV JPEGs.

Run with Blender's own Python (it has numpy):

    blender -b --factory-startup --python tools/synth/make_scene.py -- \
        --out data/brackets_synth/gt --views 40 --width 1024 --height 768 --samples 128 [--preview] [--seed 7]

    --preview    4 of the views at 16 samples into <out>/preview/ (+ tonemapped PNGs) to check look and range
    --check      only re-run the self-test on renders already in <out> (no render)
    --no-check   skip the self-test after rendering

Writes into <out>:
    view_NNN.exr               linear scene radiance, RGB float32, ZIP (no view transform)
    cameras.json               intrinsics + per view: Blender camera-to-world, COLMAP qvec/tvec, position
    sparse_gt/{cameras,images,points3D}.txt   COLMAP TEXT model (one PINHOLE camera, images named view_NNN.jpg)
    scene.json                 lamp, HDRI, seed, Blender version, render settings, range stats of view_000
    scene.blend                the scene itself, for inspection

Conventions (verified by the self-test, which projects the lamp centre and finds it in the rendered EXR):
    Blender camera: looks down its local -Z, +Y up, +X right. `blender_c2w` = camera.matrix_world (row-major).
    COLMAP camera: looks down +Z, +Y down, +X right. x_cam = R @ X_world + t, u = fx*x/z + cx, v = fy*y/z + cy,
    pixel (0, 0) = top-left corner of the top-left pixel (pixel centres at +0.5).
    R_colmap = diag(1,-1,-1) @ R_blender_w2c,  t_colmap = diag(1,-1,-1) @ t_blender_w2c.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

# ── Constants ────────────────────────────────────────────────────────────────

HDRI = Path(
    "/home/romain/Documents/Projects/20260924_Vesoul_DrivingPlate/assets/polyhaven/hdris/"
    "overcast_soil_puresky/overcast_soil_puresky_4k.hdr"
)
SKY_TARGET = 10.0  # median linear radiance of the HDRI's 0-30 deg elevation band (what the cameras see) after scaling

ROOM_X = 4.0  # interior x in [-4, 4]
ROOM_Y = 3.0  # interior y in [-3, 3]; the side y = -3 is open
ROOM_H = 3.0
WALL_T = 0.2

LAMP_POS = Vector((0.35, 0.55, 1.55))
LAMP_RADIUS = 0.06
LAMP_RADIANCE = 400.0
FILL_POWER_W = 20.0

TARGET = Vector((0.0, 0.3, 1.2))
ARC_RADIUS = 2.5
ARC_START_DEG = 40.0  # position angle of view_000 around the target (0 = +X); the arc sweeps counter-clockwise
ARC_SPAN_DEG = 150.0

FOCAL_MM = 35.0
SENSOR_W_MM = 36.0

# Alcove (dark niche) cut into the west wall, far from the opening, mouth facing +X.
NICHE_Y = (-1.1, 0.1)
NICHE_Z = (0.2, 2.2)
NICHE_DEPTH = 4.0

FLIP = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))  # Blender camera axes -> COLMAP camera axes


def log(msg: str) -> None:
    print(f"[make_scene] {msg}", flush=True)


def sig(x: float, n: int = 9) -> float:
    return float(f"{x:.{n}g}")


# ── Arguments ────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(prog="make_scene.py", description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--views", type=int, default=40)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--preview", action="store_true", help="4 views at 16 samples into <out>/preview/")
    ap.add_argument("--check", action="store_true", help="only run the self-test on existing renders")
    ap.add_argument("--no-check", action="store_true", help="skip the self-test after rendering")
    ap.add_argument("--cpu", action="store_true", help="force CPU rendering")
    return ap.parse_args(argv)


# ── Materials ────────────────────────────────────────────────────────────────


class NodeBuilder:
    """Small helper over a material's node tree; positions are irrelevant (nobody edits these by hand)."""

    def __init__(self, name: str):
        self.mat = bpy.data.materials.new(name)
        self.mat.use_nodes = True
        self.nt = self.mat.node_tree
        self.nt.nodes.clear()
        self.out = self.node("ShaderNodeOutputMaterial")
        self.bsdf = self.node("ShaderNodeBsdfPrincipled")
        self.link(self.bsdf.outputs["BSDF"], self.out.inputs["Surface"])

    def node(self, kind: str, **inputs):
        n = self.nt.nodes.new(kind)
        for key, val in inputs.items():
            if key in n.inputs:
                n.inputs[key].default_value = val
            else:
                setattr(n, key, val)
        return n

    def link(self, a, b) -> None:
        self.nt.links.new(a, b)

    def coords(self, world: bool):
        if world:
            return self.node("ShaderNodeNewGeometry").outputs["Position"]
        return self.node("ShaderNodeTexCoord").outputs["Object"]

    def ramp(self, fac, lo: tuple, hi: tuple, lo_pos: float = 0.0, hi_pos: float = 1.0):
        r = self.node("ShaderNodeValToRGB")
        r.color_ramp.elements[0].position = lo_pos
        r.color_ramp.elements[0].color = (*lo, 1.0)
        r.color_ramp.elements[1].position = hi_pos
        r.color_ramp.elements[1].color = (*hi, 1.0)
        self.link(fac, r.inputs["Fac"])
        return r.outputs["Color"]

    def mix(self, a, b, fac, blend: str = "MIX"):
        m = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        if isinstance(fac, float):
            m.inputs[0].default_value = fac
        else:
            self.link(fac, m.inputs[0])
        self.link(a, m.inputs[6])
        self.link(b, m.inputs[7])
        return m.outputs[2]

    def bump(self, height, strength: float, distance: float = 0.02) -> None:
        b = self.node("ShaderNodeBump", Strength=strength, Distance=distance)
        self.link(height, b.inputs["Height"])
        self.link(b.outputs["Normal"], self.bsdf.inputs["Normal"])

    def set(self, **inputs) -> None:
        for key, val in inputs.items():
            self.bsdf.inputs[key].default_value = val


def tint(rgb: tuple, k: float) -> tuple:
    return tuple(max(0.0, min(1.0, c * k)) for c in rgb)


def mat_floor() -> bpy.types.Material:
    """Brick tiles (large scale) with a fine noise grime and per-tile variation, bumped grout."""
    nb = NodeBuilder("floor_tiles")
    co = nb.coords(world=True)
    brick = nb.node(
        "ShaderNodeTexBrick", Scale=1.6, **{"Mortar Size": 0.012, "Mortar Smooth": 0.1, "Bias": 0.0},
        offset=0.5, squash=1.0,
    )
    brick.inputs["Color1"].default_value = (0.55, 0.42, 0.30, 1)
    brick.inputs["Color2"].default_value = (0.30, 0.22, 0.17, 1)
    brick.inputs["Mortar"].default_value = (0.08, 0.08, 0.08, 1)
    nb.link(co, brick.inputs["Vector"])
    fine = nb.node("ShaderNodeTexNoise", Scale=60.0, Detail=8.0, Roughness=0.65)
    nb.link(co, fine.inputs["Vector"])
    vor = nb.node("ShaderNodeTexVoronoi", Scale=9.0)
    nb.link(co, vor.inputs["Vector"])
    grime = nb.ramp(fine.outputs["Fac"], (0.55, 0.55, 0.55), (1.2, 1.15, 1.1), 0.3, 0.7)
    spots = nb.ramp(vor.outputs["Distance"], (0.6, 0.6, 0.6), (1.0, 1.0, 1.0), 0.0, 0.35)
    col = nb.mix(brick.outputs["Color"], grime, 1.0, "MULTIPLY")
    col = nb.mix(col, spots, 1.0, "MULTIPLY")
    nb.link(col, nb.bsdf.inputs["Base Color"])
    h = nb.mix(brick.outputs["Fac"], fine.outputs["Fac"], 0.3)
    nb.bump(h, 0.5)
    nb.set(Roughness=0.55)
    return nb.mat


def mat_wall(name: str, base: tuple, scale_big: float, scale_fine: float, stripes: bool) -> bpy.types.Material:
    """Plaster: big voronoi cells + fine noise at a second scale, optional checker wainscot band; bumped."""
    nb = NodeBuilder(name)
    co = nb.coords(world=True)
    vor = nb.node("ShaderNodeTexVoronoi", Scale=scale_big, feature="F1")
    nb.link(co, vor.inputs["Vector"])
    fine = nb.node("ShaderNodeTexNoise", Scale=scale_fine, Detail=10.0, Roughness=0.7)
    nb.link(co, fine.inputs["Vector"])
    cells = nb.ramp(vor.outputs["Distance"], tint(base, 1.15), tint(base, 0.65), 0.0, 0.6)
    speck = nb.ramp(fine.outputs["Fac"], (0.6, 0.6, 0.6), (1.25, 1.25, 1.25), 0.35, 0.65)
    col = nb.mix(cells, speck, 1.0, "MULTIPLY")
    if stripes:
        chk = nb.node("ShaderNodeTexChecker", Scale=6.0)
        chk.inputs["Color1"].default_value = (*tint(base, 0.45), 1)
        chk.inputs["Color2"].default_value = (*tint(base, 1.2), 1)
        nb.link(co, chk.inputs["Vector"])
        sep = nb.node("ShaderNodeSeparateXYZ")
        nb.link(co, sep.inputs["Vector"])
        band = nb.node("ShaderNodeMath", operation="LESS_THAN")
        band.inputs[1].default_value = 0.9
        nb.link(sep.outputs["Z"], band.inputs[0])
        col2 = nb.mix(chk.outputs["Color"], speck, 1.0, "MULTIPLY")
        m = nb.node("ShaderNodeMix", data_type="RGBA", blend_type="MIX")
        nb.link(band.outputs[0], m.inputs[0])
        nb.link(col, m.inputs[6])
        nb.link(col2, m.inputs[7])
        col = m.outputs[2]
    nb.link(col, nb.bsdf.inputs["Base Color"])
    h = nb.mix(vor.outputs["Distance"], fine.outputs["Fac"], 0.5)
    nb.bump(h, 0.35)
    nb.set(Roughness=0.8)
    return nb.mat


def mat_ceiling() -> bpy.types.Material:
    """Checker panels with a noise overlay at two scales."""
    nb = NodeBuilder("ceiling")
    co = nb.coords(world=True)
    chk = nb.node("ShaderNodeTexChecker", Scale=2.0)
    chk.inputs["Color1"].default_value = (0.70, 0.70, 0.68, 1)
    chk.inputs["Color2"].default_value = (0.45, 0.47, 0.50, 1)
    nb.link(co, chk.inputs["Vector"])
    n1 = nb.node("ShaderNodeTexNoise", Scale=4.0, Detail=4.0)
    nb.link(co, n1.inputs["Vector"])
    n2 = nb.node("ShaderNodeTexNoise", Scale=45.0, Detail=8.0, Roughness=0.7)
    nb.link(co, n2.inputs["Vector"])
    a = nb.ramp(n1.outputs["Fac"], (0.7, 0.7, 0.7), (1.1, 1.1, 1.1), 0.3, 0.7)
    b = nb.ramp(n2.outputs["Fac"], (0.6, 0.6, 0.6), (1.2, 1.2, 1.2), 0.35, 0.65)
    col = nb.mix(nb.mix(chk.outputs["Color"], a, 1.0, "MULTIPLY"), b, 1.0, "MULTIPLY")
    nb.link(col, nb.bsdf.inputs["Base Color"])
    nb.bump(n2.outputs["Fac"], 0.3)
    nb.set(Roughness=0.85)
    return nb.mat


def mat_flat(name: str, albedo: float, rough: float = 0.9, detail: bool = True) -> bpy.types.Material:
    """Near-uniform albedo (the alcove: 0.04) with a faint fine noise so it is not featureless."""
    nb = NodeBuilder(name)
    if detail:
        co = nb.coords(world=True)
        n = nb.node("ShaderNodeTexNoise", Scale=30.0, Detail=6.0)
        nb.link(co, n.inputs["Vector"])
        col = nb.ramp(n.outputs["Fac"], (albedo * 0.7,) * 3, (albedo * 1.3,) * 3, 0.3, 0.7)
        nb.link(col, nb.bsdf.inputs["Base Color"])
        nb.bump(n.outputs["Fac"], 0.2)
    else:
        nb.set(**{"Base Color": (albedo, albedo, albedo, 1)})
    nb.set(Roughness=rough)
    return nb.mat


def mat_ground() -> bpy.types.Material:
    nb = NodeBuilder("ground_outside")
    co = nb.coords(world=True)
    n1 = nb.node("ShaderNodeTexNoise", Scale=0.8, Detail=6.0)
    nb.link(co, n1.inputs["Vector"])
    n2 = nb.node("ShaderNodeTexNoise", Scale=25.0, Detail=8.0, Roughness=0.7)
    nb.link(co, n2.inputs["Vector"])
    a = nb.ramp(n1.outputs["Fac"], (0.16, 0.13, 0.09), (0.32, 0.30, 0.22), 0.35, 0.65)
    b = nb.ramp(n2.outputs["Fac"], (0.6, 0.6, 0.6), (1.3, 1.3, 1.3), 0.3, 0.7)
    nb.link(nb.mix(a, b, 1.0, "MULTIPLY"), nb.bsdf.inputs["Base Color"])
    nb.bump(n2.outputs["Fac"], 0.4)
    nb.set(Roughness=0.95)
    return nb.mat


def mat_object(name: str, rgb: tuple, rng: random.Random, metallic: bool) -> bpy.types.Material:
    """Object albedo `rgb` modulated +-35% by a voronoi/noise mix (object coords) so SfM finds features on it."""
    nb = NodeBuilder(name)
    co = nb.coords(world=False)
    kind = rng.choice(["voronoi", "checker", "noise"])
    if kind == "voronoi":
        t = nb.node("ShaderNodeTexVoronoi", Scale=rng.uniform(5, 12))
        fac = t.outputs["Distance"]
    elif kind == "checker":
        t = nb.node("ShaderNodeTexChecker", Scale=rng.uniform(4, 9))
        fac = t.outputs["Fac"]
    else:
        t = nb.node("ShaderNodeTexNoise", Scale=rng.uniform(4, 9), Detail=4.0)
        fac = t.outputs["Fac"]
    nb.link(co, t.inputs["Vector"])
    fine = nb.node("ShaderNodeTexNoise", Scale=rng.uniform(40, 70), Detail=8.0, Roughness=0.7)
    nb.link(co, fine.inputs["Vector"])
    big = nb.ramp(fac, tint(rgb, 0.75), tint(rgb, 1.25), 0.2, 0.8)
    sm = nb.ramp(fine.outputs["Fac"], (0.8, 0.8, 0.8), (1.15, 1.15, 1.15), 0.3, 0.7)
    nb.link(nb.mix(big, sm, 1.0, "MULTIPLY"), nb.bsdf.inputs["Base Color"])
    nb.bump(fine.outputs["Fac"], 0.15 if metallic else 0.3, 0.01)
    if metallic:
        nb.set(Metallic=1.0, Roughness=rng.uniform(0.12, 0.22))
    else:
        nb.set(Roughness=rng.uniform(0.45, 0.85))
    return nb.mat


# ── Geometry ─────────────────────────────────────────────────────────────────


def apply_xf(obj) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def box(name: str, lo: tuple, hi: tuple, mat) -> bpy.types.Object:
    centre = [(a + b) / 2 for a, b in zip(lo, hi)]
    size = [b - a for a, b in zip(lo, hi)]
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=centre)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    apply_xf(obj)
    obj.data.materials.append(mat)
    return obj


def build_room() -> None:
    x, y, h, t = ROOM_X, ROOM_Y, ROOM_H, WALL_T
    floor = mat_floor()
    box("floor", (-x - t, -y, -t), (x + t, y + t, 0.0), floor)
    box("ceiling", (-x - t, -y, h), (x + t, y + t, h + t), mat_ceiling())
    box("wall_north", (-x - t, y, 0.0), (x + t, y + t, h), mat_wall("wall_north", (0.62, 0.58, 0.52), 3.0, 55.0, True))
    box("wall_east", (x, -y, 0.0), (x + t, y, h), mat_wall("wall_east", (0.45, 0.55, 0.62), 5.0, 70.0, False))
    # West wall with the alcove's mouth cut out of it.
    west = mat_wall("wall_west", (0.66, 0.50, 0.42), 4.0, 40.0, True)
    (y0, y1), (z0, z1) = NICHE_Y, NICHE_Z
    box("wall_west_s", (-x - t, -y, 0.0), (-x, y0, h), west)
    box("wall_west_n", (-x - t, y1, 0.0), (-x, y, h), west)
    box("wall_west_lo", (-x - t, y0, 0.0), (-x, y1, z0), west)
    box("wall_west_hi", (-x - t, y0, z1), (-x, y1, h), west)
    # The alcove behind it: albedo 0.04 inside, a cell whose back sees only the room interior.
    dark = mat_flat("alcove_dark", 0.04)
    xb = -x - t - NICHE_DEPTH
    box("alcove_back", (xb - t, y0 - t, z0 - t), (xb, y1 + t, z1 + t), dark)
    box("alcove_side_s", (xb, y0 - t, z0 - t), (-x - t, y0, z1 + t), dark)
    box("alcove_side_n", (xb, y1, z0 - t), (-x - t, y1 + t, z1 + t), dark)
    box("alcove_floor", (xb, y0, z0 - t), (-x - t, y1, z0), dark)
    box("alcove_top", (xb, y0, z1), (-x - t, y1, z1 + t), dark)
    # Ground outside the opening (under everything; the floor slab hides it indoors).
    bpy.ops.mesh.primitive_plane_add(size=80.0, location=(0.0, 0.0, -0.01))
    g = bpy.context.active_object
    g.name = "ground_outside"
    g.data.materials.append(mat_ground())


def build_lights() -> dict:
    # Pendant lamp: an emissive sphere (radiance = emission strength) hanging from a thin cord.
    bpy.ops.mesh.primitive_uv_sphere_add(radius=LAMP_RADIUS, location=LAMP_POS, segments=48, ring_count=24)
    lamp = bpy.context.active_object
    lamp.name = "lamp"
    bpy.ops.object.shade_smooth()
    m = bpy.data.materials.new("lamp_emission")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 0.92, 0.80, 1)  # warm; Rec.709 luminance ~0.93
    em.inputs["Strength"].default_value = LAMP_RADIANCE
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    lamp.data.materials.append(m)
    cord_len = ROOM_H - LAMP_POS.z - LAMP_RADIUS
    bpy.ops.mesh.primitive_cylinder_add(
        radius=0.004, depth=cord_len, location=(LAMP_POS.x, LAMP_POS.y, ROOM_H - cord_len / 2), vertices=12
    )
    cord = bpy.context.active_object
    cord.name = "lamp_cord"
    cord.data.materials.append(mat_flat("cord", 0.05, 0.5, detail=False))
    # Weak area fill under the ceiling, pointing down, so shadows are not black.
    fill_data = bpy.data.lights.new("fill", type="AREA")
    fill_data.shape = "RECTANGLE"
    fill_data.size, fill_data.size_y = 2.0, 1.5
    fill_data.energy = FILL_POWER_W
    fill_data.color = (0.9, 0.95, 1.0)
    fill = bpy.data.objects.new("fill", fill_data)
    fill.location = (1.0, 0.8, ROOM_H - 0.05)
    bpy.context.scene.collection.objects.link(fill)
    col = em.inputs["Color"].default_value
    lum = 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2]
    return {
        "lamp_position": [sig(v, 6) for v in LAMP_POS],
        "lamp_radius_m": LAMP_RADIUS,
        "lamp_emission_strength": LAMP_RADIANCE,
        "lamp_color": [sig(v, 6) for v in col[:3]],
        "lamp_radiance_rgb": [sig(LAMP_RADIANCE * c, 6) for c in col[:3]],
        "lamp_radiance_luminance": sig(LAMP_RADIANCE * lum, 6),
        "fill": {"type": "AREA", "size_m": [2.0, 1.5], "power_w": FILL_POWER_W, "position": list(fill.location)},
    }


def build_objects(rng: random.Random) -> list[dict]:
    """8-10 floor objects (albedo 0.08..0.9, two metallic), placed by seeded rejection sampling."""
    specs = [
        # name, kind, footprint radius, albedo, metallic, extra
        ("cube", "cube", 0.36, 0.50, False, {"size": 0.5}),
        ("tall_cylinder", "cylinder", 0.27, 0.70, False, {"r": 0.25, "h": 1.45}),
        ("torus_gold", "torus", 0.47, 0.90, True, {"R": 0.35, "r": 0.11}),
        ("suzanne", "monkey", 0.40, 0.30, False, {"size": 0.38}),
        ("cone_dark", "cone", 0.32, 0.08, False, {"r": 0.3, "h": 0.8}),
        ("stack", "stack", 0.42, 0.20, False, {}),
        ("pillar", "pillar", 0.23, 0.15, False, {"w": 0.3, "h": 1.7}),
        ("chrome_cylinder", "cylinder", 0.24, 0.85, True, {"r": 0.22, "h": 0.6}),
    ]
    tints = [(1.0, 0.95, 0.9), (0.9, 1.0, 0.95), (1.0, 0.85, 0.6), (0.85, 0.9, 1.0), (1.0, 0.9, 0.95)]
    placed: list[tuple[float, float, float]] = []
    out = []

    def free_spot(rad: float) -> tuple[float, float]:
        for _ in range(5000):
            px = rng.uniform(-ROOM_X + 0.4 + rad, ROOM_X - 0.4 - rad)
            py = rng.uniform(-ROOM_Y + 0.6 + rad, ROOM_Y - 0.4 - rad)
            d = math.hypot(px - TARGET.x, py - TARGET.y)
            if not (d + rad < ARC_RADIUS - 0.7 or d - rad > ARC_RADIUS + 0.45):
                continue  # keep the camera ring clear
            if d - rad > ARC_RADIUS + 0.45 and rng.random() < 0.6:
                continue  # prefer the middle of the room, some objects outside the ring
            if math.hypot(px - LAMP_POS.x, py - LAMP_POS.y) < rad + 0.35:
                continue
            if px < -ROOM_X + 1.6 and NICHE_Y[0] - 0.8 < py < NICHE_Y[1] + 0.8:
                continue  # keep the alcove's mouth unobstructed
            if all(math.hypot(px - qx, py - qy) > rad + qr + 0.15 for qx, qy, qr in placed):
                placed.append((px, py, rad))
                return px, py
        raise RuntimeError("could not place objects; room too crowded")

    for name, kind, rad, alb, metal, ex in specs:
        px, py = free_spot(rad)
        yaw = rng.uniform(0, 2 * math.pi)
        rgb = tint(rng.choice(tints), alb)
        rgb = tuple(min(0.9, c) for c in rgb)
        objs = []
        if kind == "cube":
            s = ex["size"]
            bpy.ops.mesh.primitive_cube_add(size=s, location=(px, py, s / 2))
            objs.append(bpy.context.active_object)
        elif kind == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(
                radius=ex["r"], depth=ex["h"], location=(px, py, ex["h"] / 2), vertices=64
            )
            objs.append(bpy.context.active_object)
            bpy.ops.object.shade_smooth()
        elif kind == "torus":
            bpy.ops.mesh.primitive_torus_add(
                major_radius=ex["R"], minor_radius=ex["r"], location=(px, py, ex["r"]),
                major_segments=64, minor_segments=24,
            )
            objs.append(bpy.context.active_object)
            bpy.ops.object.shade_smooth()
        elif kind == "monkey":
            s = ex["size"]
            bpy.ops.mesh.primitive_monkey_add(size=2 * s, location=(px, py, 0.0))
            o = bpy.context.active_object
            m2 = bpy.data.objects[o.name].modifiers.new("sub", "SUBSURF")
            m2.levels = m2.render_levels = 2
            o.rotation_euler = Euler((0.0, 0.0, 0.0))
            bpy.ops.object.shade_smooth()
            bpy.context.view_layer.update()
            zmin = min((o.matrix_world @ Vector(c)).z for c in o.bound_box)
            o.location.z -= zmin
            objs.append(o)
        elif kind == "cone":
            bpy.ops.mesh.primitive_cone_add(radius1=ex["r"], depth=ex["h"], location=(px, py, ex["h"] / 2), vertices=64)
            objs.append(bpy.context.active_object)
            bpy.ops.object.shade_smooth()
        elif kind == "pillar":
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(px, py, ex["h"] / 2))
            o = bpy.context.active_object
            o.scale = (ex["w"], ex["w"], ex["h"])
            apply_xf(o)
            objs.append(o)
        elif kind == "stack":
            z = 0.0
            for k, (s, a) in enumerate([(0.55, 0.20), (0.40, 0.60), (0.28, 0.90)]):
                bpy.ops.mesh.primitive_cube_add(size=s, location=(px, py, z + s / 2))
                o = bpy.context.active_object
                o.rotation_euler.z = rng.uniform(-0.6, 0.6)
                o.name = f"stack_{k}"
                o["albedo"] = a
                objs.append(o)
                z += s
        for k, o in enumerate(objs):
            if kind != "stack":
                o.name = name
                o.rotation_euler.z = yaw
                a = alb
            else:
                a = o["albedo"]
            col = tint(rng.choice(tints), a) if kind == "stack" else rgb
            col = tuple(min(0.9, c) for c in col)
            o.data.materials.append(mat_object(f"mat_{o.name}", col, rng, metal))
            out.append({
                "name": o.name, "kind": kind, "albedo": a, "base_color": [sig(c, 4) for c in col],
                "metallic": metal, "location": [sig(v, 6) for v in o.location],
            })
    return out


# ── World ────────────────────────────────────────────────────────────────────


def hdri_strength() -> tuple[float, dict, bpy.types.Image]:
    """Strength that puts the median luminance of the HDRI's 0-30 deg elevation band at SKY_TARGET.

    This overcast sky is ~2x brighter at the zenith than at the horizon; the cameras look roughly level, so the
    band they see through the opening is what has to land in the 5-20 range.
    """
    img = bpy.data.images.load(str(HDRI), check_existing=True)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)[: , :, :3]  # row 0 = bottom = nadir
    lum = px @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    upper = lum[h // 2 :]
    band = lum[h // 2 : h // 2 + h // 6]  # elevation 0..30 deg
    med = float(np.median(band))
    return SKY_TARGET / med, {
        "calibration_band_deg": [0, 30],
        "band_median_raw": sig(med, 6),
        "band_p10_raw": sig(float(np.percentile(band, 10)), 6),
        "band_p90_raw": sig(float(np.percentile(band, 90)), 6),
        "upper_hemisphere_median_raw": sig(float(np.median(upper)), 6),
        "upper_hemisphere_p99_raw": sig(float(np.percentile(upper, 99)), 6),
        "upper_hemisphere_max_raw": sig(float(upper.max()), 6),
        "lower_hemisphere_median_raw": sig(float(np.median(lum[: h // 2])), 6),
    }, img


def build_world() -> dict:
    strength, stats, img = hdri_strength()
    world = bpy.data.worlds.new("hdri")
    bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = img
    env.projection = "EQUIRECTANGULAR"
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return {"hdri": str(HDRI), "hdri_strength": sig(strength, 6), "sky_target_median": SKY_TARGET, **stats}


# ── Cameras ──────────────────────────────────────────────────────────────────


def camera_poses(n: int, rng: random.Random) -> list[Matrix]:
    poses = []
    step = ARC_SPAN_DEG / max(1, n - 1)
    for i in range(n):
        theta = math.radians(ARC_START_DEG + i * step + rng.uniform(-0.3, 0.3) * step)
        r = ARC_RADIUS + rng.uniform(-0.15, 0.15)  # forward/back jitter: parallax between neighbours
        z = rng.uniform(1.2, 1.8)
        pos = Vector((TARGET.x + r * math.cos(theta), TARGET.y + r * math.sin(theta), z))
        rot = (TARGET - pos).to_track_quat("-Z", "Y").to_matrix()
        # +-3 deg aim jitter: yaw about world Z, then pitch about the camera's own X.
        yaw = Matrix.Rotation(math.radians(rng.uniform(-3, 3)), 3, "Z")
        pitch = Matrix.Rotation(math.radians(rng.uniform(-3, 3)), 3, "X")
        rot = yaw @ rot @ pitch
        poses.append(Matrix.Translation(pos) @ rot.to_4x4())
    return poses


def colmap_pose(c2w: Matrix) -> tuple[list, list, Matrix, Vector]:
    r_c2w = c2w.to_3x3()
    r_w2c = r_c2w.transposed()
    t_w2c = -(r_w2c @ c2w.translation)
    r = FLIP @ r_w2c
    t = FLIP @ t_w2c
    q = r.to_quaternion().normalized()
    qv = [q.w, q.x, q.y, q.z]
    if qv[0] < 0:
        qv = [-v for v in qv]
    return qv, list(t), r, t


def make_cameras(poses: list[Matrix], width: int, height: int) -> list[bpy.types.Object]:
    cams = []
    for i, m in enumerate(poses):
        cd = bpy.data.cameras.new(f"view_{i:03d}")
        cd.lens = FOCAL_MM
        cd.sensor_width = SENSOR_W_MM
        cd.sensor_fit = "HORIZONTAL"
        cd.shift_x = cd.shift_y = 0.0
        cd.clip_start, cd.clip_end = 0.05, 200.0
        ob = bpy.data.objects.new(f"view_{i:03d}", cd)
        bpy.context.scene.collection.objects.link(ob)
        ob.matrix_world = m
        cams.append(ob)
    bpy.context.view_layer.update()
    return cams


def intrinsics(width: int, height: int) -> dict:
    fx = FOCAL_MM / SENSOR_W_MM * width
    return {"width": width, "height": height, "fx": fx, "fy": fx, "cx": width / 2, "cy": height / 2,
            "sensor_width_mm": SENSOR_W_MM, "focal_mm": FOCAL_MM}


def lamp_visible(cam_pos: Vector) -> bool:
    deps = bpy.context.evaluated_depsgraph_get()
    d = LAMP_POS - cam_pos
    hit, loc, _n, _i, obj, _m = bpy.context.scene.ray_cast(deps, cam_pos, d.normalized(), distance=d.length + 0.01)
    return bool(hit) and obj is not None and obj.name == "lamp"


# ── Render setup ─────────────────────────────────────────────────────────────


def setup_render(args, samples: int) -> str:
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    device = "CPU"
    if not args.cpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            prefs.get_devices()
            devs = [d for d in prefs.devices if d.type == kind]
            if devs:
                for d in prefs.devices:
                    d.use = d.type == kind
                device = f"{kind}: " + ", ".join(d.name for d in devs)
                sc.cycles.device = "GPU"
                break
    if device == "CPU":
        sc.cycles.device = "CPU"
        log("GPU NOT AVAILABLE (no OPTIX/CUDA device) - rendering on CPU" if not args.cpu else "CPU forced")
    log(f"render device: {device}")
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.seed = args.seed
    sc.cycles.use_animated_seed = False
    sc.cycles.sample_clamp_direct = 0.0
    sc.cycles.sample_clamp_indirect = 0.0  # no clamping: keep the radiance physical (HDR ground truth)
    sc.cycles.film_exposure = 1.0
    sc.render.film_transparent = False
    sc.render.resolution_x = args.width
    sc.render.resolution_y = args.height
    sc.render.resolution_percentage = 100
    sc.render.pixel_aspect_x = sc.render.pixel_aspect_y = 1.0
    sc.render.use_border = False
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.view_settings.gamma = 1.0
    sc.display_settings.display_device = "sRGB"
    im = sc.render.image_settings
    im.file_format = "OPEN_EXR"
    im.color_mode = "RGB"
    im.color_depth = "32"
    im.exr_codec = "ZIP"
    return device


# ── EXR reading + statistics ─────────────────────────────────────────────────

LUM = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def read_exr(path: Path) -> np.ndarray:
    """(H, W, 3) float array, row 0 = TOP of the image (Blender's buffer is bottom-up)."""
    img = bpy.data.images.load(str(path), check_existing=False)
    w, h = img.size
    px = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(px)
    arr = px.reshape(h, w, img.channels)[::-1, :, :3].astype(np.float64)
    bpy.data.images.remove(img)
    return arr


def range_stats(rgb: np.ndarray) -> dict:
    y = rgb @ LUM
    p1, p50, p99 = np.percentile(y, [1, 50, 99])
    pos = y[y > 0]
    return {
        "luminance_min": sig(float(y.min()), 6),
        "luminance_min_positive": sig(float(pos.min()) if pos.size else 0.0, 6),
        "luminance_p1": sig(float(p1), 6),
        "luminance_p50": sig(float(p50), 6),
        "luminance_p99": sig(float(p99), 6),
        "luminance_max": sig(float(y.max()), 6),
        "stops_p99_over_p1": sig(math.log2(p99 / p1), 4) if p1 > 0 else None,
        "stops_max_over_p1": sig(math.log2(y.max() / p1), 4) if p1 > 0 else None,
        "fraction_below_0.005": sig(float((y < 0.005).mean()), 4),
        "fraction_above_100": sig(float((y > 100).mean()), 4),
        "pixels_above_100": int((y > 100).sum()),
        "fraction_above_1": sig(float((y > 1).mean()), 4),
        "fraction_sky_5_to_20": sig(float(((y >= 5) & (y <= 20)).mean()), 4),
    }


# ── Self-test ────────────────────────────────────────────────────────────────


def project_colmap(qvec, tvec, K: dict, X: Vector) -> tuple[float, float, float]:
    from mathutils import Quaternion

    r = Quaternion(qvec).to_matrix()
    xc = r @ X + Vector(tvec)
    return K["fx"] * xc.x / xc.z + K["cx"], K["fy"] * xc.y / xc.z + K["cy"], xc.z


def self_test(out: Path) -> dict:
    """Project the lamp centre with the COLMAP convention and find it as the brightest blob of the EXR."""
    cams = json.loads((out / "cameras.json").read_text())
    K = cams
    lamp = Vector(json.loads((out / "scene.json").read_text())["lighting"]["lamp_position"])
    tested = []
    for v in cams["views"]:
        if len(tested) == 2:
            break
        if not v.get("lamp_visible", True) or not (out / v["file"]).exists():
            continue
        u, vv, depth = project_colmap(v["colmap_qvec"], v["colmap_tvec"], K, lamp)
        if not (0 <= u < K["width"] and 0 <= vv < K["height"]):
            continue
        y = read_exr(out / v["file"]) @ LUM
        iy, ix = np.unravel_index(int(np.argmax(y)), y.shape)
        # The lamp's blob: pixels above half the brightest value, connected to the projection's neighbourhood.
        thr = 0.5 * float(y.max())
        win = 60
        y0, y1 = max(0, int(vv) - win), min(K["height"], int(vv) + win)
        x0, x1 = max(0, int(u) - win), min(K["width"], int(u) + win)
        sub = y[y0:y1, x0:x1]
        mask = sub > thr
        n_local, n_all = int(mask.sum()), int((y > thr).sum())
        if n_local:
            rows, cols = np.nonzero(mask)
            cu, cv = float(cols.mean() + x0 + 0.5), float(rows.mean() + y0 + 0.5)
        else:
            cu = cv = float("nan")
        err = math.hypot(cu - u, cv - vv)
        r_px_expected = K["fx"] * LAMP_RADIUS / depth
        r_px_measured = math.sqrt(n_local / math.pi) if n_local else 0.0
        peak_at_proj = float(y[min(K["height"] - 1, int(vv)), min(K["width"] - 1, int(u))])
        ok = (
            n_local > 0
            and err < 2.0
            and peak_at_proj > 100.0
            and n_local >= 0.8 * n_all  # the lamp is THE brightest region, not a reflection elsewhere
            and math.hypot(ix + 0.5 - u, iy + 0.5 - vv) < r_px_expected + 3
        )
        rec = {
            "view": v["name"], "projected_px": [round(u, 2), round(vv, 2)], "depth_m": round(depth, 4),
            "brightest_px": [int(ix), int(iy)], "brightest_value": round(float(y.max()), 2),
            "blob_centroid_px": [round(cu, 2), round(cv, 2)], "centroid_error_px": round(err, 3),
            "lamp_radius_px_expected": round(r_px_expected, 2), "lamp_radius_px_measured": round(r_px_measured, 2),
            "luminance_at_projection": round(peak_at_proj, 2), "blob_pixels_near": n_local, "blob_pixels_all": n_all,
            "ok": bool(ok),
        }
        log(
            f"self-test {v['name']}: lamp projected to ({u:.2f}, {vv:.2f}) px at {depth:.3f} m; "
            f"brightest pixel ({ix}, {iy}) = {y.max():.1f}; blob centroid ({cu:.2f}, {cv:.2f}) "
            f"err {err:.2f} px; radius {r_px_measured:.1f} px measured vs {r_px_expected:.1f} expected; "
            f"{n_local}/{n_all} bright px near projection -> {'PASS' if ok else 'FAIL'}"
        )
        tested.append(rec)
    if len(tested) < 2:
        log(f"self-test: only {len(tested)} view(s) with the lamp visible and rendered")
    passed = bool(tested) and all(t["ok"] for t in tested)
    log(f"self-test {'PASSED' if passed else 'FAILED'}")
    return {"passed": passed, "views": tested}


# ── Ground-truth files ───────────────────────────────────────────────────────


def write_colmap(out: Path, K: dict, views: list[dict]) -> None:
    d = out / "sparse_gt"
    d.mkdir(parents=True, exist_ok=True)
    (d / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 PINHOLE {K['width']} {K['height']} {K['fx']!r} {K['fy']!r} {K['cx']!r} {K['cy']!r}\n"
    )
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(views)}, mean observations per image: 0",
    ]
    for k, v in enumerate(views, start=1):
        q, t = v["colmap_qvec"], v["colmap_tvec"]
        lines.append(" ".join([str(k), *(f"{x:.12g}" for x in q), *(f"{x:.12g}" for x in t), "1", f"{v['name']}.jpg"]))
        lines.append("")
    (d / "images.txt").write_text("\n".join(lines) + "\n")
    (d / "points3D.txt").write_text(
        "# 3D point list with one line of data per point:\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "# Number of points: 0, mean track length: 0\n"
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    out = args.out.resolve()
    if args.check:
        res = self_test(out)
        return 0 if res["passed"] else 1

    rng = random.Random(args.seed)
    t_start = time.time()
    sc = bpy.context.scene
    for ob in list(bpy.data.objects):  # the factory startup's cube, camera and light
        bpy.data.objects.remove(ob, do_unlink=True)
    sc.frame_set(1)

    build_room()
    lighting = build_lights()
    objects = build_objects(rng)
    world = build_world()
    poses = camera_poses(args.views, rng)
    cams = make_cameras(poses, args.width, args.height)

    if args.preview:
        out = out / "preview"
        samples = 16
        idx = sorted({round(k * (args.views - 1) / 3) for k in range(4)})
    else:
        samples = args.samples
        idx = list(range(args.views))
    out.mkdir(parents=True, exist_ok=True)
    device = setup_render(args, samples)

    K = intrinsics(args.width, args.height)
    views = []
    for i in idx:
        cam = cams[i]
        qv, tv, _r, _t = colmap_pose(cam.matrix_world)
        c2w = cam.matrix_world
        views.append({
            "name": cam.name,
            "file": f"{cam.name}.exr",
            "blender_c2w": [[sig(c2w[r][c]) for c in range(4)] for r in range(4)],
            "colmap_qvec": [sig(x) for x in qv],
            "colmap_tvec": [sig(x) for x in tv],
            "position": [sig(x) for x in c2w.translation],
            "lamp_visible": lamp_visible(c2w.translation.copy()),
        })
    cam_doc = {
        **K,
        "conventions": {
            "blender_c2w": "camera.matrix_world, row-major; Blender camera looks down -Z, +Y up, +X right",
            "colmap": "world->camera x_c = R(qvec) X + tvec; camera looks +Z, +Y down; u = fx x/z + cx",
            "conversion": "R_colmap = diag(1,-1,-1) R_blender_w2c, t_colmap = diag(1,-1,-1) t_blender_w2c",
            "world": "Blender world, metres, +Z up; room interior x in [-4,4], y in [-3,3], z in [0,3]; y=-3 open",
        },
        "views": views,
    }
    (out / "cameras.json").write_text(json.dumps(cam_doc, indent=1))
    write_colmap(out, K, views)
    scene_doc = {
        "seed": args.seed,
        "blender_version": bpy.app.version_string,
        "lighting": lighting,
        "world": world,
        "objects": objects,
        "room": {"interior_min": [-ROOM_X, -ROOM_Y, 0.0], "interior_max": [ROOM_X, ROOM_Y, ROOM_H],
                 "open_side": "y = -3", "alcove": {"y": NICHE_Y, "z": NICHE_Z, "depth_m": NICHE_DEPTH,
                                                    "albedo": 0.04, "wall": "west (x = -4)"}},
        "render": {"engine": "CYCLES", "device": device, "samples": samples, "denoiser": "OPENIMAGEDENOISE",
                   "clamp_direct": 0.0, "clamp_indirect": 0.0, "film_exposure": 1.0,
                   "view_transform": "Standard (not baked into EXR)", "exr": "RGB float32 ZIP, scene-linear Rec.709",
                   "preview": bool(args.preview)},
    }
    (out / "scene.json").write_text(json.dumps(scene_doc, indent=1))
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "scene.blend"), compress=True)
    log(f"scene built in {time.time() - t_start:.1f} s; rendering {len(idx)} views at {samples} spp -> {out}")

    times = []
    for n, i in enumerate(idx):
        cam = cams[i]
        sc.camera = cam
        path = out / f"{cam.name}.exr"
        sc.render.filepath = str(path)
        t0 = time.time()
        bpy.ops.render.render(write_still=True)
        dt = time.time() - t0
        times.append(dt)
        if args.preview:
            rr = bpy.data.images["Render Result"]
            png = sc.render.image_settings
            png.file_format, png.color_mode, png.color_depth = "PNG", "RGB", "8"
            for ev in (0.0, -4.0, 3.0):
                sc.view_settings.exposure = ev
                rr.save_render(str(out / f"{cam.name}_ev{ev:+.0f}.png"), scene=sc)
            sc.view_settings.exposure = 0.0
            png.file_format, png.color_mode, png.color_depth = "OPEN_EXR", "RGB", "32"
            png.exr_codec = "ZIP"
        eta = (len(idx) - n - 1) * (sum(times) / len(times))
        log(f"view {n + 1}/{len(idx)} {cam.name}: {dt:.1f} s (ETA {eta:.0f} s)")

    stats = range_stats(read_exr(out / views[0]["file"]))
    scene_doc["range_stats"] = {"view": views[0]["name"], **stats}
    scene_doc["render"]["seconds_per_view"] = sig(sum(times) / len(times), 4)
    log(f"range {views[0]['name']}: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    (out / "scene.json").write_text(json.dumps(scene_doc, indent=1))
    if not args.no_check:
        res = self_test(out)
        scene_doc["self_test"] = res
        (out / "scene.json").write_text(json.dumps(scene_doc, indent=1))
        if not res["passed"]:
            return 1
    log(f"done in {time.time() - t_start:.0f} s")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    if code:
        sys.exit(code)
