"""Baked industrial finishes for the existing world material and image slots."""

import numpy as np


def pack_rgb(image, rgb, steps=255):
    rgb = np.round(np.clip(rgb, 0, 1) * steps) / steps
    rgba = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1))], axis=2).astype(np.float32)
    image.pixels.foreach_set(rgba.ravel())
    image.pack()


def finish_world_panels(builder):
    """Worn folds and recessed fasteners repeat quietly on per-face panel UVs."""
    for key in ("plate", "dark"):
        shader = builder.mats[key].node_tree.nodes["Principled BSDF"]
        image = shader.inputs["Base Color"].links[0].from_node.image
        width, height = image.size
        y, x = np.mgrid[:height, :width]
        y, x = y / height, x / width
        edge = np.minimum.reduce([x, y, 1 - x, 1 - y])
        recess = np.clip((0.045 - edge) / 0.045, 0, 1)
        brushed = np.sin(y * 101) * 0.004
        broad = (np.sin(x * 13) * np.sin(y * 11) + 1) * 0.016
        if builder.steam:
            base = (0.125, 0.029, 0.022) if key == "plate" else (0.071, 0.072, 0.066)
            edge_metal = (0.26, 0.185, 0.083)
        else:
            base = (0.30, 0.335, 0.36) if key == "plate" else (0.059, 0.076, 0.091)
            edge_metal = (0.36, 0.40, 0.43)
        rgb = np.broadcast_to(base, (height, width, 3)).copy()
        rgb *= (0.97 + broad + brushed - recess * 0.28)[..., None]
        seam = edge < 0.008
        rub = (edge > 0.012) & (edge < 0.021) & (np.sin(x * 29 + y * 17) > 0)
        rivet = np.zeros_like(x, dtype=bool)
        for u in (0.045, 0.955):
            for v in (0.07, 0.35, 0.65, 0.93):
                rivet |= np.hypot(x - u, y - v) < 0.006
        rgb[seam] *= 0.38
        rgb[rub | rivet] = edge_metal
        pack_rgb(image, rgb)
        rough = 0.44 + recess * 0.10 + broad * 0.25
        rough[rub | rivet] = 0.31
        pack_rgb(
            shader.inputs["Roughness"].links[0].from_node.image,
            np.repeat(rough[..., None], 3, axis=2),
            steps=47,
        )
    colors = (
        {"trim": (0.29, 0.215, 0.10), "copper": (0.19, 0.079, 0.042)}
        if builder.steam
        else {"trim": (0.18, 0.215, 0.245), "copper": (0.073, 0.045, 0.105)}
    )
    for key, color in colors.items():
        mat = builder.mats[key]
        mat.diffuse_color = (*color, 1)
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*color, 1)


def finish_world_materials(builder):
    """Separate painted shells, bare fittings, mineral host rock and tinted glass."""
    # Painted shells need a diffuse component to remain readable away from the
    # studio reflection. Bare trim keeps its existing conductive-metal response.
    for key, metallic in (("plate", 0.24 if builder.steam else 0.42), ("dark", 0.48)):
        shader = builder.mats[key].node_tree.nodes["Principled BSDF"]
        shader.inputs["Metallic"].default_value = metallic
    glass = builder.mats["window"]
    color = (0.30, 0.115, 0.027) if builder.steam else (0.025, 0.27, 0.39)
    shader = glass.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Alpha"].default_value = 0.32
    shader.inputs["Roughness"].default_value = 0.16
    glass.diffuse_color = (*color, 0.32)
