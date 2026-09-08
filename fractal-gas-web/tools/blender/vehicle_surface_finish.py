"""Final vehicle-only finish using existing packed maps and material slots.

Every polygon projects the complete panel UV square. Keep wear near its border
and avoid pictorial markings that would repeat on small mechanical fittings.
"""

import numpy as np


def finish_vehicle_surfaces(builder):
    """Separate painted armor, rubbed metal and recesses without runtime cost."""
    for key in ("plate", "dark"):
        shader = builder.mats[key].node_tree.nodes.get("Principled BSDF")
        color_image = shader.inputs["Base Color"].links[0].from_node.image
        width, height = color_image.size
        y, x = np.mgrid[0:height, 0:width]
        y, x = y / height, x / width
        edge = np.minimum.reduce([x, 1 - x, y, 1 - y])
        border = np.clip((0.045 - edge) / 0.045, 0, 1)
        brushed = (np.sin(y * 83) + np.sin(y * 137)) * 0.003
        field = (np.sin(x * 9) * np.sin(y * 7) + 1) * 0.012
        if builder.steam:
            base = (0.125, 0.027, 0.022) if key == "plate" else (0.072, 0.075, 0.074)
            metal = (0.27, 0.19, 0.085)
        else:
            base = (0.29, 0.325, 0.355) if key == "plate" else (0.062, 0.078, 0.094)
            metal = (0.38, 0.415, 0.44)
        rgb = np.broadcast_to(base, (height, width, 3)).copy()
        rgb *= (0.98 + field + brushed - border * 0.24)[..., None]
        # Folded seam and a narrow broken rubbed edge remain readable on armor.
        seam = edge < 0.007
        rub = (edge > 0.009) & (edge < 0.017) & (np.sin(x * 27 + y * 19) > -0.25)
        rgb[seam] *= 0.48
        rgb[rub] = metal
        _pack(color_image, np.round(rgb * 255) / 255)
        rough = 0.43 + border * 0.09 + field * 0.4 + brushed
        rough[rub] = 0.30
        _pack(
            shader.inputs["Roughness"].links[0].from_node.image,
            np.repeat((np.round(rough * 47) / 47)[..., None], 3, axis=2),
        )

    colors = (
        {"trim": (0.28, 0.205, 0.095), "copper": (0.19, 0.078, 0.039)}
        if builder.steam
        else {"trim": (0.18, 0.215, 0.25), "copper": (0.065, 0.044, 0.088)}
    )
    colors["glass"] = (0.18, 0.074, 0.018) if builder.steam else (0.018, 0.12, 0.17)
    if builder.kind == "drone":
        colors["gold"] = (0.43, 0.265, 0.065)
    elif builder.kind == "harvester" and builder.steam:
        colors["gold"] = (0.32, 0.20, 0.065)
    if builder.kind == "harvester" and not builder.steam:
        colors["energy"] = (0.24, 0.035, 0.58)
        shader = builder.mats["energy"].node_tree.nodes.get("Principled BSDF")
        shader.inputs["Emission Color"].default_value = (*colors["energy"], 1)
        shader.inputs["Emission Strength"].default_value = 0.28
    for key, color in colors.items():
        material = builder.mats[key]
        material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (
            *color,
            1,
        )
        material.diffuse_color = (*color, 1)


def _pack(image, rgb):
    rgba = np.concatenate([rgb, np.ones((*rgb.shape[:2], 1))], axis=2).astype(np.float32)
    image.pixels.foreach_set(rgba.ravel())
    image.pack()
