"""Shared camera calibration and authored robot appearance."""

from __future__ import annotations

from dataclasses import dataclass

CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DATASET_CAMERA_NAMES = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")
# Keep rollout rasterization aligned with the existing paired benchmark. OpenPI
# downsamples these views to the square transport size below; a lower native
# render size is faster but is not pixel-equivalent and therefore needs its own
# paired rollout validation before it can replace this contract.
POLICY_RENDER_SIZE = (640, 480)
POLICY_TRANSPORT_SIZE = (224, 224)


@dataclass(frozen=True, slots=True)
class CameraSpec:
    name: str
    parent_body: str | None
    parent_from_camera: tuple[tuple[float, float, float, float], ...]
    focal_length: float
    horizontal_aperture: float
    vertical_aperture: float
    near_clip: float = 0.0


def _apply_authored_appearance(builder, stage) -> dict:
    """Bridge Newton USD appearance omissions for both review and policy RGB."""
    from newton import ShapeFlags
    from newton._src.usd.utils import (
        _color_to_display_space,
        resolve_material_properties_for_prim,
    )
    from pxr import UsdGeom

    display_colors, hidden_proxies = [], []
    for index, path in enumerate(builder.shape_label):
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.proxy:
            if builder.shape_flags[index] & int(ShapeFlags.VISIBLE):
                hidden_proxies.append(path)
            builder.shape_flags[index] &= ~int(ShapeFlags.VISIBLE)
        # Newton 1.5 resolves bound materials but omits uniform displayColor.
        # Only bridge this omission in memory; never replace a bound material.
        properties = resolve_material_properties_for_prim(prim)
        if properties.get("color") is not None or properties.get("texture") is not None:
            continue
        attribute = prim.GetAttribute("primvars:displayColor")
        values = attribute.Get() if attribute and attribute.HasAuthoredValue() else None
        if values is not None and len(values) == 1:
            builder.shape_color[index] = _color_to_display_space(
                tuple(map(float, values[0])), attribute
            )
            display_colors.append(path)
    return {
        "authored_display_color_fallbacks": display_colors,
        "hidden_authored_proxy_shapes": hidden_proxies,
    }


def policy_observation_settings(
    *, cloth_visual: str, visual_profile: str | None = None,
    visual_profiles: tuple[str, ...] | None = None, visual_seed: int = 0,
) -> dict[str, object]:
    """Describe the visual contract using the scene's explicit cloth appearance."""
    if cloth_visual == "rgb":
        from clothdojo.assets.visual_benchmark import VISUAL_PROFILES, visual_protocol_identity

        if visual_profiles is not None:
            if visual_profile is not None:
                raise ValueError("use visual_profile or visual_profiles, not both")
            if not visual_profiles or len(set(visual_profiles)) != len(visual_profiles):
                raise ValueError("visual_profiles must be non-empty and unique")
            if any(profile not in VISUAL_PROFILES for profile in visual_profiles):
                raise ValueError("RGB observations contain an unknown visual profile")
        elif visual_profile not in VISUAL_PROFILES:
            raise ValueError("RGB observations require a visual_profile")
        if type(visual_seed) is not int or visual_seed < 0:
            raise ValueError("visual_seed must be a non-negative integer")
        return {
            "renderer": "isaac_rtx",
            "camera_projection": "newton_compatible_v1",
            "cloth_backface_scale": 0.25,
            "camera_names": list(CAMERA_NAMES),
            "render_width_height": list(POLICY_RENDER_SIZE),
            "transport_width_height": list(POLICY_TRANSPORT_SIZE),
            "cloth_visual": cloth_visual,
            **({"visual_profiles": list(visual_profiles)} if visual_profiles is not None
               else {"visual_profile": visual_profile}),
            "visual_seed": visual_seed,
            **visual_protocol_identity(),
        }
    if cloth_visual not in ("nocs", "white"):
        raise ValueError("cloth_visual must be 'nocs', 'white' or 'rgb'")
    if visual_profile is not None or visual_profiles is not None or visual_seed != 0:
        raise ValueError("visual_profile(s)/visual_seed require RGB observations")
    return {
        "renderer": "newton-sensor-tiled-camera",
        "camera_names": list(CAMERA_NAMES),
        "render_width_height": list(POLICY_RENDER_SIZE),
        "transport_width_height": list(POLICY_TRANSPORT_SIZE),
        "cloth_visual": cloth_visual,
    }
