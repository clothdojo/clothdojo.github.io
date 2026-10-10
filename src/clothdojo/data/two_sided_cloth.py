"""Cloth material with the same texture and a darker reverse side."""
from pathlib import Path


def bind_two_sided_cloth(stage, back_scale=0.25):
    from pxr import Gf, Sdf, UsdShade

    if not 0 < back_scale <= 1:
        raise ValueError('back_scale must lie in (0, 1]')
    cloth = stage.GetPrimAtPath('/World/Cloth')
    material, _ = UsdShade.MaterialBindingAPI(cloth).ComputeBoundMaterial()
    path = str(material.GetPath())
    shader = UsdShade.Shader.Define(stage, path + '/TwoSidedShader')
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(str(Path(__file__).with_name('PaperTwoSided.mdl').resolve()), 'mdl')
    shader.SetSourceAssetSubIdentifier('PaperTwoSided', 'mdl')
    for node, parameter, color_space in [
        ('BaseColor', 'color_texture', 'sRGB'),
        ('Roughness', 'roughness_texture', 'raw'),
        ('Normal', 'normal_texture', 'raw'),
    ]:
        texture = UsdShade.Shader(stage.GetPrimAtPath(path + '/' + node))
        value = shader.CreateInput(parameter, Sdf.ValueTypeNames.Asset)
        value.Set(texture.GetInput('file').Get())
        value.GetAttr().SetColorSpace(color_space)
    transform = UsdShade.Shader(stage.GetPrimAtPath(path + '/UvTransform'))
    shader.CreateInput('uv_scale', Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(*transform.GetInput('scale').Get())
    )
    shader.CreateInput('back_scale', Sdf.ValueTypeNames.Float).Set(float(back_scale))
    material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(), 'out')
