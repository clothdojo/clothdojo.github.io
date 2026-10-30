# Third-party notices

ClothDojo original code is licensed under MIT. Third-party components retain their respective terms; the root MIT license does not replace them.

| Distributed component | Source and license | Notices retained |
| --- | --- | --- |
| Guided RTC implementation | Hugging Face LeRobot, pinned revision 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5; Apache-2.0 | src/clothdojo/rollout/guided_rtc.py attribution and third_party/lerobot/LICENSE |
| Selected XPolicyLab communication runtime | Upstream revision a768326d05f68d421542636e6467bea7ef6bbe42; Apache-2.0 | XPolicyLab/LICENSE and third_party/xpolicylab/LICENSE; model subprojects are not included |
| Converted Piper robot assets | AgileX agx_arm_urdf; MIT | third_party/agilex/LICENSE and README attribution |
| Converted example garment | GarmentCodeData v2; CC BY 4.0 | README names authors and records conversion modifications |

XPolicyLab packaging metadata is modified to include its server entry point and policy namespace in the selected runtime distribution. The runtime Python source is retained unchanged. Piper USDA and binary geometry documentation metadata is normalized for distribution; authored geometry, topology, units, numerical values and asset references are retained.

Separately installed dependencies keep their own licenses. This notice does not license external model weights or datasets. Public distribution authorization for numerical fixtures, assets and dependencies is reviewed separately; no company legal approval is claimed here.

## Project website media

`website/assets/` contains rendered ClothDojo figures and videos, not the underlying trajectory datasets or external scene-asset packages. The root MIT code license does not override third-party asset terms in rendered media. Garment and robot attribution above also applies where those assets appear in the website.

- Fabric and surface PBR materials originate from [Poly Haven](https://polyhaven.com/license), CC0.
- Selected rendered backgrounds originate from [GenieSimAssets](https://huggingface.co/datasets/agibot-world/GenieSimAssets/tree/a0813aad7c16165daffbe6c2737754e0344809ee), revision `a0813aad7c16165daffbe6c2737754e0344809ee`, [CC BY-NC-SA 4.0](https://huggingface.co/datasets/agibot-world/GenieSimAssets/blob/a0813aad7c16165daffbe6c2737754e0344809ee/LICENSE). These backgrounds are transformed by the project rendering setup.
- Other rendered environments originate from [Isaac Sim 5.0 official environment assets](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/assets/usd_assets_environments.html), subject to NVIDIA's applicable asset and material terms.

`website/assets/curated/media.json` and `website/assets/catalog/catalog.json` record the selected replay sources and visualization scope. Those source records do not make all underlying resources available under MIT.
