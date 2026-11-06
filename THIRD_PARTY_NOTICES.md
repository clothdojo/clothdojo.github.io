# Third-party notices

ClothDojo original code is licensed under Apache-2.0. Third-party components retain their respective terms; the root code license does not replace them.

| Distributed component | Source and license | Notices retained |
| --- | --- | --- |
| Optional policy serving | [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/tree/6baccd57ff958570d0db1588370289ee36c8bb70), pinned Git submodule; Apache-2.0 | XPolicyLab/LICENSE |
| Guided RTC implementation | Hugging Face LeRobot, pinned revision 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5; Apache-2.0 | src/clothdojo/rollout/guided_rtc.py attribution and third_party/lerobot/LICENSE |
| RLinf ReCap extension patch | [RLinf](https://github.com/RLinf/RLinf/tree/7d07a4212ee6858cc333e1d4fab7a37256d1f839), pinned revision 7d07a4212ee6858cc333e1d4fab7a37256d1f839; Apache-2.0 | patches/rlinf-recap-value.patch and third_party/rlinf/LICENSE |
| Downloaded Piper assets and their rendered media | AgileX agx_arm_urdf; MIT | third_party/agilex/LICENSE and attribution below |
| Downloaded garments and their rendered media | GarmentCodeData v2; CC BY 4.0 | Authors and conversion modifications below |

Piper assets derive from AgileX's [Piper model](https://github.com/agilexrobotics/agx_arm_urdf), with its [MIT notice](third_party/agilex/LICENSE). The example garment derives from [GarmentCodeData v2](https://www.research-collection.ethz.ch/handle/20.500.11850/690432) (CC BY 4.0), by Maria Korosteleva, Timur Levent Kesdogan, Fabian Kemper, Stephan Wenninger, Jasmin Koller, Yuhan Zhang, Mario Botsch, and Olga Sorkine-Hornung. ClothDojo modified the converted assets; neither upstream group endorses these modifications.

Production simulation assets and teacher weights are downloaded separately from Hugging Face. This GitHub repository supplies the executable code and configuration. Downloaded packages retain their own component notices and provenance.

Separately installed dependencies keep their own licenses. This notice does not license external model weights or datasets. Public distribution authorization for numerical fixtures, assets and dependencies is reviewed separately; no company legal approval is claimed here.

## Project website and documentation media

`website/assets/` and `docs/assets/` contain rendered ClothDojo figures and videos, not the underlying trajectory datasets or external scene-asset packages. The root code license does not override third-party asset terms in rendered media. Garment and robot attribution above also applies where those assets appear in these media.

- Fabric and surface PBR materials originate from [Poly Haven](https://polyhaven.com/license), CC0.
- Selected rendered backgrounds originate from [GenieSimAssets](https://huggingface.co/datasets/agibot-world/GenieSimAssets/tree/a0813aad7c16165daffbe6c2737754e0344809ee), revision `a0813aad7c16165daffbe6c2737754e0344809ee`, [CC BY-NC-SA 4.0](https://huggingface.co/datasets/agibot-world/GenieSimAssets/blob/a0813aad7c16165daffbe6c2737754e0344809ee/LICENSE). These backgrounds are transformed by the project rendering setup.
- Other rendered environments originate from [Isaac Sim 5.0 official environment assets](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/assets/usd_assets_environments.html), subject to NVIDIA's applicable asset and material terms.

`website/assets/curated/media.json` and `website/assets/catalog/catalog.json` record the selected replay sources and visualization scope. Those source records do not relicense the underlying resources.

`docs/assets/nocs-flatten.json` records the Quickstart example's source revisions, checkpoint identity, video hashes, and preview speed.
