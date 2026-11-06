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

## Simulation

The ClothDojo simulation environment is built on [Newton](https://github.com/newton-physics/newton), by the Newton contributors. Newton was initiated by Disney Research, Google DeepMind, and NVIDIA. It builds on [NVIDIA Warp](https://github.com/NVIDIA/warp) and integrates [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp). ClothDojo also uses [MuJoCo](https://github.com/google-deepmind/mujoco) for robot modeling and retargeting.

`requirements.txt` pins the project [Newton 1.5.0 fork](https://github.com/zlong-911/newton/tree/363aab40133fa72d8536ec5e15802a466b1f6601), including the selected Style3D cloth-solver fixes. The pinned Newton source is [Apache-2.0 licensed](https://github.com/zlong-911/newton/blob/363aab40133fa72d8536ec5e15802a466b1f6601/LICENSE.md). Newton's own bundled components retain their separate notices. For scholarly attribution, use the upstream [Newton citation metadata](https://github.com/newton-physics/newton/blob/main/CITATION.cff).

## Robot assets

ClothDojo assembles and adapts third-party robot models into simulation workstations. Source geometry and component terms remain those of their authors; a workstation conversion does not make them ClothDojo originals.

| Used component | Upstream source and adaptation |
| --- | --- |
| Piper | AgileX [Piper models](https://github.com/agilexrobotics/agx_arm_urdf); MIT notice retained in [third_party/agilex/LICENSE](third_party/agilex/LICENSE). Converted and assembled into the ClothDojo workstation. |
| Piper workstation mounting structure | C2254 fixtures, columns, and central camera geometry originate from manufacturer-supplied STEP files converted by ClothDojo. Separate public redistribution terms have not been independently confirmed; AgileX's MIT notice does not cover this separately supplied CAD. |
| ALOHA 2, Panda, UR5e, and Robotiq 2F-85 models | [MuJoCo Menagerie, pinned revision](https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994). Model directories carry their own licenses; the collection is not treated as having one uniform model license. ClothDojo adapts workstations, camera placement, and simulation drives. |
| X5A robot and camera assemblies | [RoboDojo assets, pinned revision](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/tree/cfb06d1dadcdf03ccc923102b0e3f3b0a68dfc43/Assets/Robots/x5), with selected limits from [Stanford arx5-sdk](https://github.com/real-stanford/arx5-sdk/tree/a8890c9bae94464abd1cb7c5e4da7c4a62104a3a). The workstation also uses the integrated mounting plate from [OpenDriveLab/kai0](https://github.com/OpenDriveLab/kai0). |
| FR3 Duo | [Franka official 3D assets](https://franka.de/3d-assets), with dynamics from [franka_description](https://github.com/frankarobotics/franka_description/tree/02afaae282d4a8e10d7d2f781b23b3515c303ce5) and a Menagerie Robotiq gripper. Manufacturer CAD is not automatically covered by a robot software repository's license. |
| Camera meshes and adapters | [RealSense ROS](https://github.com/realsenseai/realsense-ros) and [PickNik Robotiq descriptions](https://github.com/PickNikRobotics/ros2_robotiq_gripper/tree/2c047340aeb2440f7a60e429264221aab9658707), where used in workstation assemblies. Preserve the corresponding component notices. |

The separately downloaded [asset package](https://huggingface.co/datasets/clothdojo/assets) provides component licenses and provenance. Robot `CLOTHDOJO.md` files and `provenance/` records describe the conversions and assembly changes. This source list acknowledges usage; it does not grant additional redistribution rights or imply endorsement by the upstream authors.

## Project website and documentation media

`website/assets/` and `docs/assets/` contain rendered ClothDojo figures and videos, not the underlying trajectory datasets or external scene-asset packages. The root code license does not override third-party asset terms in rendered media. Garment and robot attribution above also applies where those assets appear in these media.

- Fabric and surface PBR materials originate from [Poly Haven](https://polyhaven.com/license), CC0.
- Selected rendered backgrounds originate from [GenieSimAssets](https://huggingface.co/datasets/agibot-world/GenieSimAssets/tree/a0813aad7c16165daffbe6c2737754e0344809ee), revision `a0813aad7c16165daffbe6c2737754e0344809ee`, [CC BY-NC-SA 4.0](https://huggingface.co/datasets/agibot-world/GenieSimAssets/blob/a0813aad7c16165daffbe6c2737754e0344809ee/LICENSE). These backgrounds are transformed by the project rendering setup.
- Other rendered environments originate from [Isaac Sim 5.0 official environment assets](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/assets/usd_assets_environments.html), subject to NVIDIA's applicable asset and material terms.

The [material catalog](configs/rendering/materials.json) records selected Poly Haven asset IDs and source file hashes; individual asset pages are available at `https://polyhaven.com/a/<asset-id>`. The [scene catalog](configs/rendering/benchmark.json) and [extracted material catalog](configs/rendering/scene_materials.json) record the Isaac and GenieSim source versions, selected assets, and material terms. ClothDojo's visual adaptations include material reassignment, UV mapping, texture cropping, scene placement, and rendering. These records distinguish upstream content from project adaptations.

`website/assets/curated/media.json` and `website/assets/catalog/catalog.json` record the selected replay sources and visualization scope. Those source records do not relicense the underlying resources.

`docs/assets/nocs-flatten.json` records the Quickstart example's source revisions, checkpoint identity, video hashes, and preview speed.
