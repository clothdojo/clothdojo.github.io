# Third-party notices

ClothDojo's original code is MIT licensed. Third-party code and assets retain their own licenses.

- `assets/robots/piper/`: converted from AgileX's [Piper models](https://github.com/agilexrobotics/agx_arm_urdf); the MIT license is retained in `agilex/LICENSE`.
- `src/clothdojo/rollout/guided_rtc.py`: adapted from Hugging Face LeRobot's guided RTC implementation, with project-specific sampling and rebasing. Its original copyright attribution is retained in the file. The pinned [LeRobot license and upstream notices](https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/LICENSE) are retained in `lerobot/LICENSE`.
- `XPolicyLab/`: a selected communication runtime sourced from upstream commit `a768326d05f68d421542636e6467bea7ef6bbe42`; its Apache-2.0 license remains in that directory and in `xpolicylab/LICENSE`. Packaging metadata is modified to distribute the server entry point and namespace without upstream model subprojects; runtime source files are retained unchanged.
- The example garment derives from GarmentCodeData v2 under CC BY 4.0. Source, authors, and modifications are acknowledged in the repository README.

Dependencies installed separately keep their respective licenses. These notices do not relicense model weights, datasets, or external visual assets.
