# ClothDojo

### Learned Data Generation and Benchmarking for Bimanual Garment Flattening and Folding

Zilong Huang, Zipeng Ye, Caicheng Wang, Yuchen Xie, Jiaxiang Luo

[Website](https://clothdojo.github.io/) · [Paper](https://arxiv.org/abs/2611.01190) · [Assets](https://huggingface.co/datasets/clothdojo/assets)

ClothDojo is a simulation platform for dynamic interaction between bimanual robots and deformable garments. Built on [Newton](https://github.com/newton-physics/newton), it supports learned data generation, policy evaluation on fixed task conditions, and human intervention through VR demonstrations, rollout corrections, and trajectory review. NOCS teachers are provided for flattening and folding.

## NOCS Quickstart

Follow the [Quickstart](docs/usage.md) to generate a trajectory and watch its high and low camera videos on one NVIDIA GPU. Isaac Sim is not required.

[![NOCS: cam_high left, cam_low right; 4× preview](docs/assets/nocs-flatten-paired-preview.gif)](docs/assets/nocs-flatten-paired.mp4)

## Randomized RGB rendering

Follow the [RGB rendering guide](docs/randomized-rendering.md) to render the saved motion with different materials and lighting. Add Isaac Sim 5 to the same environment with `setup.sh --rgb`; rendering requires an RTX GPU.

[![RGB: cam_high left, cam_low right; 4× preview](docs/assets/randomized-rgb-paired-preview.gif)](docs/assets/randomized-rgb-paired.mp4)

## Resources

- [Assets](https://huggingface.co/datasets/clothdojo/assets) · [NOCS teachers](https://huggingface.co/clothdojo/nocs-teachers) · [Download details](docs/data-and-assets.md)
- [More workflows](docs/advanced.md): successful-trajectory selection, evaluation, and RGB export
- [Questions and bug reports](https://github.com/clothdojo/clothdojo.github.io/issues)

## Acknowledgements

ClothDojo builds on the work of the following projects and asset creators. We thank their authors and contributors:

- **Simulation:** [Newton](https://github.com/newton-physics/newton), [NVIDIA Warp](https://github.com/NVIDIA/warp), [MuJoCo](https://github.com/google-deepmind/mujoco), and [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp). Our environment uses a pinned Newton 1.5.0 fork; see the [source and version details](THIRD_PARTY_NOTICES.md#simulation).
- **Garments:** [GarmentCodeData](https://www.research-collection.ethz.ch/handle/20.500.11850/690432), whose garment assets we select and convert for simulation, under CC BY 4.0.
- **Textures and scenes:** [Poly Haven](https://polyhaven.com/) provides fabric and surface textures (CC0). Optional RGB scene and material sources include [GenieSimAssets](https://huggingface.co/datasets/agibot-world/GenieSimAssets) and [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/index.html), under their respective asset terms.
- **Robot assets:** [AgileX](https://github.com/agilexrobotics/agx_arm_urdf), [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo), [Franka Robotics](https://franka.de/3d-assets), and the component authors credited in the [robot provenance notes](THIRD_PARTY_NOTICES.md#robot-assets).
- **Policy and data tooling:** [OpenPI](https://github.com/Physical-Intelligence/openpi), [RLinf](https://github.com/RLinf/RLinf), and [Hugging Face LeRobot](https://github.com/huggingface/lerobot).

See [third-party notices](THIRD_PARTY_NOTICES.md) for authors, pinned sources, modifications, and component license information. These acknowledgements do not replace the original license and attribution files supplied with code, assets, or model weights.

## Citation

If you use ClothDojo in your research, please cite:

```bibtex
@misc{huang2026clothdojo,
  title = {{ClothDojo}: Learned Data Generation and Benchmarking for Bimanual Garment Flattening and Folding},
  author = {Huang, Zilong and Ye, Zipeng and Wang, Caicheng and Xie, Yuchen and Luo, Jiaxiang},
  year = {2026},
  eprint = {2611.01190},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2611.01190}
}
```

## License

The code is released under [Apache-2.0](LICENSE). Downloaded data, weights, and third-party assets retain their respective licenses; see [attribution and third-party notices](THIRD_PARTY_NOTICES.md).
