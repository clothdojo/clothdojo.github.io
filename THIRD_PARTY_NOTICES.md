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
