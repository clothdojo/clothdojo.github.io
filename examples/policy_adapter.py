"""Copy this file to my_policy.py and replace loading/prediction for your model.

The working reference below loads the example DP checkpoint. Run it with
--model examples.policy_adapter:Model --checkpoint PATH. No evaluator edits.
"""

from clothdojo.rollout.xpolicy import ArrayModel, canonical_observations


class Model(ArrayModel):
    # One current observation or two adjacent control frames. RGB is uint8 HWC.
    observation_horizon = 2
    action_horizon = 8
    transport_image_size = (224, 224)  # This trained DP's letterbox. Use (640,480) for native RGB.

    def __init__(self, config):
        super().__init__(config)
        # Replace with your model/processor. Load normalization from the checkpoint.
        from clothdojo.rollout.diffusion import DiffusionPolicy

        self.model = DiffusionPolicy(config["checkpoint"], device=config["device"])
        self.action_horizon = self.model.cfg["n_action_steps"]

    def reset(self):
        super().reset()
        # Clear your model's per-episode history/RNN state here.
        self.model.reset(self.seed, mode="chunk", exec_horizon=self.action_horizon)

    def predict(self, observations):
        # Replace this call. Return physical float32 [B,8,14] absolute targets:
        # left joints(6), left gripper-open(1), right joints(6), right gripper-open(1).
        # Each observation contains vision[cam_high/cam_left_wrist/cam_right_wrist],
        # state (named XPolicyLab joints), state_history, instruction and env_idx.
        return self.model.infer(canonical_observations(observations), "")
