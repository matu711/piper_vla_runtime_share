"""ROS wire profiles for generated runtime contracts, independent of model IO."""
from dataclasses import dataclass

from piper_vla_dual_runtime.core import POLICY_CONTRACT, DELTA_MASK


@dataclass(frozen=True)
class DualProfile:
    name: str = "dual_pi05_joint_v1"
    pi05: bool = True
    normalization: str = "quantile"
    delta_joints: bool = True

    @property
    def contract(self):
        return dict(POLICY_CONTRACT, normalization=self.normalization)

    @property
    def training_actions(self):
        return dict(joint_delta_indices=[i for i, flag in enumerate(DELTA_MASK) if flag] if self.delta_joints else [],
                    absolute_gripper_indices=[6, 13],
                    delta_reference="request_state" if self.delta_joints else None,
                    output_transforms=["Unnormalize"] + (["AbsoluteActions"] if self.delta_joints else []) + ["truncate_to_14"])


def profile_from_manifest(manifest):
    name = manifest["compatibility_profile"]
    family = manifest["model_family"]
    if name == "dual_pi05_joint_v1":
        if family != "pi05":
            raise ValueError("pi0 requires an explicit dual_pi0_joint_v1 compatibility profile")
        return DualProfile()
    if name != "dual_pi0_joint_v1" or family != "pi0":
        raise ValueError("unsupported dual model/profile")
    normalization = manifest["contract"]["normalization"]
    if normalization not in ("standard", "quantile"):
        raise ValueError("pi0 requires explicit supported training normalization; no mixed normalization")
    actions = manifest["training_actions"]
    for delta in (True, False):
        profile = DualProfile(name, False, normalization, delta)
        if actions == profile.training_actions:
            return profile
    raise ValueError("unsupported pi0 training action contract")
