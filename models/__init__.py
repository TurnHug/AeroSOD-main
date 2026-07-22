"""AeroSOD model components."""

from models.mfsa import (
    LiteMFSAState,
    MultiFactorSceneAdapter,
    TargetGuidedSCConv,
)
from models.mobilesam_encoder import BackboneOutput, MobileSAMv2Backbone
from models.object_aware_prompt_branch import (
    MobileSAMv2ObjectAwarePromptEncoder,
    ObjectAwarePromptOutput,
)
from models.s3qd_decoder import S3QD

__all__ = [
    "AeroSOD",
    "BackboneOutput",
    "LiteMFSAState",
    "MobileSAMv2Backbone",
    "MobileSAMv2ObjectAwarePromptEncoder",
    "MultiFactorSceneAdapter",
    "ObjectAwarePromptOutput",
    "S3QD",
    "TargetGuidedSCConv",
]


def __getattr__(name):
    if name == "AeroSOD":
        from models.aero_sod import AeroSOD
        return AeroSOD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
