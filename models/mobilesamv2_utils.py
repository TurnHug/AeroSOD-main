


import sys
import types
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOBILESAMV2_ROOT = _PROJECT_ROOT / "MobileSAM" / "MobileSAMv2"


def ensure_mobilesamv2_path():
    root = MOBILESAMV2_ROOT.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"未找到 MobileSAMv2 源码: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if "mobilesamv2" not in sys.modules:
        pkg = types.ModuleType("mobilesamv2")
        pkg.__path__ = [str(root / "mobilesamv2")]
        sys.modules["mobilesamv2"] = pkg

    return root


def import_prompt_encoder():
    ensure_mobilesamv2_path()
    from mobilesamv2.modeling.prompt_encoder import PromptEncoder
    return PromptEncoder
