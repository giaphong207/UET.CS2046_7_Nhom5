"""Repository paths shared by the relocated package, independent of the cwd."""
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPONENT_ROOT = REPOSITORY_ROOT / "research" / "lead_modeling" / "a4_a5"
PROTOCOL_PATH = COMPONENT_ROOT / "config" / "A2_protocol_v3.yaml"
AMENDMENTS_PATH = COMPONENT_ROOT / "config" / "PROTOCOL_AMENDMENTS_V3.json"
SHARED_SPLIT_ROOT = REPOSITORY_ROOT / "shared_splits"
