__version__ = "2.7.2.post1"

# Use relative import to support build-from-source installation in vLLM
from .flash_attn_interface import (
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa_version_supported,
)