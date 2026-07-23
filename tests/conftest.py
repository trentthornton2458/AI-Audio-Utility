"""Global pytest configuration and mock setups for heavy ML dependencies."""

import sys
from unittest.mock import MagicMock

# Mock out heavy ML/CUDA dependencies if they are not installed in test environment
for mod_name in [
    "resemble_enhance",
    "resemble_enhance.enhancer",
    "resemble_enhance.enhancer.inference",
    "audio_separator",
    "audio_separator.separator",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()
