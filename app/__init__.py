from __future__ import annotations

import io
import sys


class NullStream(io.TextIOBase):
    """Fallback stream for PyInstaller GUI executables where sys.stdout or sys.stderr is None."""

    def write(self, s: str) -> int:
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


if sys.stdout is None:
    sys.stdout = NullStream()
if sys.stderr is None:
    sys.stderr = NullStream()
