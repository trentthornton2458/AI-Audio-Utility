from __future__ import annotations

import io
import os
import sys


class NullStream(io.TextIOBase):
    """Fallback stream for PyInstaller GUI executables where sys.stdout or sys.stderr is None."""

    def write(self, s: str) -> int:
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


# A windowed (console=False) PyInstaller build has no console attached, so sys.stdout/stderr
# are None from the start -- capture that here, before patching them below, since it's our only
# signal that OS-level file descriptors 0/1/2 don't exist either.
_no_console = sys.stdout is None or sys.stderr is None

if sys.stdout is None:
    sys.stdout = NullStream()
if sys.stderr is None:
    sys.stderr = NullStream()

# Patching sys.stdout/stderr above only protects pure-Python writes; it does nothing for native
# code (torch's C++ backend, MKL/OpenMP, or any dependency that fprintf's to the C runtime's
# stdout/stderr directly). In a windowed build those writes hit a nonexistent OS handle and
# crash the process with an unrecoverable access violation -- no Python exception is ever
# raised, so nothing gets logged and the crash is silent. Redirecting the raw OS descriptors to
# NUL closes that gap for native writers too. Only do this when we've confirmed there's no
# console (the check above) -- otherwise this would silently swallow real terminal output when
# running from source for debugging.
if _no_console and os.name == "nt":
    try:
        _devnull_fd = os.open(os.devnull, os.O_RDWR)
        for _std_fd in (0, 1, 2):
            try:
                os.dup2(_devnull_fd, _std_fd)
            except OSError:
                pass
    except OSError:
        pass
