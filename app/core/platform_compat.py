"""Compatibility shims for third-party checkpoint/config/library quirks hit when running
audio-separator and resemble-enhance together on Windows with numpy>=2.
"""

from __future__ import annotations

import contextlib
import os
import pathlib


@contextlib.contextmanager
def windows_posixpath_shim():
    """Temporarily alias pathlib.PosixPath to WindowsPath so POSIX-authored checkpoints/configs
    can be unpickled/parsed on Windows.

    Both audio-separator's BS-RoFormer .ckpt and resemble-enhance's downloaded hparams.yaml
    originate from POSIX (Linux) training/packaging environments and embed pathlib.PosixPath
    objects in their pickled/YAML state. Reconstructing a PosixPath is unconditionally
    disallowed on Windows (pathlib.PosixPath.__new__ raises NotImplementedError), regardless of
    whether the embedded path is ever actually used, so both audio_separator's torch.load() of
    the BS-RoFormer checkpoint and resemble_enhance's OmegaConf.load() of hparams.yaml (which
    registers a YAML constructor that always calls pathlib.PosixPath(...) directly, ignoring the
    host platform) can fail with `NotImplementedError: cannot instantiate 'PosixPath' on your
    system`. No-op on non-Windows platforms.
    """
    if os.name != "nt":
        yield
        return

    original = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment,misc]
    try:
        yield
    finally:
        pathlib.PosixPath = original  # type: ignore[assignment,misc]


@contextlib.contextmanager
def numpy2_fsolve_scalar_shim():
    """Make scipy.optimize.fsolve results convertible to a Python scalar under numpy>=2.

    resemble-enhance's LCFM sampler (enhancer/lcfm/cfm.py's exponential_decay_mapping) does
    `float(scipy.optimize.fsolve(...))`, relying on numpy<2's implicit conversion of a
    1-element, ndim=1 ndarray to a Python scalar. numpy>=2 (required by audio-separator's own
    pinned `numpy>=2`) raises `TypeError: only 0-dimensional arrays can be converted to Python
    scalars` for any array with ndim > 0, including size-1 arrays. Wrap scipy.optimize.fsolve so
    it returns a native Python float instead of an ndarray, sidestepping the conversion.
    Platform-independent (not Windows-specific).
    """
    import scipy.optimize

    original = scipy.optimize.fsolve

    def _patched_fsolve(*args, **kwargs):
        result = original(*args, **kwargs)
        return result.item() if hasattr(result, "item") else result

    scipy.optimize.fsolve = _patched_fsolve
    try:
        yield
    finally:
        scipy.optimize.fsolve = original


@contextlib.contextmanager
def resemble_enhance_compat_shims():
    """Combines every compat shim resemble-enhance's denoise()/enhance() calls need."""
    with windows_posixpath_shim(), numpy2_fsolve_scalar_shim():
        yield
