"""Compatibility shims for third-party checkpoint/config/library quirks hit when running
audio-separator and resemble-enhance together on Windows with numpy>=2.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import os
import pathlib
import sys


def pyinstaller_metadata_shim() -> None:
    """Patch importlib.metadata.version() so missing dist-info in frozen PyInstaller bundles
    returns a safe placeholder ('0.0.0') instead of raising PackageNotFoundError.

    PyInstaller's collect_all() bundles code/data/binaries but NOT .dist-info metadata.
    The spec file's copy_metadata() calls cover *known* packages, but the dependency tree
    is deep and some packages are inevitably missed. When a missing package's __init__.py
    does ``importlib.metadata.version("pkg")`` and catches PackageNotFoundError, it falls
    back to ``__version__ = "unknown"``. Downstream code (e.g. pandas' optional-dependency
    checker, onnx2torch's compatibility gate) then calls
    ``packaging.version.Version("unknown")``, which raises InvalidVersion and crashes the
    render pipeline.

    This shim is safe because:
    - '0.0.0' is a valid PEP 440 version, so packaging.version.Version('0.0.0') succeeds.
    - No code path in this app makes minimum-version decisions that would break at '0.0.0';
      the packages are already installed and functional, just missing their metadata.
    - The shim only activates inside a frozen (PyInstaller) build, never during development.
    """
    if not getattr(sys, "frozen", False):
        return

    _original_version = importlib.metadata.version

    def _safe_version(name: str) -> str:
        try:
            return _original_version(name)
        except importlib.metadata.PackageNotFoundError:
            return "0.0.0"

    importlib.metadata.version = _safe_version  # type: ignore[assignment]


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
