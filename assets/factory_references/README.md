# Factory reference stems

This directory holds the bundled, owned/royalty-free vocal reference stems used by the
blind Raw / Humanized / Reference A/B comparison (see `app/core/reference_assets.py`).

Expected files (male/female performer x dry/tuned variant -- 4 total):

- `male_dry.wav`
- `male_tuned.wav`
- `female_dry.wav`
- `female_tuned.wav`

These audio files are **not** included in the repository and must be supplied separately
(owned or royalty-free source, WAV format). `app.core.reference_assets.get_reference_stems()`
resolves whichever of the 4 files are present; `is_reference_assets_missing()` reports whether
any are absent, which drives the app's first-run fallback modal.

Users may also set a custom global override directory (`Settings.custom_reference_override_path`)
containing the same 4 filenames, which takes precedence over this directory.
