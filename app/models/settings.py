"""Settings dataclass covering current pipeline runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from app.models.preset import Preset


@dataclass
class Settings(Preset):
    """Pipeline runtime settings object consumed by RenderJob."""

    @classmethod
    def from_preset(cls, preset: Preset) -> Settings:
        """Create a Settings instance from a Preset instance."""
        return cls(
            vocal_denoise_enabled=preset.vocal_denoise_enabled,
            vocal_denoise_intensity=preset.vocal_denoise_intensity,
            vocal_enhance_enabled=preset.vocal_enhance_enabled,
            vocal_enhance_intensity=preset.vocal_enhance_intensity,
            vocal_clean_intensity=preset.vocal_clean_intensity,
            vocal_gain_db=preset.vocal_gain_db,
            instrumental_denoise_enabled=preset.instrumental_denoise_enabled,
            instrumental_denoise_intensity=preset.instrumental_denoise_intensity,
            instrumental_enhance_enabled=preset.instrumental_enhance_enabled,
            instrumental_enhance_intensity=preset.instrumental_enhance_intensity,
            instrumental_mud_cut_hz=preset.instrumental_mud_cut_hz,
            instrumental_dehiss_shelf_hz=preset.instrumental_dehiss_shelf_hz,
            instrumental_dehiss_gain_db=preset.instrumental_dehiss_gain_db,
            instrumental_gain_db=preset.instrumental_gain_db,
            notch_depth_db=preset.notch_depth_db,
            lufs_target=preset.lufs_target,
        )

    def to_preset(self) -> Preset:
        """Convert Settings instance back to a Preset instance."""
        return Preset(
            vocal_denoise_enabled=self.vocal_denoise_enabled,
            vocal_denoise_intensity=self.vocal_denoise_intensity,
            vocal_enhance_enabled=self.vocal_enhance_enabled,
            vocal_enhance_intensity=self.vocal_enhance_intensity,
            vocal_clean_intensity=self.vocal_clean_intensity,
            vocal_gain_db=self.vocal_gain_db,
            instrumental_denoise_enabled=self.instrumental_denoise_enabled,
            instrumental_denoise_intensity=self.instrumental_denoise_intensity,
            instrumental_enhance_enabled=self.instrumental_enhance_enabled,
            instrumental_enhance_intensity=self.instrumental_enhance_intensity,
            instrumental_mud_cut_hz=self.instrumental_mud_cut_hz,
            instrumental_dehiss_shelf_hz=self.instrumental_dehiss_shelf_hz,
            instrumental_dehiss_gain_db=self.instrumental_dehiss_gain_db,
            instrumental_gain_db=self.instrumental_gain_db,
            notch_depth_db=self.notch_depth_db,
            lufs_target=self.lufs_target,
        )
