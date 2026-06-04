"""Configuration file handling for Galaxy MCP CLI."""

import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class GalaxyConfig(NamedTuple):
    """Galaxy connection configuration."""

    url: str | None
    api_key: str | None


CONFIG_DIR = Path.home() / ".galaxy-mcp"
CONFIG_FILE = CONFIG_DIR / "config.toml"

PLANEMO_PROFILES_DIR = Path.home() / ".planemo" / "profiles"
PLANEMO_PROFILE_OPTIONS = "planemo_profile_options.json"


def get_config_path() -> Path:
    """Get the path to the config file."""
    return CONFIG_FILE


def _load_planemo_profile(profile_name: str) -> GalaxyConfig | None:
    """
    Try to load a profile from planemo's profile directory.

    Planemo stores external Galaxy profiles as JSON at:
    ~/.planemo/profiles/<name>/planemo_profile_options.json

    Only external_galaxy engine profiles are usable (they have url + key).
    """
    profile_dir = PLANEMO_PROFILES_DIR / profile_name
    options_file = profile_dir / PLANEMO_PROFILE_OPTIONS
    if not options_file.exists():
        return None

    try:
        with open(options_file) as f:
            options = json.load(f)
    except Exception:
        return None

    # Only external Galaxy profiles have connection credentials
    if options.get("engine") != "external_galaxy":
        return None

    url = options.get("galaxy_url")
    api_key = options.get("galaxy_user_key") or options.get("galaxy_admin_key")
    if url and api_key:
        return GalaxyConfig(url=url, api_key=api_key)

    return None


def _list_planemo_profiles() -> list[str]:
    """List planemo profiles that are usable external Galaxy connections."""
    if not PLANEMO_PROFILES_DIR.exists():
        return []

    profiles = []
    try:
        for entry in PLANEMO_PROFILES_DIR.iterdir():
            if not entry.is_dir():
                continue
            options_file = entry / PLANEMO_PROFILE_OPTIONS
            if not options_file.exists():
                continue
            try:
                with open(options_file) as f:
                    options = json.load(f)
                if options.get("engine") == "external_galaxy":
                    profiles.append(entry.name)
            except Exception:
                continue
    except Exception:
        pass

    return profiles


def load_profile(profile: str | None = None) -> GalaxyConfig:
    """
    Load Galaxy configuration from profile.

    Priority:
    1. Environment variables (GALAXY_URL, GALAXY_API_KEY)
    2. gxy config file (~/.galaxy-mcp/config.toml)
    3. Planemo profiles (~/.planemo/profiles/<name>/)

    When a profile name is given explicitly, it is looked up first in the
    gxy config file, then in planemo profiles. When no profile is given,
    the 'default' entry in the gxy config file is used.

    Args:
        profile: Profile name to load from config file

    Returns:
        GalaxyConfig with url and api_key
    """
    # Start with environment variables
    url = os.environ.get("GALAXY_URL")
    api_key = os.environ.get("GALAXY_API_KEY")

    # If both are set from environment, use them
    if url and api_key:
        return GalaxyConfig(url=url, api_key=api_key)

    # Try to load from gxy config file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "rb") as f:
                config = tomllib.load(f)

            profile_name = profile or "default"
            if profile_name in config:
                profile_config = config[profile_name]
                if not url:
                    url = profile_config.get("url")
                if not api_key:
                    api_key = profile_config.get("api_key")

                if url and api_key:
                    return GalaxyConfig(url=url, api_key=api_key)
        except Exception:
            pass

    # Fall back to planemo profiles when an explicit profile name was given
    if profile:
        planemo_config = _load_planemo_profile(profile)
        if planemo_config:
            return GalaxyConfig(
                url=url or planemo_config.url,
                api_key=api_key or planemo_config.api_key,
            )

    return GalaxyConfig(url=url, api_key=api_key)


def list_profiles() -> list[str]:
    """List available profiles from gxy config and planemo."""
    profiles: list[str] = []

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "rb") as f:
                config = tomllib.load(f)
            profiles.extend(config.keys())
        except Exception:
            pass

    # Add planemo profiles that aren't already listed
    for name in _list_planemo_profiles():
        if name not in profiles:
            profiles.append(name)

    return profiles


def ensure_config_dir() -> Path:
    """Ensure the config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR
