"""Configuration file handling for Galaxy MCP CLI."""

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


def get_config_path() -> Path:
    """Get the path to the config file."""
    return CONFIG_FILE


def load_profile(profile: str | None = None) -> GalaxyConfig:
    """
    Load Galaxy configuration from profile.

    Priority:
    1. Environment variables (GALAXY_URL, GALAXY_API_KEY)
    2. Config file profile
    3. Config file 'default' profile

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

    # Try to load from config file
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "rb") as f:
                config = tomllib.load(f)

            # Determine which profile to use
            profile_name = profile or "default"
            if profile_name in config:
                profile_config = config[profile_name]
                # Only override if not set by environment
                if not url:
                    url = profile_config.get("url")
                if not api_key:
                    api_key = profile_config.get("api_key")
        except Exception:
            # If config file is malformed, continue with env vars only
            pass

    return GalaxyConfig(url=url, api_key=api_key)


def list_profiles() -> list[str]:
    """List available profiles from config file."""
    if not CONFIG_FILE.exists():
        return []

    try:
        with open(CONFIG_FILE, "rb") as f:
            config = tomllib.load(f)
        return list(config.keys())
    except Exception:
        return []


def ensure_config_dir() -> Path:
    """Ensure the config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR
