"""Galaxy HTTP client policy."""

from typing import Any
from urllib.parse import urlsplit

import requests
from bioblend.galaxy import GalaxyInstance as BioBlendGalaxyInstance


def normalize_galaxy_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "?" in url
        or "#" in url
        or "\\" in url
        or any(character.isspace() or ord(character) < 32 for character in url)
    ):
        raise ValueError("Galaxy URL must be HTTP(S), without credentials, query, or fragment.")
    return url.rstrip("/") + "/"


class GalaxyInstance(BioBlendGalaxyInstance):
    """Prevent Galaxy GET requests from following redirects to other destinations."""

    def make_get_request(self, url: str, **kwargs: Any) -> requests.Response:
        kwargs["allow_redirects"] = False
        return super().make_get_request(url, **kwargs)
