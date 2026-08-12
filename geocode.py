"""
Reverse geocoding via OpenStreetMap Nominatim (public instance).

Policy: https://operations.osmfoundation.org/policies/nominatim/
Use only low-frequency server-side requests and a valid identifying User-Agent.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

NOMINATIM_REVERSE = "https://nominatim.openstreetmap.org/reverse"


def reverse_geocode_osm(lat: float, lon: float, timeout: int = 12) -> dict[str, Any]:
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1,
    }
    url = f"{NOMINATIM_REVERSE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "crop-rec-education/1.0 (local academic project; no bulk use)",
            "Accept-Language": "en",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())
