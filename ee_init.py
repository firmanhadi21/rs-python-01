"""Earth Engine initialization helper for the CSK pipeline.

Mirrors the env-var contract used by ~/Github/forest-analyzer so a single
service-account key works across both projects.

Reads (in order of priority):
    GEE_SERVICE_ACCOUNT_EMAIL    - service-account email
    GEE_SERVICE_ACCOUNT_KEY_FILE - path to the JSON key
    GEE_PROJECT_ID               - GCP project id

If those env vars are unset, falls back to the values discovered in
forest-analyzer/config (key path resolved for this Mac).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import ee

log = logging.getLogger(__name__)


# Fallbacks for this machine; env vars override.
_DEFAULT_PROJECT = "ee-geodeticengineeringundip"
_DEFAULT_EMAIL   = "sci-eudr@ee-geodeticengineeringundip.iam.gserviceaccount.com"
_DEFAULT_KEY     = Path.home() / "Github/forest-analyzer/config/ee-geodetic.json"


def init_ee() -> str:
    """Initialize Earth Engine. Returns the project id used. Idempotent."""
    project = os.getenv("GEE_PROJECT_ID") or _DEFAULT_PROJECT
    email   = os.getenv("GEE_SERVICE_ACCOUNT_EMAIL") or _DEFAULT_EMAIL
    key_str = os.getenv("GEE_SERVICE_ACCOUNT_KEY_FILE")
    key     = Path(key_str).expanduser() if key_str else _DEFAULT_KEY

    if email and key.exists():
        creds = ee.ServiceAccountCredentials(email, str(key))
        ee.Initialize(credentials=creds, project=project)
        log.info("EE initialized as %s on project %s", email, project)
    else:
        if key_str and not key.exists():
            log.warning("GEE_SERVICE_ACCOUNT_KEY_FILE=%s does not exist; "
                        "falling back to user OAuth", key)
        ee.Initialize(project=project)
        log.info("EE initialized via user OAuth on project %s", project)
    return project


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    proj = init_ee()
    val = ee.Number(1).add(1).getInfo()
    print(f"EE round-trip on project {proj}: 1 + 1 = {val}")
