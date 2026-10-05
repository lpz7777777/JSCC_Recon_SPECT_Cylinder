"""Respect the user's retirement of the v3 auxiliary fine A-field branch."""
from pathlib import Path


RETIREMENT = (Path(__file__).resolve().parent / "reports/NEMA_Body_H60/"
              "compton_response_geometry_v3/fine_field_retired.json")


def require_fine_field_active():
    # Existence is deliberately sufficient: an unreadable/invalid receipt must
    # not permit an expensive retired workflow to restart.
    if RETIREMENT.exists():
        raise RuntimeError(
            "Fine A-field production was stopped by the user on 2026-10-05; "
            "its auxiliary matrices were deleted. Do not resume this branch. "
            "See process_list_global_audit_v4/PLAN.md."
        )
