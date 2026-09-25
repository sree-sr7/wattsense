from typing import Any

from .schemas import build_analysis_response


def run_analysis() -> dict[str, Any]:
    """
    Temporary mock analysis.

    This allows the dashboard/backend team to work before
    the real ML pipeline is connected.
    """

    anomalies = [
        {
            "id": "ANOM-001",
            "type": "after_hours",
            "severity": "high",
            "occurrences": 17,
            "excess_kwh": 3900.0,
            "description": (
                "Repeated overnight consumption above the "
                "building-specific expected baseline."
            ),
        },
        {
            "id": "ANOM-002",
            "type": "weekend",
            "severity": "medium",
            "occurrences": 8,
            "excess_kwh": 2600.0,
            "description": (
                "Weekend consumption is unusually similar "
                "to normal weekday operation."
            ),
        },
    ]

    recommendations = [
        {
            "rank": 1,
            "action": "Review HVAC operating schedule",
            "reason": "Repeated after-hours excess consumption",
            "annual_savings": 468000.0,
            "implementation_cost": 25000.0,
            "payback_months": 0.64,
            "confidence": "high",
        },
        {
            "rank": 2,
            "action": "Review weekend operating schedule",
            "reason": "Persistent weekend load",
            "annual_savings": 312000.0,
            "implementation_cost": 30000.0,
            "payback_months": 1.15,
            "confidence": "medium",
        },
    ]

    return build_analysis_response(
        building_id="lecture_building",
        actual_kwh=84200.0,
        expected_kwh=77350.0,
        excess_kwh=6850.0,
        cost_impact=65075.0,
        anomalies=anomalies,
        recommendations=recommendations,
    )