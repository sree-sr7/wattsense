from typing import Any


def build_analysis_response(
    building_id: str,
    actual_kwh: float,
    expected_kwh: float,
    excess_kwh: float,
    cost_impact: float,
    anomalies: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Standard response format shared by:
    ML -> Backend -> Dashboard -> Report generator.
    """

    return {
        "building_id": building_id,
        "summary": {
            "actual_kwh": round(actual_kwh, 2),
            "expected_kwh": round(expected_kwh, 2),
            "excess_kwh": round(excess_kwh, 2),
            "cost_impact": round(cost_impact, 2),
        },
        "anomalies": anomalies,
        "recommendations": recommendations,
    }