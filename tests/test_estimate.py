import sys
sys.path.insert(0, "src")
from estimate import HARDWARE, estimate_workload

def test_estimate_has_required_outputs():
    result = estimate_workload(14.5, 1, 128, HARDWARE[0])
    assert result["predicted_latency_ms"] > 0
    assert result["predicted_throughput"] > 0
    assert "fits_memory" in result

def test_hardware_options_are_comparable():
    rows = [estimate_workload(14.5, 1, 128, h) for h in HARDWARE]
    assert len(rows) >= 3
    assert all("estimated_hourly_cost" in row for row in rows)
