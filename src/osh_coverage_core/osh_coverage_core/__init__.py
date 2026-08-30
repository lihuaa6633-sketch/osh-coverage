"""Dynamic coverage planning research core.

The package intentionally has no ROS dependency.  ROS messages are converted at
the adapter boundary so algorithms and experiments remain reproducible offline.
"""

from .alignment import SE2, estimate_se2, ransac_se2
from .coverage import CoverageMonitor, DynamicRepairManager
from .grid import GridMap
from .planner import CoveragePlan, CoveragePlanner, PlannerConfig

__all__ = [
    "CoverageMonitor",
    "CoveragePlan",
    "CoveragePlanner",
    "DynamicRepairManager",
    "GridMap",
    "PlannerConfig",
    "SE2",
    "estimate_se2",
    "ransac_se2",
]

