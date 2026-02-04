"""
Battery optimization engine using Linear Programming.

Solves the optimal battery charge/discharge schedule to minimize electricity costs
or maximize profit/self-consumption based on price forecasts and solar predictions.

Power Flow Model:
                        ┌─────────────┐
    solar_to_load ─────►│             │◄───── grid_to_load
    solar_to_battery ──►│   HOME      │
    battery_to_load ───►│   LOAD      │
                        └─────────────┘

                        ┌─────────────┐
    solar_to_grid ─────►│             │
    battery_to_grid ───►│   GRID      │ (export)
                        └─────────────┘

                        ┌─────────────┐
    solar_to_battery ──►│             │
    grid_to_battery ───►│  BATTERY    │
                        └─────────────┘

Key terminology:
- consume: Battery power used to cover home load (battery_to_load)
- export: Power sent to grid (solar_to_grid + battery_to_grid)
- charge: Power going into battery (solar_to_battery + grid_to_battery)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

# Optional dependency - optimization won't work without it
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)


class CostFunction(Enum):
    """Optimization objective functions."""
    COST_MINIMIZATION = "cost"           # Minimize total electricity cost
    PROFIT_MAXIMIZATION = "profit"       # Maximize profit from grid trading
    SELF_CONSUMPTION = "self_consumption"  # Maximize solar self-consumption


@dataclass
class OptimizationConfig:
    """Configuration for the optimization problem."""
    # Battery parameters
    battery_capacity_wh: float = 13500.0  # Total battery capacity in Wh
    max_charge_w: float = 5000.0          # Maximum charge power in W
    max_discharge_w: float = 5000.0       # Maximum discharge power in W
    charge_efficiency: float = 0.90       # Round-trip efficiency for charging
    discharge_efficiency: float = 0.90    # Round-trip efficiency for discharging

    # Constraints
    backup_reserve: float = 0.20          # Minimum SOC to maintain (0-1)
    target_end_soc: float | None = None   # Optional target SOC at end of horizon
    min_soc: float = 0.0                  # Minimum allowed SOC (0-1)
    max_soc: float = 1.0                  # Maximum allowed SOC (0-1)

    # Grid constraints
    max_grid_import_w: float | None = None   # Max grid import power (None = unlimited)
    max_grid_export_w: float | None = None   # Max grid export power (None = unlimited)

    # Optimization settings
    cost_function: CostFunction = CostFunction.COST_MINIMIZATION
    interval_minutes: int = 5             # Time interval in minutes
    horizon_hours: int = 48               # Optimization horizon in hours

    # Degradation penalty (optional)
    cycle_cost: float = 0.0               # Cost per kWh cycled (for battery wear)


@dataclass
class OptimizationResult:
    """Result from the optimization solver."""
    success: bool
    status: str

    # Schedules (per interval) - legacy names for backward compatibility
    charge_schedule_w: list[float] = field(default_factory=list)    # Total charge power (W)
    discharge_schedule_w: list[float] = field(default_factory=list) # Total discharge power (W)
    grid_import_w: list[float] = field(default_factory=list)        # Total grid import (W)
    grid_export_w: list[float] = field(default_factory=list)        # Total grid export (W)
    soc_trajectory: list[float] = field(default_factory=list)       # SOC at each interval (0-1)

    # New detailed breakdown - clearer terminology
    battery_consume_w: list[float] = field(default_factory=list)    # Battery → Load (W)
    battery_export_w: list[float] = field(default_factory=list)     # Battery → Grid (W)
    solar_to_load_w: list[float] = field(default_factory=list)      # Solar → Load (W)
    solar_to_battery_w: list[float] = field(default_factory=list)   # Solar → Battery (W)
    solar_to_grid_w: list[float] = field(default_factory=list)      # Solar → Grid (W)
    grid_to_load_w: list[float] = field(default_factory=list)       # Grid → Load (W)
    grid_to_battery_w: list[float] = field(default_factory=list)    # Grid → Battery (W)

    # Timestamps for each interval
    timestamps: list[datetime] = field(default_factory=list)

    # Summary metrics
    total_cost: float = 0.0               # Total electricity cost ($)
    total_import_kwh: float = 0.0         # Total grid import (kWh)
    total_export_kwh: float = 0.0         # Total grid export (kWh)
    total_charge_kwh: float = 0.0         # Total battery charge (kWh)
    total_discharge_kwh: float = 0.0      # Total battery discharge (kWh)
    average_import_price: float = 0.0     # Average import price ($/kWh)
    average_export_price: float = 0.0     # Average export price ($/kWh)

    # New detailed metrics
    total_battery_consume_kwh: float = 0.0  # Battery used to power home (kWh)
    total_battery_export_kwh: float = 0.0   # Battery exported to grid (kWh)
    total_solar_consumed_kwh: float = 0.0   # Solar used (load + battery) (kWh)
    total_solar_exported_kwh: float = 0.0   # Solar exported to grid (kWh)

    # Comparison metrics
    baseline_cost: float = 0.0            # Cost without optimization
    savings: float = 0.0                  # Savings vs baseline ($)

    # Solver info
    solve_time_ms: float = 0.0
    solver_name: str = ""

    def get_action_at_index(self, index: int) -> dict[str, Any]:
        """Get the recommended action for a specific interval."""
        if index < 0 or index >= len(self.charge_schedule_w):
            return {"action": "idle", "power_w": 0}

        charge_w = self.charge_schedule_w[index]
        discharge_w = self.discharge_schedule_w[index]

        # Get detailed breakdown if available
        consume_w = self.battery_consume_w[index] if index < len(self.battery_consume_w) else 0
        export_w = self.battery_export_w[index] if index < len(self.battery_export_w) else 0

        if charge_w > 10:  # Small threshold to avoid floating point noise
            return {"action": "charge", "power_w": charge_w}
        elif discharge_w > 10:
            # Determine if primarily consuming or exporting
            if consume_w > export_w:
                return {"action": "consume", "power_w": discharge_w, "to_load_w": consume_w, "to_grid_w": export_w}
            else:
                return {"action": "export", "power_w": discharge_w, "to_load_w": consume_w, "to_grid_w": export_w}
        else:
            return {"action": "idle", "power_w": 0}

    def get_next_actions(self, count: int = 5) -> list[dict[str, Any]]:
        """Get the next N actions from the schedule, starting from current time."""
        from datetime import datetime, timezone

        actions = []
        now = datetime.now(timezone.utc)

        # Find the first index that is in the future (or current interval)
        start_index = 0
        for i, ts in enumerate(self.timestamps):
            # Make timestamp timezone-aware if needed
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= now:
                start_index = i
                break
            # If this is the last interval and it's still in the past,
            # start from the beginning (schedule is stale)
            if i == len(self.timestamps) - 1:
                start_index = 0

        # Collect actions from start_index
        for i in range(start_index, min(start_index + count, len(self.charge_schedule_w))):
            action = self.get_action_at_index(i)
            action["timestamp"] = self.timestamps[i].isoformat() if i < len(self.timestamps) else None
            action["soc"] = self.soc_trajectory[i + 1] if (i + 1) < len(self.soc_trajectory) else None
            actions.append(action)
        return actions

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "status": self.status,
            "schedule": {
                # Legacy fields (backward compatible)
                "charge_w": self.charge_schedule_w,
                "discharge_w": self.discharge_schedule_w,
                "grid_import_w": self.grid_import_w,
                "grid_export_w": self.grid_export_w,
                "soc_trajectory": self.soc_trajectory,
                "timestamps": [t.isoformat() for t in self.timestamps],
                # New detailed breakdown
                "battery_consume_w": self.battery_consume_w,
                "battery_export_w": self.battery_export_w,
                "solar_to_load_w": self.solar_to_load_w,
                "solar_to_battery_w": self.solar_to_battery_w,
                "solar_to_grid_w": self.solar_to_grid_w,
                "grid_to_load_w": self.grid_to_load_w,
                "grid_to_battery_w": self.grid_to_battery_w,
            },
            "summary": {
                "total_cost": round(self.total_cost, 2),
                "total_import_kwh": round(self.total_import_kwh, 2),
                "total_export_kwh": round(self.total_export_kwh, 2),
                "total_charge_kwh": round(self.total_charge_kwh, 2),
                "total_discharge_kwh": round(self.total_discharge_kwh, 2),
                "average_import_price": round(self.average_import_price, 4),
                "average_export_price": round(self.average_export_price, 4),
                "baseline_cost": round(self.baseline_cost, 2),
                "savings": round(self.savings, 2),
                # New detailed metrics
                "total_battery_consume_kwh": round(self.total_battery_consume_kwh, 2),
                "total_battery_export_kwh": round(self.total_battery_export_kwh, 2),
                "total_solar_consumed_kwh": round(self.total_solar_consumed_kwh, 2),
                "total_solar_exported_kwh": round(self.total_solar_exported_kwh, 2),
            },
            "solver": {
                "solve_time_ms": round(self.solve_time_ms, 2),
                "solver_name": self.solver_name,
            },
            "next_actions": self.get_next_actions(5),
        }


class BatteryOptimiser:
    """
    Linear Programming optimiser for battery scheduling.

    Uses CVXPY with HiGHS solver to find the optimal charge/discharge schedule
    that minimizes electricity costs while respecting battery constraints.

    The model explicitly tracks power flows:
    - Solar can go to: load, battery, or grid
    - Battery can go to: load (consume) or grid (export)
    - Grid import can go to: load or battery
    """

    def __init__(self, config: OptimizationConfig | None = None):
        """Initialize the optimiser with configuration."""
        self.config = config or OptimizationConfig()
        self._solver_available = self._check_solver()

    def _check_solver(self) -> bool:
        """Check if CVXPY, numpy, and HiGHS solver are available."""
        if not NUMPY_AVAILABLE:
            _LOGGER.warning("NumPy not installed - optimization disabled")
            return False

        try:
            import cvxpy as cp
            if "HIGHS" in cp.installed_solvers():
                _LOGGER.info("CVXPY with HiGHS solver available")
                return True
            available = cp.installed_solvers()
            _LOGGER.warning(f"HiGHS not available, using fallback. Available: {available}")
            return len(available) > 0
        except ImportError:
            _LOGGER.warning("CVXPY not installed - optimization disabled")
            return False

    @property
    def is_available(self) -> bool:
        """Check if optimization is available."""
        return self._solver_available

    def optimize(
        self,
        prices_import: list[float],
        prices_export: list[float],
        solar_forecast: list[float],
        load_forecast: list[float],
        initial_soc: float,
        start_time: datetime | None = None,
        config: OptimizationConfig | None = None,
    ) -> OptimizationResult:
        """
        Run the battery optimization.

        Args:
            prices_import: Import prices in $/kWh for each interval
            prices_export: Export prices in $/kWh for each interval (positive = you get paid)
            solar_forecast: Solar generation forecast in Watts
            load_forecast: Load consumption forecast in Watts
            initial_soc: Current battery state of charge (0-1)
            start_time: Start time of the optimization horizon
            config: Optional override configuration

        Returns:
            OptimizationResult with optimal schedule
        """
        import time
        start_solve = time.time()

        cfg = config or self.config

        if start_time is None:
            start_time = datetime.now()

        n_intervals = len(prices_import)
        if not all(len(x) == n_intervals for x in [prices_export, solar_forecast, load_forecast]):
            return OptimizationResult(
                success=False,
                status="Input arrays must have same length",
            )

        if n_intervals == 0:
            return OptimizationResult(
                success=False,
                status="No intervals provided",
            )

        if not self._solver_available:
            return self._fallback_schedule(
                prices_import, prices_export, solar_forecast, load_forecast,
                initial_soc, start_time, cfg
            )

        try:
            result = self._solve_lp(
                prices_import, prices_export, solar_forecast, load_forecast,
                initial_soc, start_time, cfg, n_intervals
            )
            result.solve_time_ms = (time.time() - start_solve) * 1000
            return result

        except Exception as e:
            _LOGGER.error(f"Optimization failed: {e}", exc_info=True)
            return OptimizationResult(
                success=False,
                status=f"Solver error: {str(e)}",
            )

    def _solve_lp(
        self,
        prices_import: list[float],
        prices_export: list[float],
        solar_forecast: list[float],
        load_forecast: list[float],
        initial_soc: float,
        start_time: datetime,
        cfg: OptimizationConfig,
        n_intervals: int,
    ) -> OptimizationResult:
        """Solve the LP optimization problem using CVXPY with explicit power flow tracking."""
        import cvxpy as cp

        # Convert to numpy arrays
        p_import = np.array(prices_import, dtype=float)
        p_export = np.array(prices_export, dtype=float)
        solar = np.array(solar_forecast, dtype=float)
        load = np.array(load_forecast, dtype=float)

        # Validate and clean data
        for name, arr in [("prices_import", p_import), ("prices_export", p_export),
                          ("solar", solar), ("load", load)]:
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                nan_count = np.sum(np.isnan(arr))
                inf_count = np.sum(np.isinf(arr))
                _LOGGER.warning(f"Invalid values in {name}: {nan_count} NaN, {inf_count} Inf")
                if name == "prices_import":
                    p_import = np.nan_to_num(arr, nan=0.30, posinf=1.0, neginf=0.0)
                elif name == "prices_export":
                    p_export = np.nan_to_num(arr, nan=0.05, posinf=1.0, neginf=0.0)
                elif name == "solar":
                    solar = np.nan_to_num(arr, nan=0.0, posinf=10000.0, neginf=0.0)
                else:
                    load = np.nan_to_num(arr, nan=0.0, posinf=10000.0, neginf=0.0)

        solar = np.maximum(solar, 0)
        load = np.maximum(load, 0)
        p_import = np.maximum(p_import, -1.0)
        p_export = np.maximum(p_export, -1.0)

        # CRITICAL: Validate and fix initial_soc to prevent infeasible constraints
        if np.isnan(initial_soc) or np.isinf(initial_soc):
            _LOGGER.warning(f"Invalid initial_soc: {initial_soc}, using 0.5")
            initial_soc = 0.5
        initial_soc = float(np.clip(initial_soc, 0.0, 1.0))

        # Ensure initial_soc doesn't exceed max_soc (would make problem infeasible)
        if initial_soc > cfg.max_soc:
            _LOGGER.warning(f"initial_soc ({initial_soc:.2%}) > max_soc ({cfg.max_soc:.2%}), clamping")
            initial_soc = cfg.max_soc

        # Ensure initial_soc isn't below backup_reserve (unless we allow recovery)
        min_soc_target = max(cfg.min_soc, cfg.backup_reserve)
        if initial_soc < min_soc_target:
            _LOGGER.info(f"initial_soc ({initial_soc:.2%}) < min_soc ({min_soc_target:.2%}), will allow recovery")

        _LOGGER.debug(f"Optimization inputs: intervals={n_intervals}, initial_soc={initial_soc:.2%}, cost_function={cfg.cost_function.value}")
        _LOGGER.debug(f"  prices_import: min={p_import.min():.3f}, max={p_import.max():.3f}, mean={p_import.mean():.3f}")
        _LOGGER.debug(f"  prices_export: min={p_export.min():.3f}, max={p_export.max():.3f}, mean={p_export.mean():.3f}")
        _LOGGER.debug(f"  solar: sum={solar.sum()*cfg.interval_minutes/60/1000:.1f}kWh, max={solar.max():.0f}W")
        _LOGGER.debug(f"  load: sum={load.sum()*cfg.interval_minutes/60/1000:.1f}kWh, max={load.max():.0f}W")

        dt_hours = cfg.interval_minutes / 60.0

        # Validate battery parameters
        capacity_wh = cfg.battery_capacity_wh
        if capacity_wh <= 0 or np.isnan(capacity_wh) or np.isinf(capacity_wh):
            _LOGGER.warning(f"Invalid battery capacity {capacity_wh}, using default 13500Wh")
            capacity_wh = 13500.0

        max_charge = cfg.max_charge_w
        max_discharge = cfg.max_discharge_w
        if max_charge <= 0 or np.isnan(max_charge):
            max_charge = 5000.0
        if max_discharge <= 0 or np.isnan(max_discharge):
            max_discharge = 5000.0

        # ========================================
        # DECISION VARIABLES - Explicit power flows
        # ========================================

        # Solar allocation
        solar_to_load = cp.Variable(n_intervals, nonneg=True)      # Solar → Home load
        solar_to_battery = cp.Variable(n_intervals, nonneg=True)   # Solar → Battery charge
        solar_to_grid = cp.Variable(n_intervals, nonneg=True)      # Solar → Grid export

        # Battery flows
        battery_to_load = cp.Variable(n_intervals, nonneg=True)    # Battery → Home load (CONSUME)
        battery_to_grid = cp.Variable(n_intervals, nonneg=True)    # Battery → Grid export (EXPORT)

        # Grid import allocation
        grid_to_load = cp.Variable(n_intervals, nonneg=True)       # Grid → Home load
        grid_to_battery = cp.Variable(n_intervals, nonneg=True)    # Grid → Battery charge

        # SOC trajectory
        soc = cp.Variable(n_intervals + 1)

        # ========================================
        # DERIVED QUANTITIES
        # ========================================

        # Total charge = solar + grid going to battery
        charge = solar_to_battery + grid_to_battery

        # Total discharge = battery going to load + grid
        discharge = battery_to_load + battery_to_grid

        # Total grid import = grid going to load + battery
        grid_import = grid_to_load + grid_to_battery

        # Total grid export = solar + battery going to grid
        grid_export = solar_to_grid + battery_to_grid

        # ========================================
        # CONSTRAINTS
        # ========================================
        constraints = []

        # Solar allocation: all solar must go somewhere
        for t in range(n_intervals):
            constraints.append(solar_to_load[t] + solar_to_battery[t] + solar_to_grid[t] == solar[t])

        # Load satisfaction: load must be covered by solar, battery, or grid
        for t in range(n_intervals):
            constraints.append(solar_to_load[t] + battery_to_load[t] + grid_to_load[t] == load[t])

        # Initial SOC
        constraints.append(soc[0] == initial_soc)

        # SOC dynamics
        for t in range(n_intervals):
            energy_in = charge[t] * cfg.charge_efficiency * dt_hours
            energy_out = discharge[t] / cfg.discharge_efficiency * dt_hours
            delta_soc = (energy_in - energy_out) / capacity_wh
            constraints.append(soc[t + 1] == soc[t] + delta_soc)

        # SOC bounds
        min_soc = max(cfg.min_soc, cfg.backup_reserve)

        if initial_soc < min_soc:
            # Allow gradual recovery
            soc_deficit = min_soc - initial_soc
            energy_deficit_wh = soc_deficit * capacity_wh
            max_energy_per_interval = max_charge * cfg.charge_efficiency * dt_hours
            intervals_to_recover = int(np.ceil(energy_deficit_wh / max_energy_per_interval)) if max_energy_per_interval > 0 else 1

            for t in range(n_intervals + 1):
                if t <= intervals_to_recover:
                    recovery_progress = t / max(intervals_to_recover, 1)
                    min_soc_at_t = initial_soc + (min_soc - initial_soc) * recovery_progress * 0.8
                    constraints.append(soc[t] >= min_soc_at_t - 0.02)
                else:
                    constraints.append(soc[t] >= min_soc)
                constraints.append(soc[t] <= cfg.max_soc)
        else:
            for t in range(n_intervals + 1):
                constraints.append(soc[t] >= min_soc)
                constraints.append(soc[t] <= cfg.max_soc)

        # Target end SOC
        if cfg.target_end_soc is not None:
            constraints.append(soc[n_intervals] >= cfg.target_end_soc)

        # Power limits
        constraints.append(charge <= max_charge)
        constraints.append(discharge <= max_discharge)

        # Grid limits
        if cfg.max_grid_import_w is not None:
            constraints.append(grid_import <= cfg.max_grid_import_w)
        if cfg.max_grid_export_w is not None:
            constraints.append(grid_export <= cfg.max_grid_export_w)

        # Self-consumption mode: only allow grid-to-battery charging when electricity is free/negative
        # Battery should charge from excess solar, or from grid when price <= 0
        if cfg.cost_function == CostFunction.SELF_CONSUMPTION:
            for t in range(n_intervals):
                if p_import[t] > 0:
                    # Price is positive - don't charge from grid
                    constraints.append(grid_to_battery[t] == 0)
                # else: price <= 0, allow charging (it's free or they pay us!)
            _LOGGER.info("Self-consumption mode: grid charging only when price <= 0")

        # CRITICAL: Prevent simultaneous grid charging AND battery export
        # This is physically wasteful (round-trip losses) and should never happen
        # We use a "big-M" style constraint to enforce mutual exclusivity
        # If grid_to_battery > 0, then battery_to_grid must be 0 (and vice versa)
        # Since we want LP (not MILP), we add a very large penalty instead
        # This is handled in the objective function below

        # ========================================
        # OBJECTIVE FUNCTION
        # ========================================

        # Thresholds
        LOW_EXPORT_THRESHOLD = 0.05   # Export prices below this are "worthless"
        MIN_WORTHWHILE_EXPORT = 0.10  # Don't export battery unless price > this

        # Cost components (common to all objectives)
        import_cost = cp.sum(cp.multiply(p_import, grid_import)) * dt_hours / 1000
        export_revenue = cp.sum(cp.multiply(p_export, grid_export)) * dt_hours / 1000

        # CRITICAL: Penalty for simultaneous grid charging AND battery export
        # This should NEVER happen - it means buying from grid, storing (losing 10%),
        # then immediately discharging to grid (losing another 10%) - always a net loss
        # unless prices are extremely skewed. Even then, it wastes battery cycles.
        # We penalize the minimum of grid_to_battery and battery_to_grid to
        # discourage having both non-zero.
        # Using element-wise minimum: min(a,b) = 0.5*(a+b - |a-b|) but cvxpy doesn't support abs on variables
        # Instead, we just add a penalty on both when they're both non-zero
        # Since we can't detect "both non-zero" in LP, we use a heuristic penalty
        SIMULTANEOUS_CHARGE_EXPORT_PENALTY = 100.0  # $/kWh penalty
        simultaneous_penalty = SIMULTANEOUS_CHARGE_EXPORT_PENALTY * cp.sum(
            cp.minimum(grid_to_battery, battery_to_grid)
        ) * dt_hours / 1000

        if cfg.cost_function == CostFunction.COST_MINIMIZATION:
            # Minimize electricity cost
            # Penalize battery_to_grid (export from battery) when prices are low
            # This is the key insight: consuming battery to cover load is FINE,
            # but exporting battery at low prices is WASTEFUL
            battery_export_penalty_weights = np.where(p_export <= LOW_EXPORT_THRESHOLD, 5.0, 0)
            battery_export_penalty = cp.sum(cp.multiply(battery_export_penalty_weights, battery_to_grid)) * dt_hours / 1000

            # Also penalize solar export at very low prices (opportunity cost)
            solar_export_penalty_weights = np.where(p_export < MIN_WORTHWHILE_EXPORT, 0.5, 0)
            solar_export_penalty = cp.sum(cp.multiply(solar_export_penalty_weights, solar_to_grid)) * dt_hours / 1000

            # Penalize grid_to_battery when import prices are high
            avg_import = np.mean(p_import)
            high_price_charge_penalty = np.where(p_import > avg_import * 1.2, 0.5, 0)
            expensive_charge_penalty = cp.sum(cp.multiply(high_price_charge_penalty, grid_to_battery)) * dt_hours / 1000

            objective = cp.Minimize(
                import_cost - export_revenue +
                battery_export_penalty + solar_export_penalty + expensive_charge_penalty +
                simultaneous_penalty
            )

        elif cfg.cost_function == CostFunction.PROFIT_MAXIMIZATION:
            # Maximize profit from grid trading
            # Key: ONLY export battery when prices are good
            # battery_to_load (consume) has NO penalty - using battery to avoid import is fine
            # battery_to_grid (export) should only happen at good prices

            battery_export_penalty_weights = np.where(p_export <= LOW_EXPORT_THRESHOLD, 10.0, 0)
            battery_export_penalty = cp.sum(cp.multiply(battery_export_penalty_weights, battery_to_grid)) * dt_hours / 1000

            # Small penalty for solar export at very low prices
            solar_export_penalty_weights = np.where(p_export < MIN_WORTHWHILE_EXPORT, 1.0, 0)
            solar_export_penalty = cp.sum(cp.multiply(solar_export_penalty_weights, solar_to_grid)) * dt_hours / 1000

            objective = cp.Minimize(
                import_cost - export_revenue +
                battery_export_penalty + solar_export_penalty +
                simultaneous_penalty
            )

        else:  # SELF_CONSUMPTION
            # Maximize solar self-consumption
            # Priority: use solar for load, then store in battery, then export
            # Never export battery at low prices

            FREE_THRESHOLD = 0.01

            # Penalize grid imports when electricity costs money
            import_penalty_weights = np.where(p_import <= FREE_THRESHOLD, 0, 50)
            import_penalty = cp.sum(cp.multiply(import_penalty_weights, grid_import)) * dt_hours / 1000

            # Incentivize charging during free periods
            charge_incentive_weights = np.where(p_import <= FREE_THRESHOLD, 0.1, 0)
            charge_incentive = cp.sum(cp.multiply(charge_incentive_weights, grid_to_battery)) * dt_hours / 1000

            # Heavy penalty for battery export at low prices
            battery_export_penalty_weights = np.where(p_export <= LOW_EXPORT_THRESHOLD, 100, 0)
            battery_export_penalty = cp.sum(cp.multiply(battery_export_penalty_weights, battery_to_grid)) * dt_hours / 1000

            # Light penalty for solar export (prefer self-consumption)
            solar_export_penalty = cp.sum(solar_to_grid) * dt_hours / 1000 * 0.1

            objective = cp.Minimize(
                import_penalty + battery_export_penalty + solar_export_penalty +
                import_cost - charge_incentive + simultaneous_penalty
            )

        # Cycle cost
        if cfg.cycle_cost > 0:
            cycle_penalty = cfg.cycle_cost * cp.sum(charge + discharge) * dt_hours / 1000
            objective = cp.Minimize(objective.args[0] + cycle_penalty)

        # ========================================
        # SOLVE
        # ========================================
        problem = cp.Problem(objective, constraints)
        SOLVER_TIMEOUT = 30

        solver_name = "HIGHS"
        try:
            if "HIGHS" in cp.installed_solvers():
                problem.solve(solver=cp.HIGHS, verbose=False, time_limit=SOLVER_TIMEOUT)
            else:
                problem.solve(verbose=False, solver_opts={"time_limit": SOLVER_TIMEOUT})
                solver_name = "default"
        except Exception as e:
            _LOGGER.warning(f"Primary solver failed: {e}, trying fallback")
            problem.solve(verbose=False)
            solver_name = "fallback"

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            _LOGGER.warning(f"Optimization failed with status: {problem.status}")
            return OptimizationResult(
                success=False,
                status=f"Solver status: {problem.status}",
                solver_name=solver_name,
            )

        # ========================================
        # EXTRACT RESULTS
        # ========================================

        # Check for None values (solver didn't find solution)
        if solar_to_load.value is None or battery_to_load.value is None:
            _LOGGER.error("Solver returned None values - optimization failed")
            return OptimizationResult(
                success=False,
                status="Solver returned no solution",
                solver_name=solver_name,
            )

        # Detailed flow results
        solar_to_load_vals = np.maximum(solar_to_load.value, 0).tolist()
        solar_to_battery_vals = np.maximum(solar_to_battery.value, 0).tolist()
        solar_to_grid_vals = np.maximum(solar_to_grid.value, 0).tolist()
        battery_to_load_vals = np.maximum(battery_to_load.value, 0).tolist()
        battery_to_grid_vals = np.maximum(battery_to_grid.value, 0).tolist()
        grid_to_load_vals = np.maximum(grid_to_load.value, 0).tolist()
        grid_to_battery_vals = np.maximum(grid_to_battery.value, 0).tolist()

        # SANITY CHECK: Detect impossible simultaneous grid charge and battery export
        # If both happen at same interval, zero out the smaller one
        for t in range(n_intervals):
            gtb = grid_to_battery_vals[t]
            btg = battery_to_grid_vals[t]
            if gtb > 10 and btg > 10:  # Both non-trivial
                _LOGGER.warning(f"Interval {t}: Impossible state - grid_to_battery={gtb:.0f}W AND battery_to_grid={btg:.0f}W. Correcting.")
                # Keep the larger one, zero the smaller
                if gtb > btg:
                    battery_to_grid_vals[t] = 0
                    # Adjust solar_to_grid to compensate if needed
                else:
                    grid_to_battery_vals[t] = 0
                    # Adjust grid_to_load to compensate if needed

        # Aggregate values (backward compatible)
        charge_w = [s + g for s, g in zip(solar_to_battery_vals, grid_to_battery_vals)]
        discharge_w = [l + g for l, g in zip(battery_to_load_vals, battery_to_grid_vals)]
        import_w = [l + b for l, b in zip(grid_to_load_vals, grid_to_battery_vals)]
        export_w = [s + b for s, b in zip(solar_to_grid_vals, battery_to_grid_vals)]
        soc_values = np.clip(soc.value, 0, 1).tolist()

        timestamps = [
            start_time + timedelta(minutes=cfg.interval_minutes * i)
            for i in range(n_intervals)
        ]

        # Summary metrics
        total_import_kwh = sum(import_w) * dt_hours / 1000
        total_export_kwh = sum(export_w) * dt_hours / 1000
        total_charge_kwh = sum(charge_w) * dt_hours / 1000
        total_discharge_kwh = sum(discharge_w) * dt_hours / 1000

        total_battery_consume_kwh = sum(battery_to_load_vals) * dt_hours / 1000
        total_battery_export_kwh = sum(battery_to_grid_vals) * dt_hours / 1000
        total_solar_consumed_kwh = (sum(solar_to_load_vals) + sum(solar_to_battery_vals)) * dt_hours / 1000
        total_solar_exported_kwh = sum(solar_to_grid_vals) * dt_hours / 1000

        total_import_cost = sum(p * e * dt_hours / 1000 for p, e in zip(p_import, import_w))
        total_export_revenue = sum(p * e * dt_hours / 1000 for p, e in zip(p_export, export_w))
        total_cost = total_import_cost - total_export_revenue

        avg_import_price = total_import_cost / total_import_kwh if total_import_kwh > 0 else 0
        avg_export_price = total_export_revenue / total_export_kwh if total_export_kwh > 0 else 0

        baseline_cost = self._calculate_baseline_cost(p_import, p_export, solar, load, dt_hours)

        _LOGGER.info(f"Optimization complete: cost=${total_cost:.2f}, "
                     f"battery_consume={total_battery_consume_kwh:.1f}kWh, "
                     f"battery_export={total_battery_export_kwh:.1f}kWh")

        return OptimizationResult(
            success=True,
            status="optimal",
            # Legacy fields
            charge_schedule_w=charge_w,
            discharge_schedule_w=discharge_w,
            grid_import_w=import_w,
            grid_export_w=export_w,
            soc_trajectory=soc_values,
            # New detailed breakdown
            battery_consume_w=battery_to_load_vals,
            battery_export_w=battery_to_grid_vals,
            solar_to_load_w=solar_to_load_vals,
            solar_to_battery_w=solar_to_battery_vals,
            solar_to_grid_w=solar_to_grid_vals,
            grid_to_load_w=grid_to_load_vals,
            grid_to_battery_w=grid_to_battery_vals,
            timestamps=timestamps,
            total_cost=total_cost,
            total_import_kwh=total_import_kwh,
            total_export_kwh=total_export_kwh,
            total_charge_kwh=total_charge_kwh,
            total_discharge_kwh=total_discharge_kwh,
            average_import_price=avg_import_price,
            average_export_price=avg_export_price,
            total_battery_consume_kwh=total_battery_consume_kwh,
            total_battery_export_kwh=total_battery_export_kwh,
            total_solar_consumed_kwh=total_solar_consumed_kwh,
            total_solar_exported_kwh=total_solar_exported_kwh,
            baseline_cost=baseline_cost,
            savings=baseline_cost - total_cost,
            solver_name=solver_name,
        )

    def _calculate_baseline_cost(
        self,
        prices_import: np.ndarray,
        prices_export: np.ndarray,
        solar: np.ndarray,
        load: np.ndarray,
        dt_hours: float,
    ) -> float:
        """Calculate baseline cost without battery optimization."""
        total_cost = 0.0
        for t in range(len(prices_import)):
            net_load = load[t] - solar[t]
            if net_load > 0:
                energy_kwh = net_load * dt_hours / 1000
                total_cost += prices_import[t] * energy_kwh
            else:
                energy_kwh = -net_load * dt_hours / 1000
                total_cost -= prices_export[t] * energy_kwh
        return total_cost

    def _fallback_schedule(
        self,
        prices_import: list[float],
        prices_export: list[float],
        solar_forecast: list[float],
        load_forecast: list[float],
        initial_soc: float,
        start_time: datetime,
        cfg: OptimizationConfig,
    ) -> OptimizationResult:
        """Generate a simple heuristic schedule when LP solver is unavailable."""
        n_intervals = len(prices_import)
        dt_hours = cfg.interval_minutes / 60.0
        capacity_wh = cfg.battery_capacity_wh

        avg_import = sum(prices_import) / n_intervals
        avg_export = sum(prices_export) / n_intervals

        # Result arrays
        charge_w = []
        discharge_w = []
        battery_to_load_w = []
        battery_to_grid_w = []
        solar_to_load_w = []
        solar_to_battery_w = []
        solar_to_grid_w = []
        grid_to_load_w = []
        grid_to_battery_w = []
        soc_values = [initial_soc]

        current_soc = initial_soc

        for t in range(n_intervals):
            solar_val = solar_forecast[t]
            load_val = load_forecast[t]

            # Initialize all flows to zero
            stl = min(solar_val, load_val)  # Solar to load
            remaining_solar = solar_val - stl
            remaining_load = load_val - stl

            stb = 0.0  # Solar to battery
            stg = 0.0  # Solar to grid
            btl = 0.0  # Battery to load (consume)
            btg = 0.0  # Battery to grid (export)
            gtl = 0.0  # Grid to load
            gtb = 0.0  # Grid to battery

            # Low import price - charge from grid
            # In self_consumption mode, only charge when price <= 0 (free/negative)
            # In other modes, charge when price is low
            should_charge_from_grid = False
            if cfg.cost_function == CostFunction.SELF_CONSUMPTION:
                should_charge_from_grid = prices_import[t] <= 0
            else:
                should_charge_from_grid = prices_import[t] < avg_import * 0.7

            if should_charge_from_grid and current_soc < cfg.max_soc - 0.1:
                available_capacity = (cfg.max_soc - current_soc) * capacity_wh / dt_hours
                gtb = min(cfg.max_charge_w, available_capacity) - remaining_solar
                gtb = max(0, gtb)

            # Store excess solar
            if remaining_solar > 0 and current_soc < cfg.max_soc:
                available_capacity = (cfg.max_soc - current_soc) * capacity_wh / dt_hours
                stb = min(remaining_solar, available_capacity, cfg.max_charge_w)
                remaining_solar -= stb

            # Export remaining solar
            stg = remaining_solar

            # High export price and low solar - discharge to export
            if prices_export[t] > avg_export * 1.3 and prices_export[t] > 0.10:
                if current_soc > cfg.backup_reserve + 0.1:
                    available_energy = (current_soc - cfg.backup_reserve) * capacity_wh / dt_hours
                    btg = min(cfg.max_discharge_w, available_energy)

            # Cover remaining load from battery or grid
            if remaining_load > 0:
                if current_soc > cfg.backup_reserve + 0.05:
                    available_energy = (current_soc - cfg.backup_reserve) * capacity_wh / dt_hours
                    btl = min(remaining_load, available_energy, cfg.max_discharge_w - btg)
                    remaining_load -= btl
                gtl = remaining_load

            # Update SOC
            total_charge = stb + gtb
            total_discharge = btl + btg
            energy_in = total_charge * cfg.charge_efficiency * dt_hours
            energy_out = total_discharge / cfg.discharge_efficiency * dt_hours
            current_soc += (energy_in - energy_out) / capacity_wh
            current_soc = max(cfg.backup_reserve, min(cfg.max_soc, current_soc))

            # Store results
            charge_w.append(total_charge)
            discharge_w.append(total_discharge)
            battery_to_load_w.append(btl)
            battery_to_grid_w.append(btg)
            solar_to_load_w.append(stl)
            solar_to_battery_w.append(stb)
            solar_to_grid_w.append(stg)
            grid_to_load_w.append(gtl)
            grid_to_battery_w.append(gtb)
            soc_values.append(current_soc)

        timestamps = [
            start_time + timedelta(minutes=cfg.interval_minutes * i)
            for i in range(n_intervals)
        ]

        # Calculate aggregates
        import_w = [gtl + gtb for gtl, gtb in zip(grid_to_load_w, grid_to_battery_w)]
        export_w = [stg + btg for stg, btg in zip(solar_to_grid_w, battery_to_grid_w)]

        total_import_kwh = sum(import_w) * dt_hours / 1000
        total_export_kwh = sum(export_w) * dt_hours / 1000
        total_charge_kwh = sum(charge_w) * dt_hours / 1000
        total_discharge_kwh = sum(discharge_w) * dt_hours / 1000

        total_cost = sum(p * e * dt_hours / 1000 for p, e in zip(prices_import, import_w))
        total_cost -= sum(p * e * dt_hours / 1000 for p, e in zip(prices_export, export_w))

        baseline_cost = self._calculate_baseline_cost(
            np.array(prices_import), np.array(prices_export),
            np.array(solar_forecast), np.array(load_forecast), dt_hours
        )

        return OptimizationResult(
            success=True,
            status="heuristic (solver unavailable)",
            charge_schedule_w=charge_w,
            discharge_schedule_w=discharge_w,
            grid_import_w=import_w,
            grid_export_w=export_w,
            soc_trajectory=soc_values,
            battery_consume_w=battery_to_load_w,
            battery_export_w=battery_to_grid_w,
            solar_to_load_w=solar_to_load_w,
            solar_to_battery_w=solar_to_battery_w,
            solar_to_grid_w=solar_to_grid_w,
            grid_to_load_w=grid_to_load_w,
            grid_to_battery_w=grid_to_battery_w,
            timestamps=timestamps,
            total_cost=total_cost,
            total_import_kwh=total_import_kwh,
            total_export_kwh=total_export_kwh,
            total_charge_kwh=total_charge_kwh,
            total_discharge_kwh=total_discharge_kwh,
            total_battery_consume_kwh=sum(battery_to_load_w) * dt_hours / 1000,
            total_battery_export_kwh=sum(battery_to_grid_w) * dt_hours / 1000,
            total_solar_consumed_kwh=(sum(solar_to_load_w) + sum(solar_to_battery_w)) * dt_hours / 1000,
            total_solar_exported_kwh=sum(solar_to_grid_w) * dt_hours / 1000,
            baseline_cost=baseline_cost,
            savings=baseline_cost - total_cost,
            solver_name="heuristic",
        )
