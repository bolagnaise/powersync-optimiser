"""
Battery optimization engine using Linear Programming.

Solves the optimal battery charge/discharge schedule to minimize electricity costs
or maximize profit/self-consumption based on price forecasts and solar predictions.
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

    # Schedules (per interval)
    charge_schedule_w: list[float] = field(default_factory=list)    # Watts to charge
    discharge_schedule_w: list[float] = field(default_factory=list) # Watts to discharge
    grid_import_w: list[float] = field(default_factory=list)        # Watts imported from grid
    grid_export_w: list[float] = field(default_factory=list)        # Watts exported to grid
    soc_trajectory: list[float] = field(default_factory=list)       # SOC at each interval (0-1)

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

        if charge_w > 10:  # Small threshold to avoid floating point noise
            return {"action": "charge", "power_w": charge_w}
        elif discharge_w > 10:
            return {"action": "discharge", "power_w": discharge_w}
        else:
            return {"action": "idle", "power_w": 0}

    def get_next_actions(self, count: int = 5) -> list[dict[str, Any]]:
        """Get the next N actions from the schedule."""
        actions = []
        for i in range(min(count, len(self.charge_schedule_w))):
            action = self.get_action_at_index(i)
            action["timestamp"] = self.timestamps[i].isoformat() if i < len(self.timestamps) else None
            # soc_trajectory has n+1 elements: [initial, after_interval_0, after_interval_1, ...]
            # Show SOC AFTER this action completes, not before
            action["soc"] = self.soc_trajectory[i + 1] if (i + 1) < len(self.soc_trajectory) else None
            actions.append(action)
        return actions

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "status": self.status,
            "schedule": {
                "charge_w": self.charge_schedule_w,
                "discharge_w": self.discharge_schedule_w,
                "grid_import_w": self.grid_import_w,
                "grid_export_w": self.grid_export_w,
                "soc": self.soc_trajectory,
                "timestamps": [t.isoformat() for t in self.timestamps],
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
            # Check if HiGHS is available
            if "HIGHS" in cp.installed_solvers():
                _LOGGER.info("CVXPY with HiGHS solver available")
                return True
            # Fall back to other solvers
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
        prices_import: list[float],      # $/kWh for each interval
        prices_export: list[float],      # $/kWh for each interval
        solar_forecast: list[float],     # Watts for each interval
        load_forecast: list[float],      # Watts for each interval
        initial_soc: float,              # Current SOC (0-1)
        start_time: datetime | None = None,            # Start time of optimization
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

        # Validate inputs
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
        """Solve the LP optimization problem using CVXPY."""
        import cvxpy as cp

        # Convert to numpy arrays and validate
        p_import = np.array(prices_import, dtype=float)  # $/kWh
        p_export = np.array(prices_export, dtype=float)  # $/kWh
        solar = np.array(solar_forecast, dtype=float)    # W
        load = np.array(load_forecast, dtype=float)      # W

        # Check for NaN/Inf values
        for name, arr in [("prices_import", p_import), ("prices_export", p_export),
                          ("solar", solar), ("load", load)]:
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                nan_count = np.sum(np.isnan(arr))
                inf_count = np.sum(np.isinf(arr))
                _LOGGER.warning(f"Invalid values in {name}: {nan_count} NaN, {inf_count} Inf")
                # Replace NaN/Inf with reasonable defaults
                if name == "prices_import":
                    arr = np.nan_to_num(arr, nan=0.30, posinf=1.0, neginf=0.0)
                elif name == "prices_export":
                    arr = np.nan_to_num(arr, nan=0.05, posinf=1.0, neginf=0.0)
                else:
                    arr = np.nan_to_num(arr, nan=0.0, posinf=10000.0, neginf=0.0)
                # Update the array reference
                if name == "prices_import":
                    p_import = arr
                elif name == "prices_export":
                    p_export = arr
                elif name == "solar":
                    solar = arr
                else:
                    load = arr

        # Clamp negative values (shouldn't have negative power/prices)
        solar = np.maximum(solar, 0)
        load = np.maximum(load, 0)
        p_import = np.maximum(p_import, -1.0)  # Allow small negative (free) but not extreme
        p_export = np.maximum(p_export, -1.0)

        # Log data summary for debugging
        _LOGGER.debug(f"Optimization inputs: intervals={n_intervals}, initial_soc={initial_soc:.2%}")
        _LOGGER.debug(f"  prices_import: min={p_import.min():.3f}, max={p_import.max():.3f}, mean={p_import.mean():.3f}")
        _LOGGER.debug(f"  prices_export: min={p_export.min():.3f}, max={p_export.max():.3f}, mean={p_export.mean():.3f}")
        _LOGGER.debug(f"  solar: min={solar.min():.0f}W, max={solar.max():.0f}W, sum={solar.sum()/1000:.1f}kWh")
        _LOGGER.debug(f"  load: min={load.min():.0f}W, max={load.max():.0f}W, sum={load.sum()/1000:.1f}kWh")
        _LOGGER.debug(f"  battery: capacity={cfg.battery_capacity_wh}Wh, reserve={cfg.backup_reserve:.0%}")

        # Time interval in hours
        dt_hours = cfg.interval_minutes / 60.0

        # Capacity in Wh - validate and use sensible defaults
        capacity_wh = cfg.battery_capacity_wh
        if capacity_wh <= 0 or np.isnan(capacity_wh) or np.isinf(capacity_wh):
            _LOGGER.warning(f"Invalid battery capacity {capacity_wh}, using default 13500Wh")
            capacity_wh = 13500.0

        # Validate power limits
        max_charge = cfg.max_charge_w
        max_discharge = cfg.max_discharge_w
        if max_charge <= 0 or np.isnan(max_charge) or np.isinf(max_charge):
            _LOGGER.warning(f"Invalid max_charge_w {max_charge}, using default 5000W")
            max_charge = 5000.0
        if max_discharge <= 0 or np.isnan(max_discharge) or np.isinf(max_discharge):
            _LOGGER.warning(f"Invalid max_discharge_w {max_discharge}, using default 5000W")
            max_discharge = 5000.0

        # Decision variables
        charge = cp.Variable(n_intervals, nonneg=True)      # Charge power (W)
        discharge = cp.Variable(n_intervals, nonneg=True)   # Discharge power (W)
        grid_import = cp.Variable(n_intervals, nonneg=True) # Grid import (W)
        grid_export = cp.Variable(n_intervals, nonneg=True) # Grid export (W)
        soc = cp.Variable(n_intervals + 1)                  # SOC at each step (0-1)

        constraints = []

        # Initial SOC constraint
        constraints.append(soc[0] == initial_soc)

        # SOC dynamics: soc[t+1] = soc[t] + (charge*eta - discharge/eta) * dt / capacity
        for t in range(n_intervals):
            energy_in = charge[t] * cfg.charge_efficiency * dt_hours  # Wh
            energy_out = discharge[t] / cfg.discharge_efficiency * dt_hours  # Wh
            delta_soc = (energy_in - energy_out) / capacity_wh
            constraints.append(soc[t + 1] == soc[t] + delta_soc)

        # SOC bounds
        min_soc = max(cfg.min_soc, cfg.backup_reserve)

        # Handle case where initial SOC is below reserve
        if initial_soc < min_soc:
            # Calculate how many intervals needed to charge from current SOC to min_soc
            # at maximum charge rate
            soc_deficit = min_soc - initial_soc
            energy_deficit_wh = soc_deficit * capacity_wh
            # Max energy per interval (accounting for efficiency)
            max_energy_per_interval = max_charge * cfg.charge_efficiency * dt_hours
            intervals_to_recover = int(np.ceil(energy_deficit_wh / max_energy_per_interval)) if max_energy_per_interval > 0 else 1

            _LOGGER.debug(f"SOC below reserve: {initial_soc:.1%} < {min_soc:.1%}, need {intervals_to_recover} intervals to recover")

            # Allow gradual recovery: linearly interpolate min SOC from current to reserve
            for t in range(n_intervals + 1):
                if t <= intervals_to_recover:
                    # During recovery period: allow linear ramp from initial to min_soc
                    # with some margin for flexibility
                    recovery_progress = t / max(intervals_to_recover, 1)
                    min_soc_at_t = initial_soc + (min_soc - initial_soc) * recovery_progress * 0.8  # 80% of ideal recovery
                    constraints.append(soc[t] >= min_soc_at_t - 0.02)  # Small tolerance
                else:
                    # After recovery period: enforce normal min_soc
                    constraints.append(soc[t] >= min_soc)
                constraints.append(soc[t] <= cfg.max_soc)
        else:
            # Normal case - enforce min_soc for all time periods
            for t in range(n_intervals + 1):
                constraints.append(soc[t] >= min_soc)
                constraints.append(soc[t] <= cfg.max_soc)

        # Target end SOC if specified
        if cfg.target_end_soc is not None:
            constraints.append(soc[n_intervals] >= cfg.target_end_soc)

        # Power limits (using validated values)
        constraints.append(charge <= max_charge)
        constraints.append(discharge <= max_discharge)

        # Grid limits if specified
        if cfg.max_grid_import_w is not None:
            constraints.append(grid_import <= cfg.max_grid_import_w)
        if cfg.max_grid_export_w is not None:
            constraints.append(grid_export <= cfg.max_grid_export_w)

        # Power balance: solar + grid_import + discharge = load + grid_export + charge
        for t in range(n_intervals):
            power_in = solar[t] + grid_import[t] + discharge[t]
            power_out = load[t] + grid_export[t] + charge[t]
            constraints.append(power_in == power_out)

        # Objective function
        # Common thresholds for all modes
        LOW_EXPORT_THRESHOLD = 0.05   # Export prices below this ($/kWh) are "not worth it"
        MIN_WORTHWHILE_EXPORT = 0.10  # Don't discharge unless export > this

        if cfg.cost_function == CostFunction.COST_MINIMIZATION:
            # Minimize: total electricity cost = import cost - export revenue
            # Key insight: Discharge to cover load is GOOD (avoids import), but only if
            # we're not wasting the stored energy that could be used/exported later at better prices
            import_cost = cp.sum(cp.multiply(p_import, grid_import)) * dt_hours / 1000
            export_revenue = cp.sum(cp.multiply(p_export, grid_export)) * dt_hours / 1000

            # Penalize grid export when prices are very low (wasting stored energy)
            low_export_penalty = np.where(p_export < MIN_WORTHWHILE_EXPORT, 2.0, 0)
            wasteful_export_cost = cp.sum(cp.multiply(low_export_penalty, grid_export)) * dt_hours / 1000

            # Also add small discharge penalty when export prices are near-zero
            # This prevents "use battery now to avoid import" when prices are HIGH
            # because the battery could have been used/exported more profitably later
            # Penalty scales with how bad the export price is
            discharge_penalty_weights = np.where(p_export <= LOW_EXPORT_THRESHOLD, 0.5, 0)
            discharge_penalty = cp.sum(cp.multiply(discharge_penalty_weights, discharge)) * dt_hours / 1000

            # Also discourage charging when import prices are high (above average)
            avg_import = np.mean(p_import)
            high_price_charge_penalty = np.where(p_import > avg_import * 1.2, 0.3, 0)
            expensive_charge_cost = cp.sum(cp.multiply(high_price_charge_penalty, charge)) * dt_hours / 1000

            objective = cp.Minimize(
                import_cost - export_revenue + wasteful_export_cost +
                discharge_penalty + expensive_charge_cost
            )

        elif cfg.cost_function == CostFunction.PROFIT_MAXIMIZATION:
            # Maximize: export revenue - import cost
            # BUT: Don't discharge when we get nothing for it (export price near zero)
            import_cost = cp.sum(cp.multiply(p_import, grid_import)) * dt_hours / 1000
            export_revenue = cp.sum(cp.multiply(p_export, grid_export)) * dt_hours / 1000

            # Heavy penalty for discharge when export prices are low
            # This prevents the optimizer from "wasting" battery at $0 export
            discharge_penalty_weights = np.where(p_export <= LOW_EXPORT_THRESHOLD, 5.0, 0)
            wasteful_discharge_penalty = cp.sum(cp.multiply(discharge_penalty_weights, discharge)) * dt_hours / 1000

            # Also penalize grid export at very low prices
            low_export_penalty = np.where(p_export < MIN_WORTHWHILE_EXPORT, 2.0, 0)
            wasteful_export_cost = cp.sum(cp.multiply(low_export_penalty, grid_export)) * dt_hours / 1000

            # Maximize profit = export_revenue - import_cost - penalties
            # (converted to minimize negative profit)
            objective = cp.Minimize(
                import_cost - export_revenue + wasteful_discharge_penalty + wasteful_export_cost
            )

        else:  # SELF_CONSUMPTION
            # Maximize solar self-consumption while:
            # 1. Allowing (encouraging) charging from FREE electricity
            # 2. Avoiding grid imports when electricity costs money
            # 3. NOT discharging/exporting when export prices are low (don't waste battery)
            FREE_THRESHOLD = 0.01  # Prices below this ($/kWh) are considered "free"

            # For free periods: no import penalty, encourage charging by giving credit
            # For paid periods: penalize imports to encourage self-consumption
            import_penalty_weights = np.where(p_import <= FREE_THRESHOLD, 0, 50)
            charge_incentive_weights = np.where(p_import <= FREE_THRESHOLD, 0.1, 0)

            # Penalize discharging when export prices are low (don't waste stored energy)
            # High penalty when export is $0 or very low - never discharge for nothing
            discharge_penalty_weights = np.where(p_export <= LOW_EXPORT_THRESHOLD, 100, 0)

            # Objective: minimize (paid imports) + (wasteful discharge) - (free charging benefit) + actual cost
            import_penalty = cp.sum(cp.multiply(import_penalty_weights, grid_import)) * dt_hours / 1000
            discharge_penalty = cp.sum(cp.multiply(discharge_penalty_weights, discharge)) * dt_hours / 1000
            charge_incentive = cp.sum(cp.multiply(charge_incentive_weights, charge)) * dt_hours / 1000
            import_cost = cp.sum(cp.multiply(p_import, grid_import)) * dt_hours / 1000

            objective = cp.Minimize(import_penalty + discharge_penalty + import_cost - charge_incentive)

        # Add cycle cost if specified
        if cfg.cycle_cost > 0:
            cycle_penalty = cfg.cycle_cost * cp.sum(charge + discharge) * dt_hours / 1000
            objective = cp.Minimize(objective.args[0] + cycle_penalty)

        # Solve the problem
        problem = cp.Problem(objective, constraints)

        # Solver timeout in seconds (prevent long-running optimizations)
        SOLVER_TIMEOUT = 30

        # Try HiGHS first, fall back to other solvers
        solver_name = "HIGHS"
        try:
            if "HIGHS" in cp.installed_solvers():
                problem.solve(solver=cp.HIGHS, verbose=False, time_limit=SOLVER_TIMEOUT)
            else:
                # Fall back to available solver with timeout
                problem.solve(verbose=False, solver_opts={"time_limit": SOLVER_TIMEOUT})
                solver_name = "default"
        except Exception as e:
            _LOGGER.warning(f"Primary solver failed: {e}, trying fallback")
            problem.solve(verbose=False)
            solver_name = "fallback"

        # Check solution status
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # Log diagnostic info for infeasible problems
            _LOGGER.warning(f"Optimization failed with status: {problem.status}")
            _LOGGER.warning(f"  Config: capacity={cfg.battery_capacity_wh}Wh, max_charge={cfg.max_charge_w}W, max_discharge={cfg.max_discharge_w}W")
            _LOGGER.warning(f"  Config: backup_reserve={cfg.backup_reserve:.0%}, min_soc={cfg.min_soc:.0%}, max_soc={cfg.max_soc:.0%}")
            _LOGGER.warning(f"  Initial SOC: {initial_soc:.2%}, min_soc_effective={max(cfg.min_soc, cfg.backup_reserve):.0%}")
            _LOGGER.warning(f"  Data ranges: prices=[{p_import.min():.3f}, {p_import.max():.3f}], solar=[{solar.min():.0f}, {solar.max():.0f}], load=[{load.min():.0f}, {load.max():.0f}]")

            # Check for potential issues
            if initial_soc < max(cfg.min_soc, cfg.backup_reserve):
                _LOGGER.warning(f"  ISSUE: Initial SOC {initial_soc:.2%} is below min_soc {max(cfg.min_soc, cfg.backup_reserve):.0%}")
            if np.any(load > cfg.max_discharge_w + cfg.max_grid_import_w if cfg.max_grid_import_w else load > 100000):
                _LOGGER.warning(f"  ISSUE: Some load values may exceed available power sources")
            if solar.sum() == 0 and load.sum() > 0:
                _LOGGER.warning(f"  NOTE: No solar generation, all load must come from battery or grid")

            return OptimizationResult(
                success=False,
                status=f"Solver status: {problem.status}",
                solver_name=solver_name,
            )

        # Extract results
        charge_w = np.maximum(charge.value, 0).tolist()
        discharge_w = np.maximum(discharge.value, 0).tolist()
        import_w = np.maximum(grid_import.value, 0).tolist()
        export_w = np.maximum(grid_export.value, 0).tolist()
        soc_values = np.clip(soc.value, 0, 1).tolist()

        # Generate timestamps
        timestamps = [
            start_time + timedelta(minutes=cfg.interval_minutes * i)
            for i in range(n_intervals)
        ]

        # Calculate summary metrics
        total_import_kwh = sum(import_w) * dt_hours / 1000
        total_export_kwh = sum(export_w) * dt_hours / 1000
        total_charge_kwh = sum(charge_w) * dt_hours / 1000
        total_discharge_kwh = sum(discharge_w) * dt_hours / 1000

        total_import_cost = sum(p * e * dt_hours / 1000 for p, e in zip(p_import, import_w))
        total_export_revenue = sum(p * e * dt_hours / 1000 for p, e in zip(p_export, export_w))
        total_cost = total_import_cost - total_export_revenue

        avg_import_price = total_import_cost / total_import_kwh if total_import_kwh > 0 else 0
        avg_export_price = total_export_revenue / total_export_kwh if total_export_kwh > 0 else 0

        # Calculate baseline (no battery optimization - direct pass-through)
        baseline_cost = self._calculate_baseline_cost(
            p_import, p_export, solar, load, dt_hours
        )

        return OptimizationResult(
            success=True,
            status="optimal",
            charge_schedule_w=charge_w,
            discharge_schedule_w=discharge_w,
            grid_import_w=import_w,
            grid_export_w=export_w,
            soc_trajectory=soc_values,
            timestamps=timestamps,
            total_cost=total_cost,
            total_import_kwh=total_import_kwh,
            total_export_kwh=total_export_kwh,
            total_charge_kwh=total_charge_kwh,
            total_discharge_kwh=total_discharge_kwh,
            average_import_price=avg_import_price,
            average_export_price=avg_export_price,
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
                # Need to import
                energy_kwh = net_load * dt_hours / 1000
                total_cost += prices_import[t] * energy_kwh
            else:
                # Excess solar - export
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
        """
        Generate a simple heuristic schedule when LP solver is unavailable.

        Strategy: Charge during lowest price periods, discharge during highest.
        """
        n_intervals = len(prices_import)
        dt_hours = cfg.interval_minutes / 60.0
        capacity_wh = cfg.battery_capacity_wh

        # Simple threshold-based strategy
        avg_import = sum(prices_import) / n_intervals
        avg_export = sum(prices_export) / n_intervals

        charge_w = []
        discharge_w = []
        import_w = []
        export_w = []
        soc_values = [initial_soc]

        current_soc = initial_soc

        for t in range(n_intervals):
            solar = solar_forecast[t]
            load = load_forecast[t]
            net_load = load - solar

            charge_power = 0.0
            discharge_power = 0.0

            # Low price - charge if battery not full
            if prices_import[t] < avg_import * 0.7 and current_soc < cfg.max_soc - 0.1:
                available_capacity = (cfg.max_soc - current_soc) * capacity_wh
                max_energy = cfg.max_charge_w * dt_hours
                charge_power = min(cfg.max_charge_w, available_capacity / dt_hours)

            # High export price - discharge if battery not at reserve
            elif prices_export[t] > avg_export * 1.3 and current_soc > cfg.backup_reserve + 0.1:
                available_energy = (current_soc - cfg.backup_reserve) * capacity_wh
                discharge_power = min(cfg.max_discharge_w, available_energy / dt_hours)

            # Update SOC
            energy_in = charge_power * cfg.charge_efficiency * dt_hours
            energy_out = discharge_power / cfg.discharge_efficiency * dt_hours
            current_soc += (energy_in - energy_out) / capacity_wh
            current_soc = max(cfg.backup_reserve, min(cfg.max_soc, current_soc))

            # Calculate grid flows
            net_with_battery = net_load + charge_power - discharge_power
            grid_in = max(0, net_with_battery)
            grid_out = max(0, -net_with_battery)

            charge_w.append(charge_power)
            discharge_w.append(discharge_power)
            import_w.append(grid_in)
            export_w.append(grid_out)
            soc_values.append(current_soc)

        timestamps = [
            start_time + timedelta(minutes=cfg.interval_minutes * i)
            for i in range(n_intervals)
        ]

        # Calculate costs
        total_import = sum(import_w) * dt_hours / 1000
        total_export = sum(export_w) * dt_hours / 1000
        total_charge = sum(charge_w) * dt_hours / 1000
        total_discharge = sum(discharge_w) * dt_hours / 1000

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
            timestamps=timestamps,
            total_cost=total_cost,
            total_import_kwh=total_import,
            total_export_kwh=total_export,
            total_charge_kwh=total_charge,
            total_discharge_kwh=total_discharge,
            baseline_cost=baseline_cost,
            savings=baseline_cost - total_cost,
            solver_name="heuristic",
        )
