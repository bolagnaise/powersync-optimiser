# PowerSync Optimiser

A Home Assistant add-on that provides ML-based battery optimization for the [PowerSync](https://github.com/bolagnaise/PowerSync) integration.

## Features

- **Linear Programming Optimization** - Uses CVXPY with HiGHS solver for mathematically optimal battery scheduling
- **48-Hour Horizon** - Optimizes battery charge/discharge over a configurable horizon
- **Multiple Cost Functions**:
  - Cost Minimization - Minimize your electricity bill
  - Profit Maximization - Maximize revenue from grid trading
  - Self-Consumption - Maximize use of your solar generation
- **Provider Price Modifications** - Supports export boost, chip mode, and other Amber features
- **Heuristic Fallback** - Simple optimization when LP solver unavailable

## Installation

### Add Repository

1. In Home Assistant, go to **Settings** → **Add-ons** → **Add-on Store**
2. Click the three dots menu (⋮) in the top right
3. Select **Repositories**
4. Add this repository URL:
   ```
   https://github.com/bolagnaise/powersync-optimiser
   ```
5. Click **Add** → **Close**
6. Find "PowerSync Optimiser" in the add-on store and click **Install**

### Configure

The add-on works out of the box with sensible defaults. Optional configuration:

| Option | Default | Description |
|--------|---------|-------------|
| `log_level` | `info` | Logging level (debug, info, warning, error) |
| `optimization_interval` | `30` | Time interval in minutes (15-120) |
| `horizon_hours` | `48` | Optimization horizon in hours (24-72) |
| `default_cost_function` | `cost` | Default objective (cost, profit, self_consumption) |

## How It Works

The add-on runs a Flask HTTP server that the PowerSync integration calls to request optimized battery schedules.

### Optimization Problem

The optimizer solves a linear program to find the optimal battery schedule:

**Objective:** Minimize electricity cost (or maximize profit/self-consumption)

**Subject to:**
- Power balance at each interval
- Battery state-of-charge dynamics
- Charge/discharge power limits
- Minimum backup reserve
- Grid import/export limits

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Optimizer status and configuration |
| `/optimize` | POST | Run battery optimization |
| `/ev/optimize` | POST | EV charging optimization (planned) |
| `/multi-battery/optimize` | POST | Multi-battery optimization (planned) |

## Requirements

- Home Assistant OS or Supervised installation
- PowerSync integration v0.5.0 or later
- Battery system supported by PowerSync (Tesla, Sigenergy, Sungrow)

## Dependencies

The add-on installs these Python packages in its container:
- `cvxpy` - Convex optimization library
- `highspy` - HiGHS LP solver
- `numpy` - Numerical computing
- `flask` - HTTP API server

## Troubleshooting

### Optimizer Not Available

If the optimizer shows as unavailable, check the add-on logs for installation errors. The most common cause is insufficient memory during package installation.

### Slow Optimization

The first optimization after startup may be slower as the solver initializes. Subsequent optimizations typically complete in under 1 second.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Related Projects

- [PowerSync](https://github.com/bolagnaise/PowerSync) - Home Assistant integration for battery management
- [EMHASS](https://github.com/davidusb-geern/emhass) - Energy Management for Home Assistant (inspiration for add-on approach)
