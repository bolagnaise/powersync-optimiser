# Changelog

## 1.0.22

- Refactored LP model to explicitly track power flows
- New terminology: "consume" (battery→load) vs "export" (battery→grid)
- Split variables: solar_to_load, solar_to_battery, solar_to_grid, battery_to_load, battery_to_grid, grid_to_load, grid_to_battery
- More accurate penalties: only battery_to_grid is penalized at low export prices, not battery_to_load
- New detailed metrics: total_battery_consume_kwh, total_battery_export_kwh, total_solar_consumed_kwh, total_solar_exported_kwh
- Backward compatible: legacy fields (charge_w, discharge_w) still provided

## 1.0.21

- Fixed profit maximization discharging at $0 export price
- Fixed cost minimization charging at high import prices
- Added discharge penalty when export prices are near-zero (all modes)
- Added charge penalty when import prices are above average (cost mode)

## 1.0.20

- Removed armv7 support (cvxpy/highspy lack pre-built wheels for 32-bit ARM)
- Supported architectures: amd64, aarch64

## 1.0.19

- Fixed array length mismatch in optimization
- Improved logging and error handling

## 1.0.18

- Initial stable release
- CVXPY with HiGHS solver for battery optimization
- Support for cost minimization, profit maximization, and self-consumption modes
- 48-hour optimization horizon
- Flask-based REST API
