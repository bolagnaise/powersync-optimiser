# Changelog

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
